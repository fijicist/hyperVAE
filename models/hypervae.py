"""
Bipartite Hypergraph Variational Autoencoder (HyperVAE) for Jet Generation

This module implements a memory-optimized VAE architecture that combines:
1. Lorentz-equivariant attention (L-GATr) for physics-preserving particle encoding
2. Bipartite graph structure for particle-level and edge/hyperedge features
3. Squared distance Chamfer loss for stable gradient flow

Key Features:
- L-GATr integration for 4-vector (E, px, py, pz) processing
- Squared distance metric to prevent gradient vanishing
- Multi-task learning with particle, edge, hyperedge, and jet-level losses
- KL divergence annealing for stable VAE training
- Auxiliary loss weighting for edge/hyperedge features

Architecture:
- Encoder: BipartiteEncoder with L-GATr particle processing
- Decoder: BipartiteDecoder with L-GATr-based particle generation
- Loss: Weighted combination of reconstruction + KL divergence

Memory Optimization:
- Gradient accumulation for effective large batch sizes
- Mixed precision (FP16) training support
- Efficient PyG batch format handling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .encoder import BipartiteEncoder
from .decoder import BipartiteDecoder
from scipy.stats import wasserstein_distance
import numpy as np


class BipartiteHyperVAE(nn.Module):
    """
    Bipartite Hypergraph Variational Autoencoder for Jet Generation.
    
    This model encodes particle jets as bipartite hypergraphs and learns a latent
    representation using variational inference. The decoder reconstructs particle
    4-vectors, edges (pairwise features), and hyperedges (higher-order correlations).
    
    Architecture Overview:
    1. Encoder: Processes particles → latent distribution q(z|x)
    2. Latent sampling: z ~ q(z|x) via reparameterization trick
    3. Decoder: Generates particles, edges, hyperedges from z
    4. Loss: Reconstruction (Chamfer) + KL divergence + auxiliary losses
    
    Key Innovation:
    - Uses L-GATr (Lorentz Group Attention) for physics-preserving transformations
    - Squared distance Chamfer loss prevents gradient vanishing (∇d² = 2x vs ∇√d² = 1/(2√x))
    - Multi-scale loss: particle-level + edge-level + jet-level features
    
    Input Format (PyTorch Geometric Data):
    - particle_x: [N_total_particles, 4] - particle 4-vectors (E, px, py, pz)
    - edge_index: [2, N_edges] - bipartite connectivity
    - edge_attr: [N_edges, 5] - edge features (ln_delta, ln_kt, etc.)
    - hyperedge_x: [N_hyperedges, 2] - hyperedge features (3pt_eec, 4pt_eec)
    - y: [batch_size, 4] - jet features (jet_type, jet_pt, jet_eta, jet_mass)
    - n_particles: [batch_size] - number of particles per jet
    
    Output:
    - Reconstructed particles, edges, hyperedges
    - Latent distribution parameters (mu, logvar)
    - Topology information (masks, multiplicities)
    """
    
    def __init__(self, config):
        """
        Initialize BipartiteHyperVAE model.
        
        Args:
            config: Dictionary with configuration parameters:
                - config['model']: Model architecture settings (encoder/decoder configs)
                - config['training']: Training settings (loss weights, hyperparameters)
                
        Key Config Sections:
        - model.encoder: L-GATr encoder settings (in_mv, out_mv, out_s, num_blocks)
        - model.decoder: L-GATr decoder settings (in_mv, out_mv, in_s, num_blocks)
        - training.loss_config: Distance metric, weights, indices
        - training.loss_weights: Relative importance of each loss term
        """
        super().__init__()
        self.config = config
        
        # Core VAE components
        self.encoder = BipartiteEncoder(config['model'])
        self.decoder = BipartiteDecoder(config['model'])
        
        # Loss weights (relative importance of each loss component)
        # Example: {'particle_features': 1.0, 'edge_features': 0.5, 'kl_divergence': 0.1}
        self.loss_weights = config['training']['loss_weights']
        
        # === Distance Metric Configuration ===
        # Defines how particle similarity is measured in Chamfer loss
        # === Distance Metric Configuration ===
        # Defines how particle similarity is measured in Chamfer loss
        self.loss_config = config['training'].get('loss_config', {})
        self.particle_loss_type = self.loss_config.get('particle_loss_type', 'chamfer')
        self.particle_distance_metric = self.loss_config.get('particle_distance_metric', 'rapidity_phi')
        
        # === Feature Indices for Different Distance Metrics ===
        # For 'rapidity_phi' metric (physics-motivated coordinates)
        self.log_pt_index = self.loss_config.get('log_pt_index', 0)
        self.eta_index = self.loss_config.get('eta_index', 1)
        self.phi_index = self.loss_config.get('phi_index', 2)
        self.log_E_index = self.loss_config.get('log_E_index', 3)
        
        # For 'euclidean_4d' metric (4-vector distance used with L-GATr)
        # Indices point to [E, px, py, pz] in particle feature tensor
        self.E_index = self.loss_config.get('E_index', 0)
        self.px_index = self.loss_config.get('px_index', 1)
        self.py_index = self.loss_config.get('py_index', 2)
        self.pz_index = self.loss_config.get('pz_index', 3)
        
        # === Distance Metric Weights ===
        # For 'rapidity_phi': weights angular vs momentum components
        self.w_angular = self.loss_config.get('w_angular', 1.0)
        self.w_momentum = self.loss_config.get('w_momentum', 1.0)
        
        # === pT-weighted Chamfer Loss ===
        # Option to weight particle matching by transverse momentum (high-pT particles more important)
        self.use_pt_weighting = self.loss_config.get('use_pt_weighting', True)
        self.pt_weight_alpha = self.loss_config.get('pt_weight_alpha', 1.0)  # α in w_i = pt_i^α
        
        # === Hyperbolic Chamfer Loss ===
        # Hyperparameter for hyperbolic transformation: d_hyp = arcosh(1 + α * d²)
        # Controls curvature of hyperbolic space; higher α → stronger hyperbolic effect
        self.hyperbolic_alpha = self.loss_config.get('hyperbolic_alpha', 1.0)
        
        # === Squared vs Euclidean Distance ===
        # CRITICAL: Prevents gradient vanishing for far predictions
        # Squared: ∇d² = 2x (linear, strong gradients)
        # Euclidean: ∇√d² = 1/(2√x) (vanishes for large x)
        # Recommendation: use_squared_distance=True for normalized features
        self.use_squared_distance = self.loss_config.get('use_squared_distance', False)
        
        # Validation/evaluation metric (can differ from training loss)
        self.evaluation_metric = self.loss_config.get('evaluation_metric', 'wasserstein')
        
        # === Jet-Level Feature Configuration ===
        # Indices into y tensor: [jet_type, jet_pt, jet_eta, jet_mass, ...]
        self.jet_pt_index = self.loss_config.get('jet_pt_index', 1)
        self.jet_eta_index = self.loss_config.get('jet_eta_index', 2)
        self.jet_mass_index = self.loss_config.get('jet_mass_index', 3)
        
        # Weights for jet-level features in loss (can emphasize physics-critical features)
        self.jet_pt_weight = self.loss_config.get('jet_pt_weight', 1.0)
        self.jet_eta_weight = self.loss_config.get('jet_eta_weight', 0.5)
        self.jet_mass_weight = self.loss_config.get('jet_mass_weight', 2.0)  # Often most important
        self.jet_n_constituents_weight = self.loss_config.get('jet_n_constituents_weight', 1.0)
        
        # === Auxiliary Loss Configuration ===
        # Edges/hyperedges are auxiliary: weighted lower than main particle loss
        self.use_auxiliary_losses = config['model'].get('use_auxiliary_losses', True)
        self.auxiliary_loss_weight = config['model'].get('auxiliary_loss_weight', 0.1)  # Typically 0.1
        
        # === Local→Global Physics Consistency Loss ===
        # Enforces agreement between particle 4-momenta and jet-level observables
        self.use_consistency_loss = self.loss_config.get('use_consistency_loss', True)
        self.consistency_pt_weight = self.loss_config.get('consistency_pt_weight', 2.0)
        self.consistency_eta_weight = self.loss_config.get('consistency_eta_weight', 1.0)
        self.consistency_mass_weight = self.loss_config.get('consistency_mass_weight', 2.5)
        
        # === Regularization ===
        # Latent noise during training (variational dropout for robustness)
        self.latent_noise_std = config['training'].get('latent_noise', 0.0)
    
    def reparameterize(self, mu, logvar, add_noise=True):
        """
        VAE Reparameterization Trick: z = μ + σ * ε, where ε ~ N(0,1)
        
        This allows backpropagation through stochastic sampling by treating
        randomness as an external input (ε) rather than part of the computation graph.
        
        Mathematical Formulation:
            z ~ q(z|x) = N(μ, σ²)
            z = μ + σ * ε, where ε ~ N(0, 1)
            σ = exp(0.5 * logvar)
        
        Optional Regularization:
            During training, can add extra noise for variational dropout:
            z_final = z + noise, where noise ~ N(0, latent_noise_std²)
        
        Args:
            mu: [batch_size, latent_dim] - Mean of latent distribution from encoder
            logvar: [batch_size, latent_dim] - Log variance of latent distribution
            add_noise: If True and training, add extra noise for regularization
        
        Returns:
            z: [batch_size, latent_dim] - Sampled latent codes
        
        Note:
            We use log variance instead of variance directly for numerical stability.
            This prevents variance from becoming negative during training.
        """
        # Convert log variance to standard deviation
        std = torch.exp(0.5 * logvar)
        
        # Sample from standard normal
        eps = torch.randn_like(std)
        
        # Reparameterization: z = μ + σ * ε
        z = mu + eps * std
        
        # Optional: Add extra noise during training for regularization (variational dropout)
        # This can improve generalization by preventing overfitting to specific latent codes
        if add_noise and self.training and self.latent_noise_std > 0:
            noise = torch.randn_like(z) * self.latent_noise_std
            z = z + noise
        
        return z
    
    def forward(self, batch, temperature=1.0, generate_all_features=None):
        """
        Full forward pass through VAE: Encode → Sample → Decode.
        
        Pipeline:
        1. Encoder: particles → latent distribution q(z|x) = N(μ, σ²)
        2. Sample: z ~ q(z|x) via reparameterization trick
        3. Decoder: z → reconstructed particles, edges, hyperedges
        
        Training vs Inference:
        - Training: Generate all features (particles + edges + hyperedges)
        - Inference: Only generate particles (edges/hyperedges optional)
        
        Args:
            batch: PyTorch Geometric Batch object containing:
                - particle_x: [N_total_particles, 4] - particle 4-vectors
                - edge_index: [2, N_edges] - bipartite connectivity
                - edge_attr: [N_edges, 5] - edge features
                - hyperedge_x: [N_hyperedges, 2] - hyperedge features
                - jet_type: [batch_size] - jet type labels (0=quark, 1=gluon, 2=top)
                - y: [batch_size, 4] - jet features [jet_type, jet_pt, jet_eta, jet_mass]
                - n_particles: [batch_size] - number of particles per jet
                
            temperature: float - Gumbel-softmax temperature for topology generation
                - High temp (>1): Soft assignments (training start)
                - Low temp (<1): Sharp assignments (training end)
                - Typically annealed from 1.0 to 0.5 over training
                
            generate_all_features: bool or None
                - None: Auto-detect (True for training, False for eval)
                - True: Generate edges/hyperedges (needed for auxiliary losses)
                - False: Only particles (faster inference)
        
        Returns:
            dict with keys:
                - 'particle_features': [batch_size, max_particles, 4] - reconstructed particles
                - 'edge_features': [batch_size, max_edges, 5] - reconstructed edges (if enabled)
                - 'hyperedge_features': [batch_size, max_hyperedges, 2] - reconstructed hyperedges (if enabled)
                - 'jet_features': [batch_size, 3] - reconstructed jet features (jet_pt, jet_eta, jet_mass)
                - 'topology': dict with masks and counts
                - 'mu': [batch_size, latent_dim] - latent means
                - 'logvar': [batch_size, latent_dim] - latent log variances
                - 'z': [batch_size, latent_dim] - sampled latent codes
        
        Example:
            >>> model = BipartiteHyperVAE(config)
            >>> output = model(batch, temperature=0.7)
            >>> reconstructed_particles = output['particle_features']
            >>> latent_codes = output['z']
        """
        # === ENCODING PHASE ===
        # Encode particles to latent distribution parameters
        mu, logvar = self.encoder(batch)
        
        # === SAMPLING PHASE ===
        # Sample latent code via reparameterization trick
        z = self.reparameterize(mu, logvar)
        
        # === DECODING PHASE ===
        # Determine what features to generate (particles always, edges/hyperedges optional)
        if generate_all_features is None:
            # Auto-detect: training needs all features, inference can skip auxiliary
            generate_all_features = self.training
        
        # Generate reconstructions from latent code
        # During training: generate all features for auxiliary losses
        # During inference: only generate particles (faster, sufficient for evaluation)
        output = self.decoder(
            z, 
            batch.jet_type,  # Condition on jet type (quark/gluon/top)
            temperature,     # Controls sharpness of topology assignments
            generate_edges=generate_all_features and self.use_auxiliary_losses,
            generate_hyperedges=generate_all_features and self.use_auxiliary_losses
        )
        
        # === ATTACH LATENT PARAMETERS ===
        # Include latent distribution parameters for KL divergence computation
        output['mu'] = mu
        output['logvar'] = logvar
        output['z'] = z
        
        return output
    
    def compute_loss(self, batch, output, epoch=0):
        """
        Compute total VAE loss: Reconstruction + KL Divergence + Auxiliary Losses + Physics Consistency.
        
        Loss Formulation:
            L_total = L_particle + w_aux * (L_edge + L_hyperedge) + w_jet * L_jet + 
                      w_cons * L_consistency + β(epoch) * L_KL
        
        Where:
        - L_particle: Chamfer distance between true and predicted particles (MAIN LOSS)
        - L_edge: MSE for pairwise edge features (ln_delta, ln_kt, etc.)
        - L_hyperedge: MSE for higher-order features (3pt_eec, 4pt_eec)
        - L_jet: Weighted MSE for jet-level features (jet_pt, jet_eta, jet_mass, n_constituents)
        - L_consistency: Local→global physics consistency (particles sum to jet observables)
        - L_KL: KL divergence between q(z|x) and p(z)=N(0,I)
        - β(epoch): KL annealing weight (gradually increases from 0 to 1)
        
        Loss Weights:
        - particle_features: 12000.0 (main objective)
        - edge_features: 2500.0 * w_aux (auxiliary)
        - hyperedge_features: 1500.0 * w_aux (auxiliary)
        - jet_features: 6000.0 (constrain jet-level properties)
        - local_global_consistency: 3000.0 (physics constraint)
        - kl_divergence: 0.3 * β(epoch) (annealed regularization)
        
        Args:
            batch: PyG Batch with ground truth data
            output: Dict from forward() with predictions
            epoch: Current epoch number (for KL annealing)
        
        Returns:
            dict with individual and total losses:
                - 'particle': Particle reconstruction loss
                - 'edge': Edge reconstruction loss (0 if disabled)
                - 'hyperedge': Hyperedge reconstruction loss (0 if disabled)
                - 'jet': Jet feature loss
                - 'consistency': Local→global physics consistency loss
                - 'kl': KL divergence
                - 'kl_weight': Current KL annealing weight β(epoch)
                - 'kl_raw': Raw KL before weighting (for monitoring)
                - 'total': Weighted sum of all losses
        
        Note:
            Auxiliary losses (edges/hyperedges) only computed during training.
            This saves computation during validation/inference.
        """
        # Fix batch.y shape if needed (PyG concatenates [4] tensors into [batch_size*4])
        if batch.y.dim() == 1 and hasattr(batch, 'num_graphs'):
            batch.y = batch.y.view(batch.num_graphs, -1)  # [batch_size, 4]
        
        losses = {}
        
        # === 1. PARTICLE RECONSTRUCTION LOSS (MAIN LOSS) ===
        # Chamfer distance: Permutation-invariant set matching
        # Measures how well predicted particle set matches true particle set
        particle_loss = self._particle_reconstruction_loss(
            batch, output['particle_features'], output['topology']['particle_mask']
        )
        losses['particle'] = particle_loss
        
        # === 2. AUXILIARY LOSSES (EDGES & HYPEREDGES) ===
        # Only computed during training to guide learning of particle relationships
        # Disabled during validation/inference for speed
        if self.training and self.use_auxiliary_losses:
            # Edge feature auxiliary loss
            if output.get('edge_features') is not None and batch.edge_attr.size(0) > 0:
                edge_loss = self._edge_reconstruction_loss(batch, output['edge_features'])
                losses['edge'] = edge_loss
            else:
                losses['edge'] = torch.tensor(0.0, device=batch.x.device)
            
            # Hyperedge feature auxiliary loss
            if output.get('hyperedge_features') is not None:
                hyperedge_loss = self._hyperedge_reconstruction_loss(
                    batch, output['hyperedge_features'], output['topology']['hyperedge_mask']
                )
                losses['hyperedge'] = hyperedge_loss
            else:
                losses['hyperedge'] = torch.tensor(0.0, device=batch.x.device)
        else:
            losses['edge'] = torch.tensor(0.0, device=batch.x.device)
            losses['hyperedge'] = torch.tensor(0.0, device=batch.x.device)
        
        # 4. Jet feature loss (ALWAYS computed - includes N_constituents)
        # Pass topology information for N_constituents prediction
        jet_loss = self._jet_feature_loss(batch, output['jet_features'], output['topology'])
        losses['jet'] = jet_loss
        
        # 5. Local→Global Physics Consistency Loss
        # Enforces that generated particles sum to correct jet-level observables
        if self.use_consistency_loss:
            consistency_loss = self._local_global_consistency_loss(
                batch, output['particle_features'], output['topology']['particle_mask']
            )
            losses['consistency'] = consistency_loss
        else:
            losses['consistency'] = torch.tensor(0.0, device=batch.x.device)
        
        # 6. KL divergence
        kl_loss = self._kl_divergence(output['mu'], output['logvar'])
        
        # KL annealing with cyclical schedule support
        kl_annealing_schedule = self.config['training'].get('kl_annealing_schedule', 'linear')
        kl_warmup_epochs = self.config['training'].get('kl_warmup_epochs', 100)
        kl_max_weight = self.config['training'].get('kl_max_weight', 1.0)
        
        if kl_annealing_schedule == 'cyclical':
            # Cyclical annealing: prevents posterior collapse and KL explosion
            # Cycle: [warmup -> plateau] repeated throughout training
            kl_cycle_epochs = self.config['training'].get('kl_cycle_epochs', 50)
            epoch_in_cycle = epoch % kl_cycle_epochs
            
            if epoch_in_cycle < kl_warmup_epochs:
                # Warmup phase: gradually increase from 0 to kl_max_weight
                kl_weight = kl_max_weight * (epoch_in_cycle / kl_warmup_epochs)
            else:
                # Plateau phase: maintain kl_max_weight
                kl_weight = kl_max_weight
        else:
            # Linear annealing (original): increases once then plateaus
            kl_weight = min(kl_max_weight, epoch / kl_warmup_epochs) if kl_warmup_epochs > 0 else kl_max_weight
        
        losses['kl'] = kl_loss
        losses['kl_raw'] = kl_loss.item()  # Log raw KL for monitoring
        
        # Total loss with auxiliary weight
        # Auxiliary losses are weighted by auxiliary_loss_weight (typically 0.1)
        # Note: topology loss removed - N_constituents now part of jet loss
        total_loss = (
            self.loss_weights['particle_features'] * losses['particle'] +
            self.auxiliary_loss_weight * self.loss_weights['edge_features'] * losses['edge'] +
            self.auxiliary_loss_weight * self.loss_weights['hyperedge_features'] * losses['hyperedge'] +
            self.loss_weights['jet_features'] * losses['jet'] +
            self.loss_weights.get('local_global_consistency', 0.0) * losses['consistency'] +
            self.loss_weights['kl_divergence'] * kl_weight * losses['kl']
        )
        
        losses['total'] = total_loss
        losses['kl_weight'] = kl_weight
        
        return losses
    
    def _particle_reconstruction_loss(self, batch, pred_features, pred_mask):
        """
        Particle reconstruction loss - supports both Chamfer Distance and MSE.
        
        Training: Uses Chamfer Distance (permutation-invariant)
        Evaluation: Configurable via evaluation_metric
        """
        if self.training:
            # Training mode: use Chamfer Distance
            if self.particle_loss_type == 'chamfer':
                return self._chamfer_distance_loss(batch, pred_features, pred_mask)
            elif self.particle_loss_type == 'hyperbolic_chamfer':
                return self._hyperbolic_chamfer_distance_loss(batch, pred_features, pred_mask)
            else:
                return self._mse_particle_loss(batch, pred_features, pred_mask)
        else:
            # Evaluation mode: use configured metric
            if self.evaluation_metric == 'wasserstein':
                return self._wasserstein_particle_loss(batch, pred_features, pred_mask)
            elif self.evaluation_metric == 'chamfer':
                return self._chamfer_distance_loss(batch, pred_features, pred_mask)
            elif self.evaluation_metric == 'hyperbolic_chamfer':
                return self._hyperbolic_chamfer_distance_loss(batch, pred_features, pred_mask)
            else:
                return self._mse_particle_loss(batch, pred_features, pred_mask)
    
    def _compute_particle_distance(self, particles1, particles2):
        """
        Compute pairwise distance between particles using configured metric.
        
        Available metrics:
        
        1. 'rapidity_phi': Physics-motivated distance with angular and momentum components
           D(i,j) = sqrt[w_angular × [(Δη)² + (Δφ_wrapped)²] + w_momentum × [(Δlog_pt)² + (Δlog_E)²]]
        
        2. 'euclidean_4d': W4D Euclidean distance for 4-vectors [E, px, py, pz]
           D(i,j) = sqrt[(E_i-E_j)² + (px_i-px_j)² + (py_i-py_j)² + (pz_i-pz_j)²]
           All positive signs (Euclidean metric)
        
        3. default: Standard Euclidean distance
        
        Args:
            particles1: [N1, num_features] particle features
            particles2: [N2, num_features] particle features
        
        Returns:
            distances: [N1, N2] pairwise distances
        """
        if self.particle_distance_metric == 'rapidity_phi':
            # Extract features
            log_pt1 = particles1[:, self.log_pt_index].unsqueeze(1)  # [N1, 1]
            eta1 = particles1[:, self.eta_index].unsqueeze(1)        # [N1, 1]
            phi1 = particles1[:, self.phi_index].unsqueeze(1)        # [N1, 1]
            log_E1 = particles1[:, self.log_E_index].unsqueeze(1)    # [N1, 1]
            
            log_pt2 = particles2[:, self.log_pt_index].unsqueeze(0)  # [1, N2]
            eta2 = particles2[:, self.eta_index].unsqueeze(0)        # [1, N2]
            phi2 = particles2[:, self.phi_index].unsqueeze(0)        # [1, N2]
            log_E2 = particles2[:, self.log_E_index].unsqueeze(0)    # [1, N2]
            
            # Angular component: (Δη)² + (Δφ_wrapped)²
            delta_eta = eta1 - eta2  # [N1, N2]
            delta_eta_sq = delta_eta ** 2
            
            # Compute wrapped phi distance (accounting for periodicity at ±π)
            delta_phi_abs = torch.abs(phi1 - phi2)  # [N1, N2]
            delta_phi = torch.where(
                delta_phi_abs <= torch.pi,
                delta_phi_abs,
                2 * torch.pi - delta_phi_abs
            )
            delta_phi_sq = delta_phi ** 2
            
            angular_dist_sq = delta_eta_sq + delta_phi_sq
            
            # Momentum component: (Δlog_pt)² + (Δlog_E)²
            delta_log_pt = log_pt1 - log_pt2  # [N1, N2]
            delta_log_E = log_E1 - log_E2    # [N1, N2]
            
            momentum_dist_sq = delta_log_pt ** 2 + delta_log_E ** 2
            
            # Combined weighted distance
            dist = torch.sqrt(
                self.w_angular * angular_dist_sq + 
                self.w_momentum * momentum_dist_sq + 
                1e-8
            )  # [N1, N2]
            return dist
        elif self.particle_distance_metric == 'euclidean_4d':
            # =====================================================================
            # EUCLIDEAN 4D DISTANCE FOR L-GATr 4-VECTORS [E, px, py, pz]
            # =====================================================================
            # Mathematical Form:
            #   D²(i,j) = (E_i-E_j)² + (px_i-px_j)² + (py_i-py_j)² + (pz_i-pz_j)²
            #   D(i,j) = √D²(i,j)  OR  D(i,j) = D²(i,j) if use_squared_distance=True
            # 
            # Key Design Decisions:
            # 1. ALL POSITIVE SIGNS (Euclidean, not Minkowski metric)
            #    - Minkowski: m² = E² - p² (can be negative)
            #    - Euclidean: always positive, suitable for distance metrics
            # 
            # 2. EQUAL WEIGHTING FOR ALL COMPONENTS
            #    - All components (E, px, py, pz) treated equally with weight 1.0
            #    - Standard Euclidean distance in 4D space
            # 
            # 3. SQUARED VS EUCLIDEAN DISTANCE (controlled by use_squared_distance)
            #    ┌──────────────┬──────────────┬────────────────────────┐
            #    │   Metric     │  Formula     │     Gradient           │
            #    ├──────────────┼──────────────┼────────────────────────┤
            #    │  Squared     │  D² = x²     │  ∂D²/∂x = 2x          │
            #    │  Euclidean   │  D = √(x²)   │  ∂D/∂x = x/√(x²)      │
            #    └──────────────┴──────────────┴────────────────────────┘
            # 
            #    Why Squared Distance?
            #    - STRONGER GRADIENTS: For far predictions (x=10), gradient is 20 vs 0.05
            #    - NO VANISHING: Gradient grows linearly with error (2x), not inverse (1/2√x)
            #    - STABLE TRAINING: Prevents plateau where loss barely decreases
            #    - CONSISTENT: Same metric used for training AND validation
            # 
            #    When to use Euclidean?
            #    - Literature comparison (Euclidean is standard)
            #    - After model converges (squared brings it close, Euclidean refines)
            #    - If training is already stable
            # 
            # 4. NORMALIZED FEATURES
            #    - Input 4-vectors are Z-score normalized: (x - μ) / σ
            #    - Typical range: [-3, 3] standard deviations
            #    - This makes distance scale interpretable and prevents numeric issues
            # =====================================================================
            
            # Extract 4-vector components with broadcasting for pairwise distances
            # Shape transformation: [N, 4] → [N, 1] (particles1) and [N, 4] → [1, N] (particles2)
            # This creates [N1, N2] pairwise difference matrices via broadcasting
            E1 = particles1[:, self.E_index].unsqueeze(1)      # [N1, 1]
            px1 = particles1[:, self.px_index].unsqueeze(1)    # [N1, 1]
            py1 = particles1[:, self.py_index].unsqueeze(1)    # [N1, 1]
            pz1 = particles1[:, self.pz_index].unsqueeze(1)    # [N1, 1]
            
            E2 = particles2[:, self.E_index].unsqueeze(0)      # [1, N2]
            px2 = particles2[:, self.px_index].unsqueeze(0)    # [1, N2]
            py2 = particles2[:, self.py_index].unsqueeze(0)    # [1, N2]
            pz2 = particles2[:, self.pz_index].unsqueeze(0)    # [1, N2]
            
            # Compute squared differences for each component
            # Broadcasting: [N1,1] - [1,N2] → [N1,N2]
            delta_E_sq = (E1 - E2) ** 2      # [N1, N2] - Energy component
            delta_px_sq = (px1 - px2) ** 2   # [N1, N2] - x-momentum component
            delta_py_sq = (py1 - py2) ** 2   # [N1, N2] - y-momentum component
            delta_pz_sq = (pz1 - pz2) ** 2   # [N1, N2] - z-momentum component
            
            # Compute unweighted squared distance (all components treated equally)
            dist_sq = delta_E_sq + delta_px_sq + delta_py_sq + delta_pz_sq
            
            # Squared vs Euclidean Distance ===
            if self.use_squared_distance:
                # SQUARED DISTANCE: D²(i,j) = Σ (x_i - x_j)²
                dist = torch.clamp(dist_sq, min=1e-8)  # Clamp for numerical stability
            else:
                # EUCLIDEAN DISTANCE: D(i,j) = √[Σ (x_i - x_j)²]
                dist = torch.sqrt(torch.clamp(dist_sq, min=1e-8))  # sqrt after clamp for stability
            
            return dist
        else:
            # Euclidean distance in full feature space
            dist = torch.cdist(particles1, particles2, p=2)
            return dist
    
    def _chamfer_distance_loss(self, batch, pred_features, pred_mask):
        """
        Compute pT-weighted bidirectional Chamfer Distance for particle reconstruction.
        
        ═══════════════════════════════════════════════════════════════════════
        CHAMFER DISTANCE - ASYMMETRIC MATCHING FOR SET GENERATION
        ═══════════════════════════════════════════════════════════════════════
        
        Mathematical Formulation:
        -------------------------
        For each jet k, compute bidirectional matching loss:
        
            L_chamfer(k) = L_true→pred(k) + L_pred→true(k)
        
        where:
            L_true→pred = Σ_{i∈true} w_i × min_{j∈pred} D(true_i, pred_j)
            L_pred→true = Σ_{j∈pred} w_j × min_{i∈true} D(pred_j, true_i)
        
        and weights are:
            w_i = (pT_i)^α / Σ_k (pT_k)^α    (normalized pT^α weights)
        
        Why Chamfer Distance?
        ---------------------
        1. PERMUTATION INVARIANT: Order of particles doesn't matter
           - Matches each particle to its nearest neighbor
           - Natural for set generation (jets are unordered particle sets)
        
        2. HANDLES VARIABLE CARDINALITY: Works with different numbers of particles
           - True jet: n_true particles
           - Predicted jet: n_pred particles (can differ!)
           - No need for padding or masking tricks
        
        3. BIDIRECTIONAL MATCHING: Prevents mode collapse
           - true→pred: Ensures ALL true particles are covered
           - pred→true: Penalizes spurious/hallucinated particles
        
        4. pT WEIGHTING: Emphasizes high-energy particles
           - High pT particles (jets cores) contribute more to loss
           - Low pT particles (soft radiation) contribute less
           - Physics-motivated: pT dominates jet observables
        
        Gradient Flow:
        --------------
        For particle i matched to nearest neighbor j:
            ∂L/∂pred_j = w_i × ∂D(true_i, pred_j)/∂pred_j
        
        With squared distance (recommended):
            ∂D²/∂pred_j = 2(pred_j - true_i)
        
        This gradient:
        - Pulls pred_j toward true_i (attractive force)
        - Magnitude proportional to w_i (pT weighting)
        - Magnitude proportional to distance (stronger for far particles)
        
        Comparison to Alternatives:
        ---------------------------
        1. MSE Loss: Requires same cardinality, sensitive to ordering
        2. Hungarian Matching: O(n³) complexity, non-differentiable
        3. EMD (Wasserstein): More expensive, similar performance
        4. Chamfer: O(n²) complexity, differentiable, handles variable n
        
        Implementation Notes:
        ---------------------
        - Processes each jet independently (different n_true, n_pred per jet)
        - Handles Gumbel-Softmax masks during training (soft → hard threshold)
        - Fallback to top-k if no particles pass threshold (prevents division by zero)
        - Computes pT weights via exp(α × log_pT) = pT^α (numerically stable)
        - Normalizes weights per jet (Σw_i = 1 for each jet)
        
        Args:
            batch: PyG Batch with:
                - particle_x: [N_particles_total, num_features] true particle features
                - n_particles: [batch_size] number of particles per jet
                - num_graphs: batch size
            pred_features: [batch_size, max_particles, num_features] predicted particles
            pred_mask: [batch_size, max_particles] validity mask (0-1 during training, bool at eval)
        
        Returns:
            torch.Tensor: Scalar average Chamfer loss across all jets in batch
        
        Shape Conventions:
            N_particles_total: Sum of n_particles across batch (variable)
            max_particles: Maximum particles per jet in this batch
            num_features: Particle feature dimension (e.g., 4 for [E, px, py, pz])
        """
        true_features = batch.x  # [N_particles_total, num_features]
        
        batch_size = batch.num_graphs
        total_loss = 0.0
        valid_count = 0
        
        cumulative_particles = 0
        for i in range(batch_size):
            n_true = batch.n_particles[i].item()
            if n_true == 0:
                continue
            
            # Extract true particles for this jet
            true_particles = true_features[cumulative_particles:cumulative_particles + n_true]  # [n_true, F]
            
            # Extract predicted particles using mask
            if pred_mask is not None:
                # Threshold mask properly (important for Gumbel-Softmax during training)
                # pred_mask may contain soft values [0, 1] during training, need to threshold
                valid_pred_mask = (pred_mask[i] > 0.5)  # Boolean mask with proper thresholding
                
                # Count valid predictions
                n_pred = valid_pred_mask.sum().item()
                
                if n_pred == 0:
                    # No particles passed threshold - use top-k particles based on mask values
                    # This prevents validation loss from being 0
                    k = min(n_true, pred_mask[i].size(0))
                    _, top_indices = torch.topk(pred_mask[i], k)
                    valid_pred_mask = torch.zeros_like(pred_mask[i], dtype=torch.bool)
                    valid_pred_mask[top_indices] = True
                
                pred_particles = pred_features[i][valid_pred_mask]  # [n_pred, F]
            else:
                # Fallback: assume first n_true are valid (training behavior)
                pred_particles = pred_features[i, :n_true]  # [n_true, F]
            
            # Need at least one predicted particle
            if pred_particles.shape[0] == 0:
                cumulative_particles += n_true
                continue
            
            # Compute pairwise distances [n_true, n_pred]
            distances = self._compute_particle_distance(true_particles, pred_particles)
            
            if self.use_pt_weighting:
                # Compute pT-based weights: w_i = (pT_i)^α / Σ (pT_k)^α
                # pT = sqrt(px² + py²) computed from denormalized momenta
                
                # Get normalization stats
                px_stats = batch.particle_norm_stats['px']
                py_stats = batch.particle_norm_stats['py']
                
                # Denormalize px and py efficiently based on normalization method
                if px_stats['method'] == 'zscore':
                    # Z-score: x = x_norm * std + mean
                    true_px = true_particles[:, self.px_index] * px_stats['std'] + px_stats['mean']
                    true_py = true_particles[:, self.py_index] * py_stats['std'] + py_stats['mean']
                    pred_px = pred_particles[:, self.px_index] * px_stats['std'] + px_stats['mean']
                    pred_py = pred_particles[:, self.py_index] * py_stats['std'] + py_stats['mean']
                else:  # minmax
                    # Min-max: x = x_norm * (max - min) + min
                    px_range = px_stats['max'] - px_stats['min']
                    py_range = py_stats['max'] - py_stats['min']
                    true_px = true_particles[:, self.px_index] * px_range + px_stats['min']
                    true_py = true_particles[:, self.py_index] * py_range + py_stats['min']
                    pred_px = pred_particles[:, self.px_index] * px_range + px_stats['min']
                    pred_py = pred_particles[:, self.py_index] * py_range + py_stats['min']
                
                # Compute pT = sqrt(px² + py²)
                true_pt = torch.sqrt(true_px ** 2 + true_py ** 2 + 1e-8)  # [n_true]
                pred_pt = torch.sqrt(pred_px ** 2 + pred_py ** 2 + 1e-8)  # [n_pred]
                
                # Compute normalized weights: w_i = (pT_i)^α / Σ (pT_k)^α
                true_weights = true_pt ** self.pt_weight_alpha  # [n_true]
                true_weights = true_weights / (true_weights.sum() + 1e-8)
                
                pred_weights = pred_pt ** self.pt_weight_alpha  # [n_pred]
                pred_weights = pred_weights / (pred_weights.sum() + 1e-8)
                
                # Weighted Chamfer Distance:
                # 1. For each true particle, find nearest predicted particle, weight by true pT
                true_to_pred_min = distances.min(dim=1)[0]  # [n_true]
                true_to_pred = (true_weights * true_to_pred_min).sum()
                
                # 2. For each predicted particle, find nearest true particle, weight by pred pT
                pred_to_true_min = distances.min(dim=0)[0]  # [n_pred]
                pred_to_true = (pred_weights * pred_to_true_min).sum()
            else:
                # Standard unweighted Chamfer distance
                true_to_pred = distances.min(dim=1)[0].mean()
                pred_to_true = distances.min(dim=0)[0].mean()
            
            # Chamfer distance is sum of both directions
            batch_loss = true_to_pred + pred_to_true
            
            # Check for NaN
            if not torch.isnan(batch_loss):
                total_loss += batch_loss
                valid_count += 1
            
            cumulative_particles += n_true
        
        return total_loss / max(valid_count, 1)
    
    def _hyperbolic_chamfer_distance_loss(self, batch, pred_features, pred_mask):
        """
        Hyperbolic Chamfer Distance for particle reconstruction.
        
        Uses hyperbolic geometry transformation to map Euclidean distances to hyperbolic space.
        This has been shown to better capture hierarchical and physical structures in particle jets
        (see paper: arXiv:2412.17951).
        
        Mathematical Formulation:
        -------------------------
        1. Compute Euclidean distance between particles:
           d_E(p_i, p_j) = ||p_i - p_j||  (using configured metric: rapidity_phi or euclidean_4d)
        
        2. Transform to hyperbolic distance:
           d_H(p_i, p_j) = arcosh(1 + α × d_E²(p_i, p_j))
           
           where α (self.hyperbolic_alpha) controls the curvature:
           - α → 0: recovers standard Euclidean (Chamfer) distance
           - α > 0: stronger hyperbolic effect, emphasizes larger distances
        
        3. Bidirectional matching (same as standard Chamfer):
           L = L_true→pred + L_pred→true
           
           L_true→pred = Σ_i w_i × min_j d_H(true_i, pred_j)
           L_pred→true = Σ_j w_j × min_i d_H(pred_j, true_i)
           
           where w_i are pT-based weights (if use_pt_weighting=True):
           w_i = (pT_i)^α / Σ_k (pT_k)^α
        
        Key Properties:
        ---------------
        - Permutation invariant (set-valued matching)
        - Handles variable cardinality (n_true ≠ n_pred)
        - Compatible with pT-weighting
        - Differentiable everywhere (arcosh is smooth for x > 1)
        - Emphasizes large distances more than standard Chamfer
        
        Args:
            batch: PyG Batch with true particle features (batch.x)
            pred_features: [batch_size, max_particles, F] predicted particle features
            pred_mask: [batch_size, max_particles] binary mask for valid predictions
        
        Returns:
            torch.Tensor: Scalar hyperbolic Chamfer loss averaged over batch
        
        Implementation Notes:
        --------------------
        - For numerical stability: arcosh(x) = log(x + sqrt(x² - 1)) with clipping x > 1+ε
        - pT-weighting uses log_pt features: pt^α = exp(α × log_pt)
        - Processes jets independently (no inter-jet matching)
        - Skips jets with zero particles or invalid predictions
        """
        true_features = batch.x
        
        batch_size = batch.num_graphs
        total_loss = 0.0
        valid_count = 0
        
        cumulative_particles = 0
        for i in range(batch_size):
            n_true = batch.n_particles[i].item()
            if n_true == 0:
                continue
            
            # Extract true particles for this jet
            true_particles = true_features[cumulative_particles:cumulative_particles + n_true]
            
            # Extract predicted particles using mask
            if pred_mask is not None:
                if not self.training:
                    # Validation/test: use binary mask with threshold
                    valid_pred_mask = (pred_mask[i] > 0.5)
                    
                    n_pred = valid_pred_mask.sum().item()
                    if n_pred == 0:
                        # No particles passed threshold - use top-k
                        k = min(n_true, pred_mask[i].size(0))
                        _, top_indices = torch.topk(pred_mask[i], k)
                        valid_pred_mask = torch.zeros_like(pred_mask[i], dtype=torch.bool)
                        valid_pred_mask[top_indices] = True
                    
                    pred_particles = pred_features[i][valid_pred_mask]
                else:
                    # Training: assume first n_true are valid
                    pred_particles = pred_features[i, :n_true]
            else:
                pred_particles = pred_features[i, :n_true]
            
            if pred_particles.shape[0] == 0:
                cumulative_particles += n_true
                continue
            
            # Compute Euclidean pairwise distances [n_true, n_pred]
            euclidean_distances = self._compute_particle_distance(true_particles, pred_particles)
            
            # Transform to hyperbolic distances: d_hyp = arcosh(1 + α × d_E²)
            # IMPORTANT: Formula requires d², but _compute_particle_distance may return d or d²
            # depending on use_squared_distance config. Handle both cases correctly.
            if self.use_squared_distance:
                # Distance metric already returns d², use directly
                euclidean_distances_sq = euclidean_distances
            else:
                # Distance metric returns d, square it for hyperbolic formula
                euclidean_distances_sq = euclidean_distances ** 2
            
            # Hyperbolic transformation with α hyperparameter
            hyperbolic_arg = 1.0 + self.hyperbolic_alpha * euclidean_distances_sq
            # Clamp to avoid numerical issues (arcosh requires x > 1)
            hyperbolic_arg = torch.clamp(hyperbolic_arg, min=1.0 + 1e-8)
            hyperbolic_distances = torch.acosh(hyperbolic_arg)
            
            if self.use_pt_weighting:
                # Compute pT-based weights: w_i = (pT_i)^α / Σ (pT_k)^α
                # pT = sqrt(px² + py²) computed from denormalized momenta
                
                # Get normalization stats
                px_stats = batch.particle_norm_stats['px']
                py_stats = batch.particle_norm_stats['py']
                
                # Denormalize px and py efficiently based on normalization method
                if px_stats['method'] == 'zscore':
                    # Z-score: x = x_norm * std + mean
                    true_px = true_particles[:, self.px_index] * px_stats['std'] + px_stats['mean']
                    true_py = true_particles[:, self.py_index] * py_stats['std'] + py_stats['mean']
                    pred_px = pred_particles[:, self.px_index] * px_stats['std'] + px_stats['mean']
                    pred_py = pred_particles[:, self.py_index] * py_stats['std'] + py_stats['mean']
                else:  # minmax
                    # Min-max: x = x_norm * (max - min) + min
                    px_range = px_stats['max'] - px_stats['min']
                    py_range = py_stats['max'] - py_stats['min']
                    true_px = true_particles[:, self.px_index] * px_range + px_stats['min']
                    true_py = true_particles[:, self.py_index] * py_range + py_stats['min']
                    pred_px = pred_particles[:, self.px_index] * px_range + px_stats['min']
                    pred_py = pred_particles[:, self.py_index] * py_range + py_stats['min']
                
                # Compute pT = sqrt(px² + py²)
                true_pt = torch.sqrt(true_px ** 2 + true_py ** 2 + 1e-8)  # [n_true]
                pred_pt = torch.sqrt(pred_px ** 2 + pred_py ** 2 + 1e-8)  # [n_pred]
                
                # Compute normalized weights: w_i = (pT_i)^α / Σ (pT_k)^α
                true_weights = true_pt ** self.pt_weight_alpha  # [n_true]
                true_weights = true_weights / (true_weights.sum() + 1e-8)
                
                pred_weights = pred_pt ** self.pt_weight_alpha  # [n_pred]
                pred_weights = pred_weights / (pred_weights.sum() + 1e-8)
                
                # Weighted hyperbolic Chamfer distance
                true_to_pred_min = hyperbolic_distances.min(dim=1)[0]
                true_to_pred = (true_weights * true_to_pred_min).sum()
                
                pred_to_true_min = hyperbolic_distances.min(dim=0)[0]
                pred_to_true = (pred_weights * pred_to_true_min).sum()
            else:
                # Unweighted hyperbolic Chamfer distance
                true_to_pred = hyperbolic_distances.min(dim=1)[0].mean()
                pred_to_true = hyperbolic_distances.min(dim=0)[0].mean()
            
            # Bidirectional hyperbolic Chamfer loss
            batch_loss = true_to_pred + pred_to_true
            
            # Check for NaN
            if not torch.isnan(batch_loss):
                total_loss += batch_loss
                valid_count += 1
            
            cumulative_particles += n_true
        
        return total_loss / max(valid_count, 1)
    
    def _mse_particle_loss(self, batch, pred_features, pred_mask):
        """
        MSE loss for particle features (fallback/baseline alternative to Chamfer).
        
        Note: Requires same cardinality (n_pred = n_true) and is order-sensitive,
        so generally INFERIOR to Chamfer distance for set-valued outputs. Only use
        for ablation studies or when particle ordering is meaningful.
        
        Args:
            batch: Batch data with true particle features
            pred_features: [batch_size, max_particles, num_features] predicted features
            pred_mask: [batch_size, max_particles] boolean mask for valid predictions
        
        Returns:
            torch.Tensor: Scalar MSE loss averaged over valid jets
        """
        true_features = batch.x
        
        batch_size = batch.num_graphs
        loss = 0.0
        valid_count = 0
        
        cumulative_particles = 0
        for i in range(batch_size):
            n_true = batch.n_particles[i].item()
            if n_true == 0:
                continue
            
            true_particles = true_features[cumulative_particles:cumulative_particles + n_true]
            
            # Extract predicted particles using mask
            if pred_mask is not None:
                valid_pred_mask = pred_mask[i].bool()  # Ensure boolean type
                pred_particles = pred_features[i][valid_pred_mask]
                # For MSE, we need same number of particles, so take min
                n_common = min(n_true, pred_particles.shape[0])
                true_particles = true_particles[:n_common]
                pred_particles = pred_particles[:n_common]
            else:
                pred_particles = pred_features[i, :n_true]
            
            if pred_particles.shape[0] == 0:
                cumulative_particles += n_true
                continue
            
            # MSE on existing particles
            batch_loss = F.mse_loss(pred_particles, true_particles)
            
            # Check for NaN
            if not torch.isnan(batch_loss):
                loss += batch_loss
                valid_count += 1
            
            cumulative_particles += n_true
        
        return loss / max(valid_count, 1)
    
    def _wasserstein_particle_loss(self, batch, pred_features, pred_mask):
        """
        1-Wasserstein (Earth Mover's) Distance for particle features.
        
        Used during EVALUATION ONLY for more robust distribution comparison.
        Computes W1 distance independently per feature dimension, then averages.
        
        Note: Uses scipy.stats.wasserstein_distance (CPU-only, slower than Chamfer).
        Only computed in eval mode when evaluation_metric='wasserstein'.
        
        Args:
            batch: Batch data with true particle features
            pred_features: [batch_size, max_particles, num_features] predicted features
            pred_mask: [batch_size, max_particles] boolean mask for valid predictions
        
        Returns:
            torch.Tensor: Scalar Wasserstein distance averaged over features and jets
        """
        true_features = batch.x.cpu().numpy()
        pred_features_np = pred_features.cpu().numpy()
        
        batch_size = batch.num_graphs
        total_loss = 0.0
        valid_count = 0
        
        cumulative_particles = 0
        for i in range(batch_size):
            n_true = batch.n_particles[i].item()
            if n_true == 0:
                continue
            
            true_particles = true_features[cumulative_particles:cumulative_particles + n_true]
            pred_particles = pred_features_np[i, :n_true]
            
            # Compute 1-Wasserstein distance for each feature dimension
            batch_loss = 0.0
            num_features = true_particles.shape[1]
            
            for feat_idx in range(num_features):
                true_feat = true_particles[:, feat_idx]
                pred_feat = pred_particles[:, feat_idx]
                
                # Wasserstein distance (Earth Mover's Distance)
                w_dist = wasserstein_distance(true_feat, pred_feat)
                batch_loss += w_dist
            
            # Average over features
            batch_loss /= num_features
            
            if not np.isnan(batch_loss):
                total_loss += batch_loss
                valid_count += 1
            
            cumulative_particles += n_true
        
        # Convert back to tensor
        return torch.tensor(total_loss / max(valid_count, 1), device=batch.x.device)
    
    def _edge_reconstruction_loss(self, batch, pred_features):
        """Edge feature loss - MSE for training, Wasserstein for evaluation"""
        true_features = batch.edge_attr
        
        if true_features.size(0) == 0:
            return torch.tensor(0.0, device=pred_features.device)
        
        if self.training or self.evaluation_metric != 'wasserstein':
            # Training: use MSE distribution matching
            true_mean = true_features.mean(dim=0)
            num_features = true_features.shape[-1]
            pred_mean = pred_features.reshape(-1, num_features).mean(dim=0)
            loss = F.mse_loss(pred_mean, true_mean)
            
            if torch.isnan(loss):
                return torch.tensor(0.0, device=pred_features.device)
            return loss
        else:
            # Evaluation: use Wasserstein distance
            true_np = true_features.cpu().numpy()
            pred_np = pred_features.reshape(-1, true_features.shape[-1]).cpu().numpy()
            
            total_w_dist = 0.0
            num_features = true_np.shape[1]
            
            for feat_idx in range(num_features):
                w_dist = wasserstein_distance(true_np[:, feat_idx], pred_np[:, feat_idx])
                total_w_dist += w_dist
            
            # Average over features
            return torch.tensor(total_w_dist / num_features, device=pred_features.device)
    
    def _hyperedge_reconstruction_loss(self, batch, pred_features, pred_mask):
        """Hyperedge feature loss - MSE for training, Wasserstein for evaluation"""
        true_features = batch.hyperedge_attr
        
        if true_features.size(0) == 0:
            return torch.tensor(0.0, device=pred_features.device)
        
        if self.training or self.evaluation_metric != 'wasserstein':
            # Training: use MSE distribution matching
            true_mean = true_features.mean(dim=0)
            num_features = true_features.shape[-1]
            pred_mean = pred_features.reshape(-1, num_features).mean(dim=0)
            loss = F.mse_loss(pred_mean, true_mean)
            
            if torch.isnan(loss):
                return torch.tensor(0.0, device=pred_features.device)
            return loss
        else:
            # Evaluation: use Wasserstein distance
            true_np = true_features.cpu().numpy()
            pred_np = pred_features.reshape(-1, true_features.shape[-1]).cpu().numpy()
            
            total_w_dist = 0.0
            num_features = true_np.shape[1]
            
            for feat_idx in range(num_features):
                w_dist = wasserstein_distance(true_np[:, feat_idx], pred_np[:, feat_idx])
                total_w_dist += w_dist
            
            # Average over features
            return torch.tensor(total_w_dist / num_features, device=pred_features.device)
        
        # Safety check
        if torch.isnan(loss):
            return torch.tensor(0.0, device=pred_features.device)
        
        return loss
    
    def _topology_loss(self, batch, topology):
        """Topology prediction loss with robust normalization and clamping"""
        # Number of particles loss (normalize by max_particles)
        max_particles = float(self.config['model']['max_particles'])
        
        # Clamp predictions to valid range [0, max_particles] BEFORE normalization
        true_n_particles = torch.clamp(batch.n_particles.float(), min=0.0, max=max_particles)
        pred_n_particles = torch.clamp(topology['n_particles'].float(), min=0.0, max=max_particles)
        
        # Normalize to [0, 1]
        true_n_particles = true_n_particles / max_particles
        pred_n_particles = pred_n_particles / max_particles
        
        particle_count_loss = F.mse_loss(pred_n_particles, true_n_particles)
        
        # Safety check for NaN/Inf (critical!)
        if torch.isnan(particle_count_loss) or torch.isinf(particle_count_loss):
            particle_count_loss = torch.tensor(0.0, device=particle_count_loss.device)
        
        return particle_count_loss
    
    def _jet_feature_loss(self, batch, pred_jet_features, topology):
        """
        Compute weighted MSE loss for jet-level observables.
        
        ═══════════════════════════════════════════════════════════════════════
        JET-LEVEL CONSTRAINTS - PHYSICS-MOTIVATED AUXILIARY LOSS
        ═══════════════════════════════════════════════════════════════════════
        
        Purpose:
        --------
        Enforce consistency between generated particles and jet-level observables.
        Jet properties (pT, η, mass) are computed from particle 4-momenta, so this
        loss acts as a SOFT CONSTRAINT ensuring particle kinematics are physically
        consistent.
        
        Mathematical Form:
        ------------------
            L_jet = w_pT × MSE(pT_jet) + w_η × MSE(η_jet) + 
                    w_m × MSE(m_jet) + w_N × MSE(N_constituents)
        
        where each term compares predicted vs true jet-level features.
        
        Why Jet-Level Loss?
        -------------------
        1. PHYSICS CONSISTENCY: Particles must sum to correct jet 4-momentum
           - pT_jet = |Σ p⃗_i|  (vector sum of transverse momenta)
           - m_jet = √[(Σ E_i)² - |Σ p⃗_i|²]  (invariant mass)
           - η_jet = -ln[tan(θ_jet/2)]  (pseudorapidity from momentum direction)
        
        2. STABILITY: Prevents mode collapse where particles are individually good
           but don't form correct jet structure
        
        3. INTERPRETABILITY: These observables are what physicists measure/care about
        
        4. AUXILIARY SIGNAL: Provides additional gradient signal beyond particle-level
           matching (e.g., Chamfer distance)
        
        Feature Definitions:
        --------------------
        - pT (transverse momentum): Energy component perpendicular to beam axis
          Units: GeV (typically 100-1000 GeV for high-energy jets)
          Importance: Primary jet energy scale, heavily weighted in analyses
        
        - η (pseudorapidity): Angular coordinate along beam axis
          Units: Dimensionless (typically -2.5 to 2.5 for detector coverage)
          Importance: Determines detector region, affects acceptance
        
        - m (invariant mass): Rest mass of jet
          Units: GeV (typically 0-200 GeV for light jets, higher for boosted objects)
          Importance: Discriminates jet substructure (W/Z/Higgs tagging)
        
        - N_constituents: Number of particles in jet
          Units: Count (typically 10-50 particles per jet)
          Importance: Related to jet fragmentation, affects resolution
        
        Implementation Notes:
        ---------------------
        - Individual loss terms have configurable weights (allows physics-based tuning)
        - NaN checks on each term (robust to numerical issues)
        - N_constituents loss only computed if weight > 0 (optional regularization)
        - Uses true jet features from batch.y tensor (preprocessed & normalized)
        - Predicted features come from decoder auxiliary head
        
        Typical Weight Settings:
        ------------------------
        - w_pT = 1.0-2.0: Highest weight (most important observable)
        - w_η = 0.5-1.0: Medium weight (well-measured but less critical)
        - w_m = 0.5-1.5: Medium-high weight (important for substructure)
        - w_N = 0.1-0.5: Low weight (soft regularization only)
        
        Args:
            batch: PyG Batch with y tensor [batch_size, num_features] where:
                   y[:, jet_pt_index] = jet transverse momentum
                   y[:, jet_eta_index] = jet pseudorapidity
                   y[:, jet_mass_index] = jet invariant mass
                   n_particles = number of constituents per jet
            pred_jet_features: [batch_size, 3] predicted (jet_pt, jet_eta, jet_mass)
            topology: Dict with 'n_particles' [batch_size] predicted constituent count
        
        Returns:
            torch.Tensor: Scalar weighted MSE loss for jet features
        """
        # Extract true jet features from y tensor
        # y tensor shape: [batch_size, num_features]
        # Indices: [0: jet_type, 1: jet_pt, 2: jet_eta, 3: jet_mass, ...]
        true_jet_pt = batch.y[:, self.jet_pt_index]      # [batch_size]
        true_jet_eta = batch.y[:, self.jet_eta_index]    # [batch_size]
        true_jet_mass = batch.y[:, self.jet_mass_index]  # [batch_size]
        
        # Extract predicted features (columns from pred_jet_features)
        pred_jet_pt = pred_jet_features[:, 0]    # [batch_size]
        pred_jet_eta = pred_jet_features[:, 1]   # [batch_size]
        pred_jet_mass = pred_jet_features[:, 2]  # [batch_size]
        
        # Compute individual MSE losses with NaN checks
        loss_pt = F.mse_loss(pred_jet_pt, true_jet_pt)
        loss_eta = F.mse_loss(pred_jet_eta, true_jet_eta)
        loss_mass = F.mse_loss(pred_jet_mass, true_jet_mass)
        
        # Check for NaN in individual losses
        if torch.isnan(loss_pt):
            print(f"⚠️  NaN in jet pt loss - pred: {pred_jet_pt[:5]}, true: {true_jet_pt[:5]}")
            loss_pt = torch.tensor(0.0, device=pred_jet_features.device)
        if torch.isnan(loss_eta):
            print(f"⚠️  NaN in jet eta loss - pred: {pred_jet_eta[:5]}, true: {true_jet_eta[:5]}")
            loss_eta = torch.tensor(0.0, device=pred_jet_features.device)
        if torch.isnan(loss_mass):
            print(f"⚠️  NaN in jet mass loss - pred: {pred_jet_mass[:5]}, true: {true_jet_mass[:5]}")
            loss_mass = torch.tensor(0.0, device=pred_jet_features.device)
        
        # Number of constituents loss (only if weight > 0)
        loss_n_constituents = torch.tensor(0.0, device=pred_jet_features.device)
        if self.jet_n_constituents_weight > 0:
            try:
                true_n_constituents = batch.n_particles.float()  # [batch_size]
                pred_n_constituents = topology['n_particles'].float()  # [batch_size]
                loss_n_constituents = F.mse_loss(pred_n_constituents, true_n_constituents)
                
                if torch.isnan(loss_n_constituents):
                    print(f"⚠️  NaN in N_constituents loss")
                    loss_n_constituents = torch.tensor(0.0, device=pred_jet_features.device)
            except Exception as e:
                print(f"⚠️  Error computing N_constituents loss: {e}")
                loss_n_constituents = torch.tensor(0.0, device=pred_jet_features.device)
        
        # Weighted combination
        jet_loss = (
            self.jet_pt_weight * loss_pt +
            self.jet_eta_weight * loss_eta +
            self.jet_mass_weight * loss_mass +
            self.jet_n_constituents_weight * loss_n_constituents
        )
        
        # Safety check for NaN/Inf
        if torch.isnan(jet_loss) or torch.isinf(jet_loss):
            print("⚠️  Warning: NaN/Inf detected in jet feature loss")
            jet_loss = torch.tensor(0.0, device=pred_jet_features.device)
        
        return jet_loss
    
    def _local_global_consistency_loss(self, batch, pred_features, pred_mask):
        """
        Compute local→global physics consistency loss.
        
        ═══════════════════════════════════════════════════════════════════════
        LOCAL→GLOBAL PHYSICS CONSISTENCY LOSS
        ═══════════════════════════════════════════════════════════════════════
        
        Purpose:
        --------
        Enforce agreement between generated particle 4-momenta and true jet-level
        observables by computing jet properties from particles and comparing to truth.
        
        This is a PHYSICS CONSTRAINT that ensures:
        1. Energy-momentum conservation: Σ particles ≈ jet 4-momentum
        2. Kinematic consistency: pT, η, mass computed from constituents match truth
        3. Multi-scale correctness: Local (particles) and global (jet) views agree
        
        Mathematical Formulation:
        -------------------------
        1. Denormalize particle 4-vectors [E, px, py, pz] from normalized predictions
        2. Sum particle momenta to compute jet-level observables:
           - pT_sum = √(Σpx)² + (Σpy)²
           - m_sum = √(ΣE)² - (Σpx)² - (Σpy)² - (Σpz)²
           - η_sum = asinh(Σpz / (pT_sum + ε))
        
        3. Compare to true jet observables using normalized residuals:
           - e_pT = (pT_sum - pT_true) / σ_pT
           - e_η = (η_sum - η_true) / σ_η
           - e_m = (m_sum - m_true) / σ_m
        
        4. Weighted MSE loss:
           L_cons = w_pT × 𝔼[e_pT²] + w_η × 𝔼[e_η²] + w_m × 𝔼[e_m²]
        
        Workflow:
        ---------
        1. Extract predicted particles (respecting masks for valid particles)
        2. Denormalize using batch.particle_norm_stats (supports zscore and minmax)
        3. Sum 4-momenta per jet
        4. Compute jet observables (pT, η, m) from sums
        5. Denormalize true jet features from batch.y using batch.jet_norm_stats
        6. Compute normalized residuals using jet-level σ values
        7. Weighted MSE of residuals
        
        Args:
            batch: PyG Batch with:
                - x: [N_particles_total, 4] true particle features (for reference, not used here)
                - y: [batch_size, 4] true jet features [jet_type, log_pt, eta, log_mass]
                - particle_norm_stats: {'E': stats, 'px': stats, 'py': stats, 'pz': stats}
                - jet_norm_stats: {'pt': stats, 'eta': stats, 'mass': stats}
                - n_particles: [batch_size] true particle counts per jet
                - batch: [N_particles_total] batch assignment vector
                
            pred_features: [batch_size, max_particles, 4] predicted normalized particles
            pred_mask: [batch_size, max_particles] validity mask (0-1 soft, or bool)
        
        Returns:
            torch.Tensor: Scalar consistency loss averaged over batch
        """
        batch_size = batch.num_graphs
        device = pred_features.device
        
        # Avoid repeated dict lookups
        particle_norm_stats = batch.particle_norm_stats
        jet_norm_stats = batch.jet_norm_stats
        
        E_stats = particle_norm_stats['E']
        px_stats = particle_norm_stats['px']
        py_stats = particle_norm_stats['py']
        pz_stats = particle_norm_stats['pz']
        
        pt_stats = jet_norm_stats['pt']
        eta_stats = jet_norm_stats['eta']
        mass_stats = jet_norm_stats['mass']
        
        # Check normalization method once (same for all components)
        is_zscore = E_stats['method'] == 'zscore'
        
        # Extract valid particle mask
        if pred_mask is not None:
            if pred_mask.dtype == torch.bool:
                valid_mask = pred_mask  # [batch_size, max_particles]
            else:
                valid_mask = pred_mask > 0.5  # Threshold soft mask
        else:
            # Fallback: assume all particles valid (shouldn't happen)
            valid_mask = torch.ones_like(pred_features[..., 0], dtype=torch.bool)
        
        # Flatten batch dimension: [batch_size, max_particles, 4] → [batch_size*max_particles, 4]
        pred_features_flat = pred_features.reshape(-1, 4)  # [B*M, 4]
        valid_mask_flat = valid_mask.reshape(-1)  # [B*M]
        
        # Create batch indices for scatter operations
        # batch_indices[i] tells which jet particle i belongs to
        max_particles = pred_features.shape[1]
        batch_indices = torch.arange(batch_size, device=device).repeat_interleave(max_particles)  # [B*M]
        
        # Filter to valid particles only
        valid_particles = pred_features_flat[valid_mask_flat]  # [N_valid, 4]
        valid_batch_indices = batch_indices[valid_mask_flat]  # [N_valid]
        
        # Early exit if no valid particles
        if valid_particles.shape[0] == 0:
            return torch.tensor(0.0, device=device)
        
        # Vectorized denormalization (all particles at once)
        # Extract components (vectorized slice)
        E_norm = valid_particles[:, self.E_index]    # [N_valid]
        px_norm = valid_particles[:, self.px_index]  # [N_valid]
        py_norm = valid_particles[:, self.py_index]  # [N_valid]
        pz_norm = valid_particles[:, self.pz_index]  # [N_valid]
        
        # Denormalize based on method (single branch, applied to all)
        if is_zscore:
            # Z-score: x = x_norm * std + mean
            E_phys = E_norm * E_stats['std'] + E_stats['mean']
            px_phys = px_norm * px_stats['std'] + px_stats['mean']
            py_phys = py_norm * py_stats['std'] + py_stats['mean']
            pz_phys = pz_norm * pz_stats['std'] + pz_stats['mean']
            
            # Pre-compute jet denorm factors (for true jet denormalization)
            pt_denorm_scale = pt_stats['std']
            pt_denorm_offset = pt_stats['mean']
            eta_denorm_scale = eta_stats['std']
            eta_denorm_offset = eta_stats['mean']
            mass_denorm_scale = mass_stats['std']
            mass_denorm_offset = mass_stats['mean']
            
        else:  # minmax
            # Min-max: x = x_norm * (max - min) + min
            E_range = E_stats['max'] - E_stats['min']
            px_range = px_stats['max'] - px_stats['min']
            py_range = py_stats['max'] - py_stats['min']
            pz_range = pz_stats['max'] - pz_stats['min']
            
            E_phys = E_norm * E_range + E_stats['min']
            px_phys = px_norm * px_range + px_stats['min']
            py_phys = py_norm * py_range + py_stats['min']
            pz_phys = pz_norm * pz_range + pz_stats['min']
            
            # Pre-compute jet denorm factors (for true jet denormalization)
            pt_range = pt_stats['max'] - pt_stats['min']
            eta_range = eta_stats['max'] - eta_stats['min']
            mass_range = mass_stats['max'] - mass_stats['min']
            
            pt_denorm_scale = pt_range
            pt_denorm_offset = pt_stats['min']
            eta_denorm_scale = eta_range
            eta_denorm_offset = eta_stats['min']
            mass_denorm_scale = mass_range
            mass_denorm_offset = mass_stats['min']
        
        # === PARALLEL REDUCTION (scatter_add for per-jet sums) ===
        # Sum particles by jet using scatter_add (GPU-optimized parallel reduction)
        E_sum = torch.zeros(batch_size, device=device).scatter_add_(0, valid_batch_indices, E_phys)
        px_sum = torch.zeros(batch_size, device=device).scatter_add_(0, valid_batch_indices, px_phys)
        py_sum = torch.zeros(batch_size, device=device).scatter_add_(0, valid_batch_indices, py_phys)
        pz_sum = torch.zeros(batch_size, device=device).scatter_add_(0, valid_batch_indices, pz_phys)
        
        # Count valid particles per jet (for filtering jets with zero particles)
        valid_count = torch.zeros(batch_size, device=device, dtype=torch.int32).scatter_add_(
            0, valid_batch_indices, torch.ones_like(valid_batch_indices, dtype=torch.int32)
        )
        
        # VECTORIZED JET OBSERVABLES 
        # Transverse momentum: pT = √(px² + py²)
        pred_jet_pt = torch.sqrt(px_sum ** 2 + py_sum ** 2 + 1e-8)  # [batch_size]
        
        # Invariant mass: m² = E² - p²
        mass_sq = E_sum ** 2 - px_sum ** 2 - py_sum ** 2 - pz_sum ** 2
        pred_jet_mass = torch.sqrt(torch.clamp(mass_sq, min=0.0) + 1e-8)  # [batch_size]
        
        # Pseudorapidity: η = asinh(pz / pT)
        pred_jet_eta = torch.asinh(pz_sum / (pred_jet_pt + 1e-8))  # [batch_size]
        
        # === VECTORIZED TRUE JET DENORMALIZATION ===
        # Extract normalized true jet features (all jets at once)
        true_jet_pt_norm = batch.y[:, self.jet_pt_index]     # [batch_size]
        true_jet_eta_norm = batch.y[:, self.jet_eta_index]   # [batch_size]
        true_jet_mass_norm = batch.y[:, self.jet_mass_index] # [batch_size]
        
        # Denormalize (vectorized, already have scales/offsets)
        true_jet_logpt = true_jet_pt_norm * pt_denorm_scale + pt_denorm_offset
        true_jet_eta = true_jet_eta_norm * eta_denorm_scale + eta_denorm_offset
        true_jet_logmass = true_jet_mass_norm * mass_denorm_scale + mass_denorm_offset
        
        # Apply exp to recover physical values from log
        true_jet_pt = torch.exp(true_jet_logpt)
        true_jet_mass = torch.exp(true_jet_logmass)
        
        # === COMPUTE RESIDUAL NORMALIZATION SCALES FROM PHYSICAL VALUES ===
        # For proper normalization, compute std/range from ACTUAL physical distributions
        if is_zscore:
            # For z-score: use std of physical (exponentiated) values
            # η is already physical, pT and mass are exponentiated from log-space
            pt_residual_scale = true_jet_pt.std() + 1e-8      # std of exp(log_pt)
            eta_residual_scale = eta_stats['std'] + 1e-8    # std of eta (already physical)
            mass_residual_scale = true_jet_mass.std() + 1e-8  # std of exp(log_mass)
        else:  # minmax
            # For minmax: use range (max - min) of physical values
            pt_residual_scale = (true_jet_pt.max() - true_jet_pt.min()) + 1e-8
            eta_residual_scale = eta_stats['max'] - eta_stats['min'] + 1e-8
            mass_residual_scale = (true_jet_mass.max() - true_jet_mass.min()) + 1e-8
        
        # === VECTORIZED RESIDUALS & MASKING ===
        # Filter to jets with at least one valid particle
        valid_jets_mask = valid_count > 0  # [batch_size]
        
        if not valid_jets_mask.any():
            return torch.tensor(0.0, device=device)
        
        # Compute residuals (only for valid jets using masked operations)
        residual_pt = (pred_jet_pt - true_jet_pt)[valid_jets_mask]
        residual_eta = (pred_jet_eta - true_jet_eta)[valid_jets_mask]
        residual_mass = (pred_jet_mass - true_jet_mass)[valid_jets_mask]
        
        # Normalize residuals (vectorized division, already have scales)
        normalized_residual_pt = residual_pt / (pt_residual_scale + 1e-8)
        normalized_residual_eta = residual_eta / (eta_residual_scale + 1e-8)
        normalized_residual_mass = residual_mass / (mass_residual_scale + 1e-8)
        
        # Compute MSE and weight in one step (avoids storing intermediate tensors)
        loss_pt = (normalized_residual_pt ** 2).mean()
        loss_eta = (normalized_residual_eta ** 2).mean()
        loss_mass = (normalized_residual_mass ** 2).mean()
        
        # NaN checks (use where to avoid branches)
        loss_pt = torch.where(torch.isnan(loss_pt), torch.zeros_like(loss_pt), loss_pt)
        loss_eta = torch.where(torch.isnan(loss_eta), torch.zeros_like(loss_eta), loss_eta)
        loss_mass = torch.where(torch.isnan(loss_mass), torch.zeros_like(loss_mass), loss_mass)
        
        # Weighted combination (fused multiply-add pattern)
        consistency_loss = (
            self.consistency_pt_weight * loss_pt +
            self.consistency_eta_weight * loss_eta +
            self.consistency_mass_weight * loss_mass
        )
        
        # Final NaN/Inf check
        consistency_loss = torch.where(
            torch.isnan(consistency_loss) | torch.isinf(consistency_loss),
            torch.zeros_like(consistency_loss),
            consistency_loss
        )
        
        return consistency_loss
    
    def _kl_divergence(self, mu, logvar):
        """
        Compute KL divergence D_KL(q(z|x) || p(z)) with free bits regularization.
        
        ═══════════════════════════════════════════════════════════════════════
        KL DIVERGENCE - VAE REGULARIZATION TERM
        ═══════════════════════════════════════════════════════════════════════
        
        Mathematical Formulation:
        -------------------------
        KL divergence between approximate posterior q(z|x) = N(μ(x), σ²(x))
        and prior p(z) = N(0, I):
        
            D_KL(q||p) = -1/2 × Σ_d [1 + log(σ_d²) - μ_d² - σ_d²]
        
        where:
            μ = encoder mean: [batch_size, latent_dim]
            σ² = exp(logvar): encoder variance
            d indexes latent dimensions (typically 128)
        
        Expanded per dimension:
            KL_d = -1/2 × [1 + log(σ_d²) - μ_d² - σ_d²]
                 = 1/2 × [μ_d² + σ_d² - log(σ_d²) - 1]
        
        Total: D_KL = Σ_d KL_d  (sum over all latent dimensions)
        
        Why KL Divergence?
        ------------------
        1. PREVENTS POSTERIOR COLLAPSE: Without KL term, encoder can output
           deterministic latents (σ→0), reducing VAE to deterministic autoencoder
        
        2. ENABLES GENERATION: Forces latent space to match prior N(0,I), so we
           can sample z ~ N(0,I) at generation time
        
        3. INFORMATION BOTTLENECK: Encourages encoder to compress information
           efficiently (only store what's needed for reconstruction)
        
        4. DISENTANGLEMENT: With proper annealing, encourages factorized latent
           representations where dimensions encode independent factors
        
        Free Bits Regularization:
        -------------------------
        Problem: Early in training, KL can collapse to 0, preventing learning
        Solution: Only penalize KL if it exceeds minimum threshold per dimension
        
            KL_d^{free} = max(0, KL_d - λ_{free})
        
        where λ_{free} ∈ [0.5, 2.0] bits per dimension (typical: 0.5-1.0)
        
        Effect:
        - Allows each latent dimension to carry ≥ λ_{free} bits of information
        - Prevents aggressive compression early in training
        - Stabilizes training by avoiding KL → 0 collapse
        
        KL Annealing (handled in compute_loss):
        ---------------------------------------
        Schedule: β(epoch) = min(1.0, epoch / T_warmup) × β_max
        
        Why anneal?
        - Early: β ≈ 0 → focus on reconstruction, learn meaningful representations
        - Late: β → β_max → enforce prior matching for generation
        - Typical: T_warmup = 50-100 epochs, β_max = 0.05-0.5
        
        Numerical Stability (CRITICAL):
        -------------------------------
        1. CLAMP μ, logvar ∈ [-10, 10]:
           - Prevents exp(logvar) overflow (exp(10) ≈ 22026, exp(50) = inf!)
           - Prevents μ² overflow (100² = 10000, manageable)
        
        2. CLAMP KL_d ∈ [0, 100]:
           - Theoretical max with clamped inputs ≈ 50
           - 100 provides safety margin
           - Prevents gradient explosion from single bad dimension
        
        3. NaN/Inf final check:
           - Return 0 if anything went wrong (graceful degradation)
           - Allows training to continue with reconstruction loss only
        
        Typical Values During Training:
        --------------------------------
        - Epoch 1: KL ≈ 0.01-0.1 (posterior collapse, annealing helps)
        - Epoch 50: KL ≈ 10-50 (healthy, balanced with reconstruction)
        - Epoch 100+: KL ≈ 20-80 (stable, latent space well-formed)
        - Per dimension: 0.1-1.0 (most dimensions used, some pruned)
        
        Troubleshooting:
        ----------------
        - KL → 0: Increase kl_max_weight or reduce kl_warmup_epochs
        - KL → ∞: Reduce kl_max_weight, check for encoder numerical issues
        - KL oscillates: Reduce learning rate or increase gradient clipping
        - KL per dim >> 1: Some dimensions are overfitting, increase free bits
        
        Args:
            mu: [batch_size, latent_dim] encoder means
            logvar: [batch_size, latent_dim] encoder log-variances (log σ²)
        
        Returns:
            torch.Tensor: Scalar KL divergence averaged over batch
        """
        # CRITICAL: Re-clamp mu and logvar to prevent numerical explosion
        # Even though encoder clamps, gradients can push values outside range
        mu = torch.clamp(mu, min=-10.0, max=10.0)
        logvar = torch.clamp(logvar, min=-10.0, max=10.0)
        
        # KL per dimension: -0.5 * (1 + logvar - mu^2 - exp(logvar))
        kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        
        # Additional safety: clamp KL per dimension to prevent explosion
        # Theoretical max KL per dim with clamped values ≈ 50, so 100 is safe upper bound
        kl_per_dim = torch.clamp(kl_per_dim, min=0.0, max=100.0)
        
        # Apply free bits (only penalize if KL > threshold)
        free_bits = self.config['training'].get('kl_free_bits', 0.0)
        if free_bits > 0:
            kl_per_dim = torch.clamp(kl_per_dim - free_bits, min=0.0)
        
        kl_loss = torch.sum(kl_per_dim, dim=-1)
        kl_mean = kl_loss.mean()
        
        # Final safety check for NaN/Inf
        if torch.isnan(kl_mean) or torch.isinf(kl_mean):
            return torch.tensor(0.0, device=mu.device)
        
        return kl_mean
    
    def generate(self, num_samples, jet_type, device='cuda'):
        """
        Generate new jets from random latent vectors.
        
        Args:
            num_samples: Number of jets to generate
            jet_type: Jet type tensor [num_samples]
            device: Device to generate on
        
        Returns:
            Dictionary with generated features
        """
        self.eval()
        with torch.no_grad():
            # Sample from prior
            z = torch.randn(num_samples, self.config['model']['latent_dim'], device=device)
            
            # Decode
            output = self.decoder(z, jet_type, temperature=0.5)
        
        return output


if __name__ == "__main__":
    import yaml
    from data.bipartite_dataset import BipartiteJetDataset, collate_bipartite_batch
    from torch.utils.data import DataLoader
    
    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Create dataset and loader
    dataset = BipartiteJetDataset(generate_dummy=True)
    loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_bipartite_batch)
    batch = next(iter(loader))
    
    # Create model
    model = BipartiteHyperVAE(config)
    
    # Forward pass
    output = model(batch)
    
    # Compute loss
    losses = model.compute_loss(batch, output, epoch=0)
    
    print("Forward pass successful!")
    print(f"Total loss: {losses['total'].item():.4f}")
    print(f"  Particle loss: {losses['particle'].item():.4f}")
    print(f"  Edge loss: {losses['edge'].item():.4f}")
    print(f"  Hyperedge loss: {losses['hyperedge'].item():.4f}")
    print(f"  Jet loss: {losses['jet'].item():.4f}")
    print(f"  Topology loss: {losses['topology'].item():.4f}")
    print(f"  KL loss: {losses['kl'].item():.4f}")
    
    # Test generation
    print("\nTesting generation...")
    generated = model.generate(4, torch.tensor([0, 1, 2, 0]), device='cpu')
    print(f"Generated particle features: {generated['particle_features'].shape}")