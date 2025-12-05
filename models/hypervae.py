"""
Particle Jet Variational Autoencoder (HyperVAE) for Jet Generation

This module implements a physics-informed VAE architecture that combines:
1. Lorentz-equivariant attention (L-GATr) for particle 4-momenta processing
2. Graph structure learning via edge/hyperedge observables from dataset
3. Distribution matching losses for multi-particle correlations
4. Local→global consistency for physics-preserving generation

Key Features:
- L-GATr integration for 4-vector (E, px, py, pz) processing with Lorentz symmetry
- Hyperbolic Chamfer loss for robust set matching
- Distribution losses: Wasserstein distance on edge/hyperedge observables
- Local→global consistency: Enforces particles sum to correct jet observables
- KL divergence annealing with cyclical schedule for stable training

Architecture:
- Encoder: Processes particles + graph structure → latent distribution q(z|x)
- Decoder: Generates particles + jet features from latent code z
- Loss: Particle Chamfer + Distribution matching + Jet features + Consistency + KL

Loss Components:
1. Particle: Hyperbolic Chamfer distance (pT-weighted set matching)
2. Distribution: Wasserstein on edge (2-pt EEC) and hyperedge (N-pt EEC) observables
3. Jet: MSE on jet_pt, jet_eta, jet_mass predictions
4. Consistency: Particles 4-momenta sum to jet-level observables
5. KL: Latent regularization with cyclical annealing

Note: Edge/hyperedge features computed FROM particles during loss, not decoder outputs
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .encoder import BipartiteEncoder
from .decoder import BipartiteDecoder
from scipy.stats import wasserstein_distance
import numpy as np
import math
import sys
# Add parent directory to path to import utils
sys.path.insert(0, '/home/anuranja/sanmay_project/work/hyperVAE')
from utils import get_eec_ls_values, normalize_array


class BipartiteHyperVAE(nn.Module):
    """
    Particle Jet Variational Autoencoder for Physics-Informed Generation.
    
    Encodes jets into latent space and reconstructs particles with physics constraints.
    Uses graph structure (edges/hyperedges) for encoding but generates only particles.
    
    Architecture:
    1. Encoder: Particles + graph observables → latent q(z|x)
    2. Latent: z ~ q(z|x) via reparameterization trick
    3. Decoder: z → particles (4-momenta) + jet features
    4. Loss: Chamfer + Distribution matching + Jet features + Consistency + KL
    
    Key Innovations:
    - Hyperbolic Chamfer loss: Better captures hierarchical jet structure
    - Distribution losses: Match 2-pt (edge) and N-pt (hyperedge) correlations
    - Local→global consistency: Particles sum to correct jet-level observables
    - Graph-informed encoding: Uses pre-computed edge/hyperedge observables
    
    Input (PyTorch Geometric Data):
    - particle_x: [N, 4] - 4-momenta (E, px, py, pz)
    - edge_attr: [M, 5] - Edge observables [2pt_EEC, ln_delta, ln_kT, ln_z, ln_m2]
    - hyperedge_attr: [K, features] - N-point correlations [3pt_EEC, 4pt_EEC, ...]
    - y: [batch, 4] - Jet features [jet_type, jet_pt, jet_eta, jet_mass]
    
    Output:
    - Reconstructed particles (4-momenta)
    - Jet-level features (pt, eta, mass)
    - Latent parameters (mu, logvar)
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
        self.loss_config = config['training'].get('loss_config', {})
        self.particle_loss_type = self.loss_config.get('particle_loss_type', 'chamfer')
        self.particle_distance_metric = self.loss_config.get('particle_distance_metric', 'rapidity_phi')    
        
        # For 'euclidean_4d' metric (4-vector distance used with L-GATr)
        # Indices point to [E, px, py, pz] in particle feature tensor
        self.E_index = self.loss_config.get('E_index', 0)
        self.px_index = self.loss_config.get('px_index', 1)
        self.py_index = self.loss_config.get('py_index', 2)
        self.pz_index = self.loss_config.get('pz_index', 3)
        
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
        
        # === Local→Global Physics Consistency Loss ===
        # Enforces agreement between particle 4-momenta and jet-level observables
        self.use_consistency_loss = self.loss_config.get('use_consistency_loss', True)
        self.consistency_pt_weight = self.loss_config.get('consistency_pt_weight', 2.0)
        self.consistency_eta_weight = self.loss_config.get('consistency_eta_weight', 1.0)
        self.consistency_mass_weight = self.loss_config.get('consistency_mass_weight', 2.5)
        
        # === Edge/Hyperedge Distribution Loss ===
        # Compute edge/hyperedge observables from particles and match distributions
        self.use_distribution_loss = self.loss_config.get('use_distribution_loss', False)
        self.distribution_loss_type = self.loss_config.get('distribution_loss_type', 'mse')  # 'mse' or 'wasserstein'
        self.edge_distribution_weight = self.loss_weights.get('edge_distribution', 1.0)
        self.hyperedge_distribution_weight = self.loss_weights.get('hyperedge_distribution', 1.0)
        
        # EEC computation parameters (for hyperedge features)
        self.eec_prop = self.loss_config.get('eec_prop', [[2, 3], 200, [1e-4, 2]])
        
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
                - True: Generate edges/hyperedges (legacy, not used)
                - False: Only particles (current default)
        
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
        # Generate reconstructions from latent code (particles only - decoder no longer outputs edges/hyperedges)
        output = self.decoder(
            z, 
            batch.jet_type,  # Condition on jet type (quark/gluon/top)
            temperature      # Controls sharpness of topology assignments
        )
        
        # === ATTACH LATENT PARAMETERS ===
        # Include latent distribution parameters for KL divergence computation
        output['mu'] = mu
        output['logvar'] = logvar
        output['z'] = z
        
        return output
    
    def compute_loss(self, batch, output, epoch=0):
        """
        Compute total VAE loss: Reconstruction + KL Divergence + Distribution Matching + Physics Consistency.
        
        Loss Formulation:
            L_total = L_particle + L_edge_dist + L_hyperedge_dist + L_jet + L_consistency + β(epoch) * L_KL
        
        Where:
        - L_particle: Chamfer distance between true and predicted particles (MAIN LOSS)
        - L_edge_dist: Wasserstein distance for edge observable distributions (2-pt EEC, ln_delta, etc.)
        - L_hyperedge_dist: Wasserstein distance for hyperedge distributions (3-pt+ EEC)
        - L_jet: Weighted MSE for jet-level features (jet_pt, jet_eta, jet_mass, n_constituents)
        - L_consistency: Local→global physics consistency (particles sum to jet observables)
        - L_KL: KL divergence between q(z|x) and p(z)=N(0,I)
        - β(epoch): KL annealing weight (gradually increases from 0 to 1)
        
        Loss Weights:
        - particle_features: 12000.0 (main objective)
        - edge_distribution: 1.0 (edge observable matching)
        - hyperedge_distribution: 1.0 (hyperedge observable matching)
        - jet_features: 3000.0 (constrain jet-level properties)
        - local_global_consistency: 3000.0 (physics constraint)
        - kl_divergence: 0.3 * β(epoch) (annealed regularization)
        
        Args:
            batch: PyG Batch with ground truth data
            output: Dict from forward() with predictions
            epoch: Current epoch number (for KL annealing)
        
        Returns:
            dict with individual and total losses:
                - 'particle': Particle reconstruction loss
                - 'edge_distribution': Edge distribution matching loss (0 if disabled)
                - 'hyperedge_distribution': Hyperedge distribution matching loss (0 if disabled)
                - 'jet': Jet feature loss
                - 'consistency': Local→global physics consistency loss
                - 'kl': KL divergence
                - 'kl_weight': Current KL annealing weight β(epoch)
                - 'kl_raw': Raw KL before weighting (for monitoring)
                - 'total': Weighted sum of all losses
        
        Note:
            Edge/hyperedge losses are computed from generated particles (not decoder outputs).
            Distribution matching uses Wasserstein distance on observable features.
        """
        # Fix batch.y shape if needed (PyG concatenates [4] tensors into [batch_size*4])
        if batch.y.dim() == 1 and hasattr(batch, 'num_graphs'):
            batch.y = batch.y.view(batch.num_graphs, -1)  # [batch_size, 4]
        
        losses = {}
        
        # === 1. PARTICLE RECONSTRUCTION LOSS (MAIN LOSS) ===
        # Chamfer distance: Permutation-invariant set matching
        # Measures how well predicted particle set matches true particle set
        particle_loss = self._particle_reconstruction_loss(
            batch, output['particle_features'], output['particle_mask']
        )
        losses['particle'] = particle_loss
        
        # === 2. DISTRIBUTION LOSS (EDGE/HYPEREDGE FROM PARTICLES) ===
        # Compute edge/hyperedge observables from generated particles
        if self.use_distribution_loss:
            if self.distribution_loss_type=='wasserstein':
                distribution_loss = self._edge_hyperedge_wasserstein_loss(
                    batch, output['particle_features'], output['particle_mask']
                )
                losses['edge_distribution'] = distribution_loss['edge']
                losses['hyperedge_distribution'] = distribution_loss['hyperedge']
            elif self.distribution_loss_type=='mse':
                distribution_loss = self._edge_hyperedge_mse_loss(
                    batch, output['particle_features'], output['particle_mask']
                )
                losses['edge_distribution'] = distribution_loss['edge']
                losses['hyperedge_distribution'] = distribution_loss['hyperedge']
        else:
            losses['edge_distribution'] = torch.tensor(0.0, device=batch.x.device)
            losses['hyperedge_distribution'] = torch.tensor(0.0, device=batch.x.device)
        
        # === 3. JET FEATURE LOSS ===
        # Pass topology information for N_constituents prediction
        jet_loss = self._jet_feature_loss(batch, output['jet_features'], output['topology'])
        losses['jet'] = jet_loss
        
        # === 4. LOCAL→GLOBAL PHYSICS CONSISTENCY LOSS ===
        # Enforces that generated particles sum to correct jet-level observables
        if self.use_consistency_loss:
            consistency_loss = self._local_global_consistency_loss(
                batch, output['particle_features'], output['particle_mask']
            )
            losses['consistency'] = consistency_loss
        else:
            losses['consistency'] = torch.tensor(0.0, device=batch.x.device)
        
        # === 5. KL DIVERGENCE ===
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
        
        # === TOTAL LOSS ===
        total_loss = (
            self.loss_weights['particle_features'] * losses['particle'] +
            self.loss_weights.get('edge_distribution', 0.0) * losses.get('edge_distribution', 0.0) +
            self.loss_weights.get('hyperedge_distribution', 0.0) * losses.get('hyperedge_distribution', 0.0) +
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
        """
        Edge feature loss - Distribution matching approach.
        
        Similar to hyperedge loss: decoder predicts limited edges while dataset
        has full pairwise set. Match statistical distributions for guidance.
        
        Uses:
        - Training: MSE on mean + std + higher moments
        - Evaluation: Wasserstein distance
        """
        true_features = batch.edge_attr  # [total_edges_in_batch, num_features]
        
        if true_features.size(0) == 0:
            return torch.tensor(0.0, device=pred_features.device)
        
        # Get number of features
        num_features = true_features.shape[-1]
        
        # Flatten predicted features
        pred_flat = pred_features.reshape(-1, num_features)  # [batch*max_edges, num_features]
        
        if self.training or self.evaluation_metric != 'wasserstein':
            # Training: Match distribution moments (mean, std, skewness)
            
            # 1. Match means
            true_mean = true_features.mean(dim=0)  # [num_features]
            pred_mean = pred_flat.mean(dim=0)  # [num_features]
            mean_loss = F.mse_loss(pred_mean, true_mean)
            
            # 2. Match standard deviations
            true_std = true_features.std(dim=0) + 1e-6  # [num_features]
            pred_std = pred_flat.std(dim=0) + 1e-6  # [num_features]
            std_loss = F.mse_loss(pred_std, true_std)
            
            # 3. Match higher moments (distribution shape)
            true_centered = (true_features - true_mean) / true_std
            pred_centered = (pred_flat - pred_mean) / pred_std
            true_moment3 = (true_centered ** 3).mean(dim=0)
            pred_moment3 = (pred_centered ** 3).mean(dim=0)
            moment3_loss = F.mse_loss(pred_moment3, true_moment3)
            
            # Weighted combination
            loss = 0.5 * mean_loss + 0.3 * std_loss + 0.2 * moment3_loss
            
            if torch.isnan(loss) or torch.isinf(loss):
                return torch.tensor(0.0, device=pred_features.device)
            return loss
        else:
            # Evaluation: use Wasserstein distance
            true_np = true_features.cpu().numpy()
            pred_np = pred_flat.cpu().numpy()
            
            total_w_dist = 0.0
            
            for feat_idx in range(num_features):
                w_dist = wasserstein_distance(true_np[:, feat_idx], pred_np[:, feat_idx])
                total_w_dist += w_dist
            
            # Average over features
            return torch.tensor(total_w_dist / num_features, device=pred_features.device)
    
    def _hyperedge_reconstruction_loss(self, batch, pred_features, pred_mask):
        """
        Hyperedge feature loss - Distribution matching approach.
        
        Since decoder predicts limited hyperedges (e.g., 20) while dataset has full
        combinatorial set (e.g., C(30,3)=4060), we match statistical distributions
        rather than per-hyperedge alignment.
        
        Uses:
        - Training: MSE on mean + std + higher moments
        - Evaluation: Wasserstein distance
        """
        true_features = batch.hyperedge_attr  # [total_hyperedges_in_batch, num_features]
        
        if true_features.size(0) == 0:
            return torch.tensor(0.0, device=pred_features.device)
        
        # Get number of features
        num_features = true_features.shape[-1]
        
        # Flatten predicted features and apply mask
        # pred_features: [batch, max_hyperedges, num_features]
        # pred_mask: [batch, max_hyperedges]
        pred_flat = pred_features.reshape(-1, num_features)  # [batch*max_hyperedges, num_features]
        mask_flat = pred_mask.reshape(-1)  # [batch*max_hyperedges]
        
        # Keep only valid (masked) predictions
        pred_valid = pred_flat[mask_flat > 0.5]  # [num_valid_predictions, num_features]
        
        if pred_valid.size(0) == 0:
            return torch.tensor(0.0, device=pred_features.device)
        
        if self.training or self.evaluation_metric != 'wasserstein':
            # Training: Match distribution moments (mean, std, skewness)
            # This ensures generated hyperedges have similar statistical properties
            
            # 1. Match means
            true_mean = true_features.mean(dim=0)  # [num_features]
            pred_mean = pred_valid.mean(dim=0)  # [num_features]
            mean_loss = F.mse_loss(pred_mean, true_mean)
            
            # 2. Match standard deviations (distribution spread)
            true_std = true_features.std(dim=0) + 1e-6  # [num_features]
            pred_std = pred_valid.std(dim=0) + 1e-6  # [num_features]
            std_loss = F.mse_loss(pred_std, true_std)
            
            # 3. Match higher moments (distribution shape)
            # Normalized third moment (skewness-like)
            true_centered = (true_features - true_mean) / true_std
            pred_centered = (pred_valid - pred_mean) / pred_std
            true_moment3 = (true_centered ** 3).mean(dim=0)
            pred_moment3 = (pred_centered ** 3).mean(dim=0)
            moment3_loss = F.mse_loss(pred_moment3, true_moment3)
            
            # Weighted combination (mean is most important)
            loss = 0.5 * mean_loss + 0.3 * std_loss + 0.2 * moment3_loss
            
            if torch.isnan(loss) or torch.isinf(loss):
                return torch.tensor(0.0, device=pred_features.device)
            return loss
        else:
            # Evaluation: use Wasserstein distance (Earth Mover's Distance)
            # Measures minimum cost to transform one distribution into another
            true_np = true_features.cpu().numpy()
            pred_np = pred_valid.cpu().numpy()
            
            total_w_dist = 0.0
            
            for feat_idx in range(num_features):
                w_dist = wasserstein_distance(true_np[:, feat_idx], pred_np[:, feat_idx])
                total_w_dist += w_dist
            
            # Average over features
            return torch.tensor(total_w_dist / num_features, device=pred_features.device)
    
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
        JET-LEVEL CONSISTENCY LOSS - PHYSICS-MOTIVATED CONSTRAINT
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
        # Match dtype of input tensors for scatter_add compatibility
        E_sum = torch.zeros(batch_size, device=device, dtype=E_phys.dtype).scatter_add_(0, valid_batch_indices, E_phys)
        px_sum = torch.zeros(batch_size, device=device, dtype=px_phys.dtype).scatter_add_(0, valid_batch_indices, px_phys)
        py_sum = torch.zeros(batch_size, device=device, dtype=py_phys.dtype).scatter_add_(0, valid_batch_indices, py_phys)
        pz_sum = torch.zeros(batch_size, device=device, dtype=pz_phys.dtype).scatter_add_(0, valid_batch_indices, pz_phys)
        
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
            pt_residual_scale = true_jet_pt.std()      # std of exp(log_pt)
            eta_residual_scale = eta_stats['std']    # std of eta (already physical)
            mass_residual_scale = true_jet_mass.std()  # std of exp(log_mass)
        else:  # minmax
            # For minmax: use range (max - min) of physical values
            pt_residual_scale = (true_jet_pt.max() - true_jet_pt.min())
            eta_residual_scale = eta_stats['max'] - eta_stats['min']
            mass_residual_scale = (true_jet_mass.max() - true_jet_mass.min())
        
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
    
    def _denorm_to_pt_eta_phi(self, x_norm, particle_norm_stats):
        """
        Denormalize particles and convert to (pt, eta, phi) coordinates.
        Supports both single jet [N, 4] and batched [B, N, 4] tensors.
        
        Args:
            x_norm: [N, 4] or [B, N, 4] normalized (E, px, py, pz)
            particle_norm_stats: dict with normalization stats for each component
        
        Returns:
            E, px, py, pz, pt, eta, phi: tensors matching input shape
        """
        device = x_norm.device
        method = particle_norm_stats['E']['method']
        
        # Denormalize based on method
        if method == 'zscore':
            E = x_norm[..., 0] * particle_norm_stats['E']['std'] + particle_norm_stats['E']['mean']
            px = x_norm[..., 1] * particle_norm_stats['px']['std'] + particle_norm_stats['px']['mean']
            py = x_norm[..., 2] * particle_norm_stats['py']['std'] + particle_norm_stats['py']['mean']
            pz = x_norm[..., 3] * particle_norm_stats['pz']['std'] + particle_norm_stats['pz']['mean']
        else:  # minmax
            E_range = particle_norm_stats['E']['max'] - particle_norm_stats['E']['min']
            px_range = particle_norm_stats['px']['max'] - particle_norm_stats['px']['min']
            py_range = particle_norm_stats['py']['max'] - particle_norm_stats['py']['min']
            pz_range = particle_norm_stats['pz']['max'] - particle_norm_stats['pz']['min']
            
            E = x_norm[..., 0] * E_range + particle_norm_stats['E']['min']
            px = x_norm[..., 1] * px_range + particle_norm_stats['px']['min']
            py = x_norm[..., 2] * py_range + particle_norm_stats['py']['min']
            pz = x_norm[..., 3] * pz_range + particle_norm_stats['pz']['min']
        
        # Compute pt, eta, phi
        pt = torch.sqrt(px**2 + py**2 + 1e-12)
        phi = torch.atan2(py, px)
        eta = torch.asinh(pz / (pt + 1e-12))
        
        return E, px, py, pz, pt, eta, phi

    
    def _compute_edge_observables_vectorized(self, E, px, py, pz, pt, eta, phi):
        """
        Compute edge observables for all pairs in a single jet (fully vectorized, tensor-based).
        
        Args:
            E, px, py, pz, pt, eta, phi: torch tensors [N] for N particles (already denormalized)
        
        Returns:
            dict with 'ln_delta', 'ln_kT', 'ln_z', 'ln_m2' as torch tensors, or None if < 2 particles
        """
        N = pt.shape[0]
        if N < 2:
            return None
        
        # Generate all pairs (i < j) - vectorized
        i_idx, j_idx = torch.triu_indices(N, N, offset=1, device=pt.device)
        
        # Angular distance with phi wrap-around
        delta_eta = eta[i_idx] - eta[j_idx]
        delta_phi_raw = phi[i_idx] - phi[j_idx]
        delta_phi = torch.where(
            torch.abs(delta_phi_raw) <= math.pi,
            torch.abs(delta_phi_raw),
            2 * math.pi - torch.abs(delta_phi_raw)
        )
        delta_R = torch.sqrt(delta_eta**2 + delta_phi**2 + 1e-12)
        
        # Compute observables (vectorized)
        pt_min = torch.minimum(pt[i_idx], pt[j_idx])
        k_T = pt_min * delta_R
        z = pt_min / (pt[i_idx] + pt[j_idx] + 1e-12)
        
        # Invariant mass squared (using already-computed 4-momenta, no recalculation!)
        E_sum = E[i_idx] + E[j_idx]
        px_sum = px[i_idx] + px[j_idx]
        py_sum = py[i_idx] + py[j_idx]
        pz_sum = pz[i_idx] + pz[j_idx]
        m2 = torch.clamp(E_sum**2 - (px_sum**2 + py_sum**2 + pz_sum**2), min=1e-12)
        
        # Log-transform (IRC-safe)
        ln_delta = torch.log(delta_R + 1e-12)
        ln_kT = torch.log(k_T + 1e-12)
        ln_z = torch.log(z + 1e-12)
        ln_m2 = torch.log(m2)
        
        # Mask NaN/Inf (single operation)
        valid = torch.isfinite(ln_delta) & torch.isfinite(ln_kT) & torch.isfinite(ln_z) & torch.isfinite(ln_m2)
        
        return {
            'ln_delta': ln_delta[valid],
            'ln_kT': ln_kT[valid],
            'ln_z': ln_z[valid],
            'ln_m2': ln_m2[valid]
        }
    
    def _edge_hyperedge_wasserstein_loss(self, batch, pred_particles, pred_mask):
        """
        FULLY VECTORIZED & BATCHED: Compute Wasserstein distance between edge/hyperedge distributions.
        
        Edge Loss (2-pt correlations):
        - Uses PRE-COMPUTED features from dataset (batch.edge_attr):
          [2pt_EEC, ln_delta, ln_kT, ln_z, ln_m2]
        - Computes same features from generated particles
        - Wasserstein distance for distribution matching
        
        Hyperedge Loss (3-pt+ correlations):
        - Computes 3-point+ EEC for both real and generated particles
        - Uses batched EEC computation (one call per N-point)
        - Wasserstein distance on EEC histograms
        
        Optimization highlights:
        - No recomputation of real edge features (uses dataset)
        - Stays in torch tensors until Wasserstein call
        - Batched EEC matching graph_constructor.py pattern
        - Minimal loops (only for PyG Batch extraction)
        
        Args:
            batch: PyG Batch with:
                - x: [N_total, 4] real particles (normalized)
                - edge_attr: [N_edges, 5] pre-computed edge features
                - n_particles: [B] particle counts
                - particle_norm_stats: normalization statistics
            pred_particles: [B, max_P, 4] normalized predicted (E, px, py, pz)
            pred_mask: [B, max_P] float mask (sigmoid output) or bool mask
        
        Returns:
            scalar torch.Tensor (edge + hyperedge distribution loss)
        """
        if not self.use_distribution_loss:
            return torch.tensor(0.0, device=pred_particles.device)
        
        batch_size = batch.num_graphs
        device = pred_particles.device
        particle_norm_stats = batch.particle_norm_stats
        
        # Get edge/hyperedge normalization stats from batch (attached by collate_with_stats in train.py)
        edge_norm_stats = getattr(batch, 'edge_norm_stats', {})
        hyperedge_norm_stats = getattr(batch, 'hyperedge_norm_stats', {})
        
        # Convert pred_mask to boolean
        if pred_mask.dtype in [torch.float32, torch.float16]:
            pred_mask = pred_mask > 0.5
        
        # ═══════════════════════════════════════════════════════════════════════
        # STEP 1: EXTRACT REAL AND GENERATED PARTICLES
        # ═══════════════════════════════════════════════════════════════════════
        
        real_particles_list = []
        gen_particles_list = []
        
        cumulative = 0
        for i in range(batch_size):
            n_true = batch.n_particles[i].item()
            
            # Real particles [n_true, 4]
            real_norm = batch.x[cumulative:cumulative + n_true]
            cumulative += n_true
            
            # Generated particles [n_pred, 4]
            gen_norm = pred_particles[i][pred_mask[i]]
            
            # Skip jets with < 2 particles
            if real_norm.shape[0] >= 2:
                real_particles_list.append(real_norm)
            if gen_norm.shape[0] >= 2:
                gen_particles_list.append(gen_norm)
        
        if len(real_particles_list) == 0 or len(gen_particles_list) == 0:
            return torch.tensor(0.0, device=device)
        
        # ═══════════════════════════════════════════════════════════════════════
        # STEP 2: EXTRACT REAL EDGE FEATURES 
        # ═══════════════════════════════════════════════════════════════════════
        
        # Real edge features are already computed in dataset: [2pt_EEC, ln_delta, ln_kT, ln_z, ln_m2]
        # Extract them directly from batch.edge_attr 
        real_edge_features = batch.edge_attr  # [total_edges, 5]
        
        # ═══════════════════════════════════════════════════════════════════════
        # STEP 3: COMPUTE GENERATED EDGE FEATURES (from predicted particles)
        # ═══════════════════════════════════════════════════════════════════════
        
        gen_edge_obs = []
        gen_pt_eta_phi_list = []  # For 3-pt+ EEC computation
        
        # Process generated particles
        for gen_norm in gen_particles_list:
            E, px, py, pz, pt, eta, phi = self._denorm_to_pt_eta_phi(gen_norm, particle_norm_stats)
            
            # Compute edge observables (stays in torch tensors - fast!)
            obs = self._compute_edge_observables_vectorized(E, px, py, pz, pt, eta, phi)
            if obs is not None:
                gen_edge_obs.append(obs)
            
            # Store (pt, eta, phi) for 3-pt+ EEC computation
            gen_pt_eta_phi = torch.stack([pt, eta, phi], dim=1)
            gen_pt_eta_phi_list.append(gen_pt_eta_phi)
        
        # Also prepare real particles for 3-pt+ EEC (only denormalize for EEC, not edge features)
        real_pt_eta_phi_list = []
        for real_norm in real_particles_list:
            E, px, py, pz, pt, eta, phi = self._denorm_to_pt_eta_phi(real_norm, particle_norm_stats)
            real_pt_eta_phi = torch.stack([pt, eta, phi], dim=1)
            real_pt_eta_phi_list.append(real_pt_eta_phi)
        
        # ═══════════════════════════════════════════════════════════════════════
        # STEP 4: EDGE WASSERSTEIN DISTANCES (2-pt EEC + edge observables)
        # ═══════════════════════════════════════════════════════════════════════
        
        edge_loss = 0.0
        edge_count = 0
        
        if real_edge_features.shape[0] > 0 and gen_edge_obs:
            # Real edge features: [total_edges, 5] = [2pt_EEC, ln_delta, ln_kT, ln_z, ln_m2]
            # Concatenate generated edge observables
            gen_edges_stacked = {key: torch.cat([obs[key] for obs in gen_edge_obs]) 
                                for key in gen_edge_obs[0].keys()}
            
            # Feature names in dataset order
            feature_names = ['2pt_EEC', 'ln_delta', 'ln_kT', 'ln_z', 'ln_m2']
            
            # Compute Wasserstein for each observable
            for idx, key in enumerate(feature_names):
                # Real values from pre-computed dataset features (already normalized)
                real_vals_norm = real_edge_features[:, idx]
                
                # Generated values - compute and NORMALIZE to match real
                if key == '2pt_EEC':
                    # Compute 2-pt EEC for generated particles
                    # Convert generated pt/eta/phi to numpy for EEC library
                    gen_pt_eta_phi_numpy = [x.cpu().detach().numpy() for x in gen_pt_eta_phi_list]
                    gen_2pt_eec = get_eec_ls_values(gen_pt_eta_phi_numpy, N=2, bins=self.eec_prop[1], 
                                                   axis_range=self.eec_prop[2], print_every=0)
                    gen_eec_hist = np.array(gen_2pt_eec.get_hist_errs(0, False)[0])
                    gen_eec2_hist_raw = gen_eec_hist.copy()  # Store raw histogram for hyperedge division
                    with np.errstate(divide='ignore'):
                        gen_eec_hist = np.where(gen_eec_hist > 0, np.log(gen_eec_hist), 0.0)
                    
                    # Convert to tensor and NORMALIZE
                    gen_vals = torch.tensor(gen_eec_hist, device=device, dtype=pred_particles.dtype)
                    if key in edge_norm_stats:
                        stats = edge_norm_stats[key]
                        if stats['method'] == 'zscore':
                            gen_vals = (gen_vals - stats['mean']) / (stats['std'] + 1e-8)
                        else:  # minmax
                            gen_vals = (gen_vals - stats['min']) / (stats['max'] - stats['min'] + 1e-8)
                    gen_vals = gen_vals.cpu().numpy()
                else:
                    # Use computed edge observables (physical space)
                    edge_key_map = {'ln_delta': 'ln_delta', 'ln_kT': 'ln_kT', 'ln_z': 'ln_z', 'ln_m2': 'ln_m2'}
                    gen_vals = gen_edges_stacked[edge_key_map[key]]
                    
                    # NORMALIZE generated values to match real
                    if key in edge_norm_stats:
                        stats = edge_norm_stats[key]
                        if stats['method'] == 'zscore':
                            gen_vals = (gen_vals - stats['mean']) / (stats['std'] + 1e-8)
                        else:  # minmax
                            gen_vals = (gen_vals - stats['min']) / (stats['max'] - stats['min'] + 1e-8)
                    gen_vals = gen_vals.cpu().numpy()
                
                real_vals = real_vals_norm.cpu().numpy()
                
                if len(real_vals) >= 2 and len(gen_vals) >= 2:
                    # Both real and generated are now in normalized space
                    W = wasserstein_distance(real_vals, gen_vals)
                    edge_loss += W
                    edge_count += 1
            
            if edge_count > 0:
                edge_loss = edge_loss / edge_count
        
        # ═══════════════════════════════════════════════════════════════════════
        # STEP 5: HYPEREDGE (3-pt+ EEC) WASSERSTEIN - BATCHED COMPUTATION
        # ═══════════════════════════════════════════════════════════════════════
        
        hyperedge_loss = 0.0
        
        if self.hyperedge_distribution_weight > 0:
            try:
                # Real 3-pt+ EEC values are PRE-COMPUTED in dataset (batch.hyperedge_attr)
                # hyperedge_attr columns: [3pt_EEC, 4pt_EEC, ...] depending on N_points config
                real_hyperedge_features = batch.hyperedge_attr  # [total_hyperedges, num_N_points]
                
                if real_hyperedge_features.shape[0] > 0:
                    N_points = self.eec_prop[0]  # e.g., [2, 3, 4]
                    bins = self.eec_prop[1]
                    axis_range = self.eec_prop[2]
                    
                    # Filter to only 3-pt+ EEC (2-pt is handled in edge loss)
                    N_points_hyperedge = [n for n in N_points if n >= 3]
                    
                    if len(N_points_hyperedge) > 0:
                        # Reuse 2-point EEC from edge loss (already computed as gen_eec2_hist_raw)
                        gen_eec2_hist = gen_eec2_hist_raw
                        
                        eec_count = 0
                        for idx, n in enumerate(N_points_hyperedge):
                            # Real hyperedge EEC values (already processed: log(N-pt/2-pt) and normalized)
                            real_eec_values = real_hyperedge_features[:, idx]
                            
                            # Generated N-point EEC: compute histogram
                            gen_eec_n = get_eec_ls_values(gen_pt_eta_phi_numpy, N=n, bins=bins, 
                                                         axis_range=axis_range, print_every=0)
                            gen_eec_hist = np.array(gen_eec_n.get_hist_errs(0, False)[0])
                            
                            # Apply proper hyperedge processing:
                            # 1. Divide by 2-point EEC (element-wise)
                            gen_eec_hist = np.divide(gen_eec_hist, gen_eec2_hist,
                                                    out=np.zeros_like(gen_eec_hist),
                                                    where=gen_eec2_hist != 0)
                            
                            # 2. Log transform
                            with np.errstate(divide='ignore'):
                                gen_eec_hist = np.where(gen_eec_hist > 0, np.log(gen_eec_hist), 0.0)
                            
                            # 3. Convert to tensor and NORMALIZE
                            gen_vals = torch.tensor(gen_eec_hist, device=device, dtype=pred_particles.dtype)
                            eec_key = f'{n}pt_EEC'
                            if eec_key in hyperedge_norm_stats:
                                stats = hyperedge_norm_stats[eec_key]
                                if stats['method'] == 'zscore':
                                    gen_vals = (gen_vals - stats['mean']) / (stats['std'] + 1e-8)
                                else:  # minmax
                                    gen_vals = (gen_vals - stats['min']) / (stats['max'] - stats['min'] + 1e-8)
                            
                            gen_vals = gen_vals.cpu().numpy()
                            real_vals = real_eec_values.cpu().numpy()
                            
                            if len(real_vals) > 0 and len(gen_vals) > 0:
                                # Both real and generated are now in normalized space
                                W_eec = wasserstein_distance(real_vals, gen_vals)
                                hyperedge_loss += W_eec
                                eec_count += 1
                        
                        if eec_count > 0:
                            hyperedge_loss = hyperedge_loss / eec_count
                    
            except Exception as e:
                # Silent fail for EEC (non-critical)
                hyperedge_loss = 0.0
        
        # ═══════════════════════════════════════════════════════════════════════
        # STEP 5: RETURN SEPARATE LOSSES (weighting done in compute_loss)
        # ═══════════════════════════════════════════════════════════════════════
        
        return {
            'edge': torch.tensor(edge_loss, device=device, dtype=pred_particles.dtype),
            'hyperedge': torch.tensor(hyperedge_loss, device=device, dtype=pred_particles.dtype)
        }
    
    def _edge_hyperedge_mse_loss(self, batch, pred_particles, pred_mask):
        """
        Compare edge/hyperedge feature distributions using histogram-based MSE.
        
        Key Advantages over Wasserstein:
        - 10-50x faster (stays entirely on GPU, no CPU transfers)
        
        Methodology:
        1. For edge observables (ln_delta, ln_kT, ln_z, ln_m2):
           - Create histograms from real and generated values
           - Normalize to probability distributions (sum to 1)
           - Compute MSE between histogram bins
        
        2. For EEC features (2pt_EEC, 3pt_EEC, etc.):
           - Already histograms from get_eec_ls_values
           - Normalize to probability distributions
           - Compute MSE between histogram bins
        
        Edge Loss (2-pt correlations):
        - Real: batch.edge_attr [total_edges, 5]
        - Generated: Compute observables → histogram → normalize → MSE
        - Features: [2pt_EEC, ln_delta, ln_kT, ln_z, ln_m2]
        
        Hyperedge Loss (3-pt+ EEC):
        - Real: batch.hyperedge_attr [total_hyperedges, num_N_points]
        - Generated: Compute EEC histograms → normalize → MSE
        - Features: [3pt_EEC, 4pt_EEC, ...]
        
        Args:
            batch: PyG Batch with edge/hyperedge features
            pred_particles: [B, max_P, 4] normalized predicted particles
            pred_mask: [B, max_P] validity mask
        
        Returns:
            dict: {'edge': tensor, 'hyperedge': tensor} - MSE losses on histogram bins
        """
        # TEMPORARY: Flag to enable/disable plotting (set to False to disable)
        PLOT_HISTOGRAMS = False
        
        if not self.use_distribution_loss:
            return {
                'edge': torch.tensor(0.0, device=pred_particles.device),
                'hyperedge': torch.tensor(0.0, device=pred_particles.device)
            }
        
        batch_size = batch.num_graphs
        device = pred_particles.device
        particle_norm_stats = batch.particle_norm_stats
        method = particle_norm_stats['E']['method']
        
        # Get edge/hyperedge normalization stats from batch
        edge_norm_stats = getattr(batch, 'edge_norm_stats', {})
        hyperedge_norm_stats = getattr(batch, 'hyperedge_norm_stats', {})
        
        # Convert pred_mask to boolean
        if pred_mask.dtype in [torch.float32, torch.float16]:
            pred_mask = pred_mask > 0.5
        
        # ═══════════════════════════════════════════════════════════════════════
        # STEP 1: EXTRACT REAL AND GENERATED PARTICLES
        # ═══════════════════════════════════════════════════════════════════════
        
        real_particles_list = []
        gen_particles_list = []
        
        cumulative = 0
        for i in range(batch_size):
            n_true = batch.n_particles[i].item()
            
            # Real particles [n_true, 4]
            real_norm = batch.x[cumulative:cumulative + n_true]
            cumulative += n_true
            
            # Generated particles [n_pred, 4]
            gen_norm = pred_particles[i][pred_mask[i]]
            
            # Skip jets with < 2 particles
            if real_norm.shape[0] >= 2:
                real_particles_list.append(real_norm)
            if gen_norm.shape[0] >= 2:
                gen_particles_list.append(gen_norm)
        
        if len(real_particles_list) == 0 or len(gen_particles_list) == 0:
            return {
                'edge': torch.tensor(0.0, device=device),
                'hyperedge': torch.tensor(0.0, device=device)
            }
        
        # ═══════════════════════════════════════════════════════════════════════
        # STEP 2: EXTRACT REAL EDGE FEATURES (already normalized)
        # ═══════════════════════════════════════════════════════════════════════
        
        # Real edge features are pre-computed and normalized: [2pt_EEC, ln_delta, ln_kT, ln_z, ln_m2]
        real_edge_features = batch.edge_attr  # [total_edges, 5]
        
        # ═══════════════════════════════════════════════════════════════════════
        # STEP 3: COMPUTE GENERATED EDGE FEATURES (from predicted particles)
        # ═══════════════════════════════════════════════════════════════════════
        
        gen_edge_obs = []
        gen_pt_eta_phi_list = []  # For 3-pt+ EEC computation
        
        # Process generated particles
        for gen_norm in gen_particles_list:
            E, px, py, pz, pt, eta, phi = self._denorm_to_pt_eta_phi(gen_norm, particle_norm_stats)
            
            # Compute edge observables (stays in torch tensors - fast!)
            obs = self._compute_edge_observables_vectorized(E, px, py, pz, pt, eta, phi)
            if obs is not None:
                gen_edge_obs.append(obs)
            
            # Store (pt, eta, phi) for 3-pt+ EEC computation
            gen_pt_eta_phi = torch.stack([pt, eta, phi], dim=1)
            gen_pt_eta_phi_list.append(gen_pt_eta_phi)
        
        # ═══════════════════════════════════════════════════════════════════════
        # STEP 4: EDGE MSE LOSS (2-pt EEC + edge observables) - HISTOGRAM BASED
        # ═══════════════════════════════════════════════════════════════════════
        
        edge_loss = 0.0
        edge_count = 0
        
        if real_edge_features.shape[0] > 0 and gen_edge_obs:
            # Concatenate generated edge observables
            gen_edges_stacked = {key: torch.cat([obs[key] for obs in gen_edge_obs]) 
                                for key in gen_edge_obs[0].keys()}
            
            # Feature names in dataset order
            feature_names = ['2pt_EEC', 'ln_delta', 'ln_kT', 'ln_z', 'ln_m2']
            
            # Number of bins for histogram comparison (same for all features)
            num_bins = 500
            
            # Compute MSE for each observable using histogram comparison
            for idx, key in enumerate(feature_names):
                # Real values (already normalized) - shape: [n_real] - SCALAR VALUES PER EDGE
                real_vals_norm = real_edge_features[:, idx]
                
                # Generated values (need to be normalized)
                if key == '2pt_EEC':
                    # Compute generated EEC histogram (store for reuse in hyperedge loss)
                    gen_pt_eta_phi_numpy = [x.cpu().detach().numpy() for x in gen_pt_eta_phi_list]
                    gen_2pt_eec = get_eec_ls_values(gen_pt_eta_phi_numpy, N=2, bins=self.eec_prop[1], 
                                               axis_range=self.eec_prop[2], print_every=0)
                    gen_eec_hist = np.array(gen_2pt_eec.get_hist_errs(0, False)[0])
                    gen_eec2_hist_raw = gen_eec_hist.copy()  # Store raw histogram before log for hyperedge division
                    with np.errstate(divide='ignore'):
                        gen_eec_hist = np.where(gen_eec_hist > 0, np.log(gen_eec_hist), 0.0)
                    
                    # Convert to tensor
                    gen_vals = torch.tensor(gen_eec_hist, device=device, dtype=pred_particles.dtype)
                    
                    # NORMALIZE generated EEC values using same method as real data
                    stats = edge_norm_stats[key]
                    if stats['method'] == 'zscore':
                        gen_vals_norm = (gen_vals - stats['mean']) / (stats['std'] + 1e-8)
                    else:  # minmax
                        gen_vals_norm = (gen_vals - stats['min']) / (stats['max'] - stats['min'] + 1e-8)
                    # Ensure float32 for histc
                    gen_vals_norm = gen_vals_norm.float()
                    
                    # Create histogram from real EEC scalar values (already normalized)
                    # Get value range from combined data
                    all_vals = torch.cat([real_vals_norm, gen_vals_norm.to(device)])
                    min_val = all_vals.min().item()
                    max_val = all_vals.max().item()
                    
                    # Create histograms from scalar values
                    real_hist = torch.histc(real_vals_norm, bins=num_bins, min=min_val, max=max_val)
                    gen_hist = torch.histc(gen_vals_norm.to(device), bins=num_bins, min=min_val, max=max_val)
                    
                    # Normalize both histograms to probability distributions
                    real_hist_norm = real_hist / (real_hist.sum() + 1e-8)
                    gen_hist_norm = gen_hist / (gen_hist.sum() + 1e-8)
                    
                else:
                    # Use computed edge observables (physical space) - shape: [n_gen]
                    edge_key_map = {'ln_delta': 'ln_delta', 'ln_kT': 'ln_kT', 'ln_z': 'ln_z', 'ln_m2': 'ln_m2'}
                    gen_vals = gen_edges_stacked[edge_key_map[key]]  # Keep as tensor
                    
                    # NORMALIZE generated values using same method as real data
                    if key in edge_norm_stats:
                        stats = edge_norm_stats[key]
                        if stats['method'] == 'zscore':
                            gen_vals_norm = (gen_vals - stats['mean']) / (stats['std'] + 1e-8)
                        else:  # minmax
                            gen_vals_norm = (gen_vals - stats['min']) / (stats['max'] - stats['min'] + 1e-8)
                        # Ensure float32 for histc
                        gen_vals_norm = gen_vals_norm.float()
                    
                    # Create histograms for real and generated (both now normalized)
                    # Get value range from combined data
                    all_vals = torch.cat([real_vals_norm, gen_vals_norm.to(device)])
                    min_val = all_vals.min().item()
                    max_val = all_vals.max().item()
                    
                    # Create histograms with same bins
                    real_hist = torch.histc(real_vals_norm, bins=num_bins, min=min_val, max=max_val)
                    gen_hist = torch.histc(gen_vals_norm.to(device), bins=num_bins, min=min_val, max=max_val)
                    
                    # Normalize histograms to probability distributions (sum to 1)
                    real_hist_norm = real_hist / (real_hist.sum() + 1e-8)
                    gen_hist_norm = gen_hist / (gen_hist.sum() + 1e-8)
                
                # Compute MSE between normalized histograms
                n_real_bins = real_hist_norm.shape[0]
                n_gen_bins = gen_hist_norm.shape[0]
                
                # For EEC, histograms should already be same size (from get_eec_ls_values)
                # For edge observables, they're guaranteed same size (num_bins)
                if n_real_bins == n_gen_bins and n_real_bins > 0:
                    mse = F.mse_loss(gen_hist_norm, real_hist_norm)
                    edge_loss += mse
                    edge_count += 1
            
            if edge_count > 0:
                edge_loss = edge_loss / edge_count
        
        # ═══════════════════════════════════════════════════════════════════════
        # TEMPORARY: PLOT EDGE HISTOGRAMS FOR DEBUGGING
        # ═══════════════════════════════════════════════════════════════════════
        
        if PLOT_HISTOGRAMS and real_edge_features.shape[0] > 0 and gen_edge_obs:
            import matplotlib.pyplot as plt
            import os
            
            # Create plots directory if it doesn't exist
            os.makedirs('plots/histogram_debug', exist_ok=True)
            
            # Create figure with subplots for all edge features
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            axes = axes.flatten()
            
            plot_idx = 0
            for idx, key in enumerate(feature_names):
                if plot_idx >= len(axes):
                    break
                
                ax = axes[plot_idx]
                
                # Get real values
                real_vals_norm = real_edge_features[:, idx]
                
                if key == '2pt_EEC':
                    # Real: scalar values → create histogram
                    gen_pt_eta_phi_numpy = [x.cpu().detach().numpy() for x in gen_pt_eta_phi_list]
                    gen_eec = get_eec_ls_values(gen_pt_eta_phi_numpy, N=2, bins=self.eec_prop[1], 
                                               axis_range=self.eec_prop[2], print_every=0)
                    gen_eec_hist = np.array(gen_eec.get_hist_errs(0, False)[0])
                    with np.errstate(divide='ignore'):
                        gen_eec_hist = np.where(gen_eec_hist > 0, np.log(gen_eec_hist), 0.0)
                    
                    gen_vals_plot = torch.tensor(gen_eec_hist, device=device, dtype=torch.float32)
                    
                    # NORMALIZE generated EEC values using same method as real data (FOR PLOTTING)
                    if key in edge_norm_stats:
                        stats = edge_norm_stats[key]
                        if stats['method'] == 'zscore':
                            gen_vals_plot = (gen_vals_plot - stats['mean']) / (stats['std'] + 1e-8)
                        else:  # minmax
                            gen_vals_plot = (gen_vals_plot - stats['min']) / (stats['max'] - stats['min'] + 1e-8)
                    
                    all_vals = torch.cat([real_vals_norm, gen_vals_plot])
                    min_val = all_vals.min().item()
                    max_val = all_vals.max().item()
                    
                    real_hist = torch.histc(real_vals_norm, bins=num_bins, min=min_val, max=max_val)
                    gen_hist = torch.histc(gen_vals_plot, bins=num_bins, min=min_val, max=max_val)
                    
                    real_hist_norm_plot = real_hist / (real_hist.sum() + 1e-8)
                    gen_hist_norm_plot = gen_hist / (gen_hist.sum() + 1e-8)
                    
                    # Plot
                    bin_edges = np.linspace(min_val, max_val, num_bins + 1)
                    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                    width = (bin_edges[1] - bin_edges[0]) * 0.4
                    
                    ax.bar(bin_centers - width/2, real_hist_norm_plot.cpu().numpy(), width=width,
                           alpha=0.6, label=f'Real (n={len(real_vals_norm)})', color='blue')
                    ax.bar(bin_centers + width/2, gen_hist_norm_plot.cpu().numpy(), width=width,
                           alpha=0.6, label=f'Gen (n={len(gen_vals_plot)}) NORM', color='red')
                    
                    ax.set_title(f'{key} (NORMALIZED)\nReal: {len(real_vals_norm)} scalars → {num_bins} bins\nGen: {len(gen_vals_plot)} values → {num_bins} bins')
                
                else:
                    # Other edge observables (ln_delta, ln_kT, ln_z, ln_m2)
                    gen_vals_plot = gen_edges_stacked[edge_key_map[key]]
                    
                    # NORMALIZE generated values using same method as real data (FOR PLOTTING)
                    if key in edge_norm_stats:
                        stats = edge_norm_stats[key]
                        if stats['method'] == 'zscore':
                            gen_vals_plot = (gen_vals_plot - stats['mean']) / (stats['std'] + 1e-8)
                        else:  # minmax
                            gen_vals_plot = (gen_vals_plot - stats['min']) / (stats['max'] - stats['min'] + 1e-8)
                        gen_vals_plot = gen_vals_plot.float()
                    
                    all_vals = torch.cat([real_vals_norm, gen_vals_plot.to(device)])
                    min_val = all_vals.min().item()
                    max_val = all_vals.max().item()
                    
                    real_hist = torch.histc(real_vals_norm, bins=num_bins, min=min_val, max=max_val)
                    gen_hist = torch.histc(gen_vals_plot.to(device), bins=num_bins, min=min_val, max=max_val)
                    
                    real_hist_norm_plot = real_hist / (real_hist.sum() + 1e-8)
                    gen_hist_norm_plot = gen_hist / (gen_hist.sum() + 1e-8)
                    
                    # Plot
                    bin_edges = np.linspace(min_val, max_val, num_bins + 1)
                    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                    width = (bin_edges[1] - bin_edges[0]) * 0.4
                    
                    ax.bar(bin_centers - width/2, real_hist_norm_plot.cpu().numpy(), width=width,
                           alpha=0.6, label=f'Real (n={len(real_vals_norm)})', color='blue')
                    ax.bar(bin_centers + width/2, gen_hist_norm_plot.cpu().numpy(), width=width,
                           alpha=0.6, label=f'Gen (n={len(gen_vals_plot)}) NORM', color='red')
                    
                    ax.set_title(f'{key} (NORMALIZED)\nReal: {len(real_vals_norm)} scalars\nGen: {len(gen_vals_plot)} scalars')
                
                ax.set_xlabel(f'{key} value')
                ax.set_ylabel('Probability Density')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                plot_idx += 1
            
            # Hide unused subplot
            if plot_idx < len(axes):
                axes[plot_idx].axis('off')
            
            plt.tight_layout()
            plt.savefig('plots/histogram_debug/edge_features.png', dpi=150, bbox_inches='tight')
            plt.close()
            print(f"📊 Saved edge feature histograms to plots/histogram_debug/edge_features.png")
        
        # ═══════════════════════════════════════════════════════════════════════
        # STEP 5: HYPEREDGE (3-pt+ EEC) MSE LOSS - HISTOGRAM BASED
        # ═══════════════════════════════════════════════════════════════════════
        
        hyperedge_loss = 0.0
        
        if self.hyperedge_distribution_weight > 0:
            try:
                # Real 3-pt+ EEC values (pre-computed and normalized)
                real_hyperedge_features = batch.hyperedge_attr  # [total_hyperedges, num_N_points]
                
                if real_hyperedge_features.shape[0] > 0:
                    N_points = self.eec_prop[0]  # e.g., [2, 3, 4]
                    bins = self.eec_prop[1]
                    axis_range = self.eec_prop[2]
                    
                    # Filter to only 3-pt+ EEC (2-pt is handled in edge loss)
                    N_points_hyperedge = [n for n in N_points if n >= 3]
                    
                    if len(N_points_hyperedge) > 0:
                        # Number of bins for histogram comparison
                        num_bins = 500
                        
                        # Reuse 2-point EEC from edge loss (already computed as gen_eec2_hist_raw)
                        gen_eec2_hist = gen_eec2_hist_raw
                        
                        eec_count = 0
                        for idx, n in enumerate(N_points_hyperedge):
                            # Real hyperedge EEC values: [total_hyperedges] - scalar values per hyperedge
                            # Already processed: log(N-point / 2-point) and normalized
                            real_eec_values = real_hyperedge_features[:, idx]
                            
                            # Generated N-point EEC: compute histogram
                            gen_enc = get_eec_ls_values(gen_pt_eta_phi_numpy, N=n, bins=bins, 
                                                       axis_range=axis_range, print_every=0)
                            gen_enc_hist = np.array(gen_enc.get_hist_errs(0, False)[0])
                            
                            # Divide by 2-point EEC (element-wise)
                            gen_enc_hist = np.divide(gen_enc_hist, gen_eec2_hist,
                                                    out=np.zeros_like(gen_enc_hist),
                                                    where=gen_eec2_hist != 0)
                            
                            # Log transform
                            with np.errstate(divide='ignore'):
                                gen_enc_hist = np.where(gen_enc_hist > 0, np.log(gen_enc_hist), 0.0)
                            
                            # Convert to tensor
                            gen_vals = torch.tensor(gen_enc_hist, device=device, dtype=torch.float32)
                            
                            # NORMALIZE generated values using same method as real data
                            eec_key = f'{n}pt_EEC'
                            if eec_key in hyperedge_norm_stats:
                                stats = hyperedge_norm_stats[eec_key]
                                if stats['method'] == 'zscore':
                                    gen_vals = (gen_vals - stats['mean']) / (stats['std'] + 1e-8)
                                else:  # minmax
                                    gen_vals = (gen_vals - stats['min']) / (stats['max'] - stats['min'] + 1e-8)
                            
                            # Create histogram from real EEC scalar values (already normalized)
                            # Get value range from combined data
                            all_vals = torch.cat([real_eec_values, gen_vals])
                            min_val = all_vals.min().item()
                            max_val = all_vals.max().item()
                            
                            # Create histograms from scalar values
                            real_hist = torch.histc(real_eec_values, bins=num_bins, min=min_val, max=max_val)
                            gen_hist = torch.histc(gen_vals, bins=num_bins, min=min_val, max=max_val)
                            
                            # Normalize both histograms to probability distributions
                            real_hist_norm = real_hist / (real_hist.sum() + 1e-8)
                            gen_hist_norm = gen_hist / (gen_hist.sum() + 1e-8)
                            
                            # Histograms should now have same size (num_bins)
                            n_real_bins = real_hist_norm.shape[0]
                            n_gen_bins = gen_hist_norm.shape[0]
                            
                            if n_real_bins == n_gen_bins and n_real_bins > 0:
                                # Compute MSE between normalized histograms
                                mse_eec = F.mse_loss(gen_hist_norm, real_hist_norm)
                                hyperedge_loss += mse_eec
                                eec_count += 1
                        
                        if eec_count > 0:
                            hyperedge_loss = hyperedge_loss / eec_count
                    
            except Exception as e:
                # Silent fail for EEC (non-critical)
                hyperedge_loss = 0.0
        
        # ═══════════════════════════════════════════════════════════════════════
        # TEMPORARY: PLOT HYPEREDGE HISTOGRAMS FOR DEBUGGING
        # ═══════════════════════════════════════════════════════════════════════
        
        if PLOT_HISTOGRAMS and self.hyperedge_distribution_weight > 0:
            import matplotlib.pyplot as plt
            import os
            
            try:
                real_hyperedge_features = batch.hyperedge_attr
                
                if real_hyperedge_features.shape[0] > 0:
                    N_points = self.eec_prop[0]
                    bins = self.eec_prop[1]
                    axis_range = self.eec_prop[2]
                    N_points_hyperedge = [n for n in N_points if n >= 3]
                    
                    if len(N_points_hyperedge) > 0 and len(gen_pt_eta_phi_list) > 0:
                        # Create figure
                        n_features = len(N_points_hyperedge)
                        fig, axes = plt.subplots(1, n_features, figsize=(6*n_features, 5))
                        if n_features == 1:
                            axes = [axes]
                        
                        # Reuse 2-point EEC from edge loss (already computed as gen_eec2_hist_raw)
                        gen_eec2_hist = gen_eec2_hist_raw
                        
                        for plot_idx, n in enumerate(N_points_hyperedge):
                            if plot_idx >= len(axes):
                                break
                            
                            ax = axes[plot_idx]
                            
                            # Real hyperedge EEC values (scalars, already normalized)
                            # Already processed: log(N-point / 2-point) and normalized
                            real_eec_values = real_hyperedge_features[:, plot_idx]
                            
                            # Generated N-point EEC histogram
                            gen_eec = get_eec_ls_values(gen_pt_eta_phi_numpy, N=n, bins=bins,
                                                       axis_range=axis_range, print_every=0)
                            gen_eec_hist = np.array(gen_eec.get_hist_errs(0, False)[0])
                            
                            # Divide by 2-point EEC (element-wise)
                            gen_eec_hist = np.divide(gen_eec_hist, gen_eec2_hist,
                                                    out=np.zeros_like(gen_eec_hist),
                                                    where=gen_eec2_hist != 0)
                            
                            # Log transform
                            with np.errstate(divide='ignore'):
                                gen_eec_hist = np.where(gen_eec_hist > 0, np.log(gen_eec_hist), 0.0)
                            
                            gen_vals_plot = torch.tensor(gen_eec_hist, device=device, dtype=torch.float32)
                            
                            # NORMALIZE generated values using same method as real data (FOR PLOTTING)
                            eec_key = f'{n}pt_EEC'
                            if eec_key in hyperedge_norm_stats:
                                stats = hyperedge_norm_stats[eec_key]
                                if stats['method'] == 'zscore':
                                    gen_vals_plot = (gen_vals_plot - stats['mean']) / (stats['std'] + 1e-8)
                                else:  # minmax
                                    gen_vals_plot = (gen_vals_plot - stats['min']) / (stats['max'] - stats['min'] + 1e-8)
                            
                            # Create histogram from real scalars (already normalized)
                            all_vals = torch.cat([real_eec_values, gen_vals_plot])
                            min_val = all_vals.min().item()
                            max_val = all_vals.max().item()
                            
                            real_hist = torch.histc(real_eec_values, bins=num_bins, min=min_val, max=max_val)
                            gen_hist = torch.histc(gen_vals_plot, bins=num_bins, min=min_val, max=max_val)
                            
                            real_hist_norm_plot = real_hist / (real_hist.sum() + 1e-8)
                            gen_hist_norm_plot = gen_hist / (gen_hist.sum() + 1e-8)
                            
                            # Plot
                            bin_edges = np.linspace(min_val, max_val, num_bins + 1)
                            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                            width = (bin_edges[1] - bin_edges[0]) * 0.4
                            
                            ax.bar(bin_centers - width/2, real_hist_norm_plot.cpu().numpy(), width=width,
                                   alpha=0.6, label=f'Real (n={len(real_eec_values)})', color='blue')
                            ax.bar(bin_centers + width/2, gen_hist_norm_plot.cpu().numpy(), width=width,
                                   alpha=0.6, label=f'Gen (n={len(gen_vals_plot)}) NORM', color='red')
                            
                            ax.set_title(f'{n}pt_EEC (NORMALIZED)\nReal: {len(real_eec_values)} scalars → {num_bins} bins\nGen: {len(gen_vals_plot)} values → {num_bins} bins')
                            ax.set_xlabel(f'{n}pt_EEC value')
                            ax.set_ylabel('Probability Density')
                            ax.legend()
                            ax.grid(True, alpha=0.3)
                        
                        plt.tight_layout()
                        os.makedirs('plots/histogram_debug', exist_ok=True)
                        plt.savefig('plots/histogram_debug/hyperedge_features.png', dpi=150, bbox_inches='tight')
                        plt.close()
                        print(f"📊 Saved hyperedge feature histograms to plots/histogram_debug/hyperedge_features.png")
            
            except Exception as e:
                print(f"Warning: Could not plot hyperedge histograms: {e}")
        
        # ═══════════════════════════════════════════════════════════════════════
        # STEP 6: RETURN SEPARATE LOSSES (weighting done in compute_loss)
        # ═══════════════════════════════════════════════════════════════════════
        
        # Convert to tensors if they are scalars, otherwise use as-is
        if isinstance(edge_loss, (int, float)):
            edge_loss = torch.tensor(edge_loss, device=device, dtype=pred_particles.dtype)
        if isinstance(hyperedge_loss, (int, float)):
            hyperedge_loss = torch.tensor(hyperedge_loss, device=device, dtype=pred_particles.dtype)
        
        return {
            'edge': edge_loss,
            'hyperedge': hyperedge_loss
        }
    
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