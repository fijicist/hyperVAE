"""
═══════════════════════════════════════════════════════════════════════════════
DECODER - VAE LATENT SPACE TO BIPARTITE HYPERGRAPH
═══════════════════════════════════════════════════════════════════════════════

Decodes VAE latent vectors into jets (bipartite hypergraph representation).

Architecture:
-------------
    Latent z [latent_dim] + Jet Type [3] → MLP Expander
                         ↓
    ┌────────────────────┴───────────────────┐
    │                                         │
    Topology Decoder                    Feature Decoders
    (Gumbel-Softmax)                    (Parallel Streams)
    │                                         │
    ├─ Particle mask [max_particles]         ├─ Particles → L-GATr → [N, 4]
    ├─ N_edges prediction                    ├─ Edges → MLP → [M, 5]
    └─ N_hyperedges prediction               └─ Hyperedges → MLP → [K, features]
                         ↓
                 Jet Features
            (pT, η, mass, N_particles)

Key Components:
---------------
1. Latent Expander: Projects latent z to hidden states for all components
2. Topology Decoder: Predicts which particles/edges/hyperedges exist (Gumbel-Softmax)
3. L-GATr Particle Decoder: Generates physically-valid 4-momenta
4. Edge/Hyperedge MLPs: Generate graph structure features
5. Jet Feature Head: Predicts jet-level observables (auxiliary supervision)

Generation Modes:
-----------------
- Training: Gumbel-Softmax with temperature annealing (differentiable sampling)
- Inference: Hard thresholding at 0.5 (deterministic for reproducibility)

Outputs:
--------
Dictionary with:
    'particle_features': [batch, max_particles, 4] - 4-momenta
    'particle_mask': [batch, max_particles] - Validity mask
    'edge_features': [batch, max_edges, 5] - Edge features
    'hyperedge_features': [batch, max_hyperedges, features] - Hyperedge features
    'topology': dict with n_particles, n_edges, n_hyperedges predictions
    'jet_features': [batch, 3] - (jet_pt, jet_eta, jet_mass)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .lgatr_wrapper import LGATrParticleDecoder
from torch_geometric.nn import GATv2Conv
import math


class BipartiteDecoder(nn.Module):
    """
    Decoder for bipartite hypergraph VAE.
    
    Architecture:
    1. Latent z + Jet Type -> MLP Expander
    2. Topology Decoder (Gumbel-Softmax sampling)
       - Particle existence mask
       - Edge/Hyperedge existence
    3. Feature Decoders (parallel)
       - Particle features: L-GATr -> (configurable features)
       - Edge features: MLP -> (configurable features)
       - Hyperedge features: MLP -> (configurable features)
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Dimensions
        latent_dim = config['latent_dim']
        edge_hidden = config['edge_hidden']
        hyperedge_hidden = config['hyperedge_hidden']
        max_particles = config['max_particles']
        dropout = config['decoder']['dropout']
        jet_types = 3
        
        # Get L-GATr output dimension (scalars)
        lgatr_config = config.get('decoder', {}).get('lgatr', {})
        lgatr_scalar_dim = lgatr_config.get('in_s_channels', 64)  # Scalars from latent
        
        # 1. MLP Expander with residual path (latent + jet type -> hidden states)
        # Now expands to edge_hidden + hyperedge_hidden + lgatr_scalar_dim (for L-GATr conditioning)
        output_dim = lgatr_scalar_dim + edge_hidden + hyperedge_hidden
        
        # Pre-projection for residual
        self.latent_pre_proj = nn.Linear(latent_dim + jet_types, output_dim)
        
        # Main expansion path
        self.latent_expander = nn.Sequential(
            nn.Linear(latent_dim + jet_types, latent_dim * 2),
            nn.LayerNorm(latent_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim * 2, output_dim)
        )
        
        # 2. Topology Decoder (uses lgatr_scalar_dim instead of particle_hidden)
        self.topology_decoder = TopologyDecoder(
            lgatr_scalar_dim, edge_hidden, hyperedge_hidden, 
            max_particles, config['decoder']['use_gumbel_softmax']
        )
        
        # 3. Particle Feature Decoder using official L-GATr
        self.particle_decoder = LGATrParticleDecoder(config)
        
        # 3.5. Per-particle variation module (CRITICAL FIX for particle diversity)
        # Creates unique features per particle using particle index + latent
        # Prevents all particles from having identical initial states
        self.particle_variation = nn.Sequential(
            nn.Linear(lgatr_scalar_dim + 1, lgatr_scalar_dim),  # +1 for particle index
            nn.GELU(),
            nn.Linear(lgatr_scalar_dim, lgatr_scalar_dim)
        )
        
        # Learnable scale parameter for per-particle variation
        # Controls how much variation is applied (model learns optimal value during training)
        # Initial value from config (default: 0.5, range typically 0.1-1.0)
        initial_scale = config['decoder'].get('particle_variation_scale', 0.5)
        self.particle_variation_scale = nn.Parameter(torch.tensor(initial_scale))
        
        # 3.6. Initial 4-momentum projection from per-particle scalars
        # Projects unique per-particle scalars to initial 4-momentum guess
        # This gives L-GATr meaningful, diverse starting points for refinement
        self.scalar_to_4mom_init = nn.Sequential(
            nn.Linear(lgatr_scalar_dim, lgatr_scalar_dim),
            nn.GELU(),
            nn.Linear(lgatr_scalar_dim, 4)
        )
        
        # Pre-compute and cache particle indices for efficiency (computed once, reused)
        # Normalized indices [0, 1/(max_p-1), 2/(max_p-1), ..., 1.0]
        # Leading particles (low index) vs soft particles (high index)
        self.register_buffer(
            'particle_indices_normalized',
            torch.arange(max_particles, dtype=torch.float32) / max(max_particles - 1, 1)
        )
        
        # 4. Edge Feature Decoder with residual connections
        num_heads = config.get('attention_heads', 4)  # Keep for backward compatibility
        edge_features = config.get('edge_features', 5)
        edge_layers = config['decoder'].get('edge_layers', 3)
        edge_variation_scale = config['decoder'].get('edge_variation_scale', 0.1)
        self.edge_decoder = EdgeFeatureDecoder(
            edge_hidden, num_heads, dropout,
            num_features=edge_features,
            num_layers=edge_layers,
            variation_scale=edge_variation_scale
        )
        
        # 5. Hyperedge Feature Decoder
        hyperedge_features = config.get('hyperedge_features', 2)
        hyperedge_layers = config['decoder'].get('hyperedge_layers', config['decoder'].get('mlp_layers', 2))
        hyperedge_variation_scale = config['decoder'].get('hyperedge_variation_scale', 0.5)
        self.hyperedge_decoder = HyperedgeFeatureDecoder(
            hyperedge_hidden, num_heads, dropout,
            num_features=hyperedge_features,
            num_layers=hyperedge_layers,
            variation_scale=hyperedge_variation_scale
        )
        
        # 6. Jet Feature Decoder (configurable depth and width)
        # Read from new symmetric config structure
        jet_layers_count = config['decoder'].get('jet_layers', 2)
        jet_hidden_dim = config.get('jet_hidden', latent_dim)  # Read from top-level hidden dimensions
        jet_dropout = config['decoder'].get('dropout', 0.2)  # Use general decoder dropout
        feature_proj_dropout = config['decoder'].get('feature_proj_dropout', 0.2)
        
        # Build jet feature decoder with configurable depth
        jet_layers = []
        input_dim = latent_dim + jet_types
        
        # Input projection
        jet_layers.extend([
            nn.Linear(input_dim, jet_hidden_dim),
            nn.LayerNorm(jet_hidden_dim),
            nn.GELU(),
            nn.Dropout(jet_dropout)
        ])
        
        # Hidden layers with residual connections
        for _ in range(jet_layers_count - 1):
            jet_layers.extend([
                nn.Linear(jet_hidden_dim, jet_hidden_dim),
                nn.LayerNorm(jet_hidden_dim),
                nn.GELU(),
                nn.Dropout(jet_dropout)
            ])
        
        self.jet_mlp = nn.Sequential(*jet_layers)
        
        # Deep feature projectors (similar to particle projectors)
        # One 3-layer MLP per jet feature: jet_pt, jet_eta, jet_mass
        num_jet_features = config.get('jet_features', 3)
        self.jet_feature_projectors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(jet_hidden_dim, jet_hidden_dim),
                nn.GELU(),
                nn.Dropout(feature_proj_dropout),
                nn.Linear(jet_hidden_dim, jet_hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(feature_proj_dropout),
                nn.Linear(jet_hidden_dim // 2, 1)
            )
            for _ in range(num_jet_features)
        ])
        
        self.max_particles = max_particles
    
    def forward(self, z, jet_type, temperature=1.0, generate_edges=True, generate_hyperedges=True):
        """
        Args:
            z: Latent vectors [batch_size, latent_dim]
            jet_type: Jet types [batch_size]
            temperature: Gumbel-softmax temperature
            generate_edges: Whether to generate edge features (training only)
            generate_hyperedges: Whether to generate hyperedge features (training only)
        
        Returns:
            Dictionary with generated features and topology
        """
        batch_size = z.size(0)
        device = z.device
        
        # One-hot encode jet type
        jet_type_onehot = F.one_hot(jet_type, num_classes=3).float()
        latent_input = torch.cat([z, jet_type_onehot], dim=-1)
        
        # 1. Expand latent with residual connection
        latent_proj = self.latent_pre_proj(latent_input)  # Direct projection
        latent_expanded = latent_proj + self.latent_expander(latent_input)  # Residual
        
        # Split into components: lgatr_scalar_dim + edge_hidden + hyperedge_hidden
        lgatr_config = self.config.get('decoder', {}).get('lgatr', {})
        lgatr_scalar_dim = lgatr_config.get('in_s_channels', 64)
        
        particle_scalars, edge_h, hyperedge_h = torch.split(
            latent_expanded, 
            [lgatr_scalar_dim, self.config['edge_hidden'], 
             self.config['hyperedge_hidden']], 
            dim=-1
        )
        
        # 2. Decode topology
        topology = self.topology_decoder(particle_scalars, edge_h, hyperedge_h, temperature)
        
        # 3. ALWAYS decode particle features using official L-GATr
        batch_size_actual = z.size(0)
        max_p = topology['particle_mask'].size(1)
        device = z.device
        
        # Generate per-particle scalars with positional encoding
        # Get cached indices (pre-computed in __init__) [max_p]
        indices = self.particle_indices_normalized[:max_p].view(1, max_p, 1)
        
        # Expand base scalars [batch, max_p, lgatr_scalar_dim]
        particle_scalars_base = particle_scalars.unsqueeze(1).expand(batch_size_actual, max_p, -1)
        
        # Single concat operation [batch, max_p, lgatr_scalar_dim+1]
        particle_scalars_with_idx = torch.cat([
            particle_scalars_base, 
            indices.expand(batch_size_actual, -1, -1)
        ], dim=-1)
        
        # Single reshape + forward + fused scale application
        # Flatten [batch*max_p, lgatr_scalar_dim+1]
        flat_input = particle_scalars_with_idx.view(-1, lgatr_scalar_dim + 1)
        
        # Generate variation [batch*max_p, lgatr_scalar_dim]
        particle_variation_flat = self.particle_variation(flat_input)
        
        # Fused operations - add base + scaled variation in one step
        # Flatten base for addition
        particle_scalars_base_flat = particle_scalars_base.reshape(-1, lgatr_scalar_dim)
        
        # Apply scale and combine (fused operation, more efficient)
        scale = torch.sigmoid(self.particle_variation_scale)
        particle_scalars_expanded = (
            particle_scalars_base_flat + particle_variation_flat * scale
        ).view(batch_size_actual, max_p, -1)
        
        # Generate per-particle initial 4-momenta [batch, max_p, 4]
        # Each particle gets unique initial guess based on its position and latent
        # Reuse flattened tensor, single reshape at end
        initial_fourmomenta = self.scalar_to_4mom_init(
            particle_scalars_expanded.view(-1, lgatr_scalar_dim)
        ).view(batch_size_actual, max_p, 4)
        
        # Add small noise for regularization (stochastic gradient)
        initial_fourmomenta = initial_fourmomenta + torch.randn_like(initial_fourmomenta) * 0.01
        
        # Pass through L-GATr decoder
        lgatr_output = self.particle_decoder(
            fourmomenta=initial_fourmomenta,
            scalars=particle_scalars_expanded,  # Condition on latent scalars
            mask=topology['particle_mask']
        )
        
        # Extract 4-vectors [E, px, py, pz]
        particle_features = lgatr_output['fourmomenta']  # [batch_size, max_particles, 4]
        
        # 4. Conditionally decode edge features (auxiliary - training only)
        if generate_edges:
            total_edges = sum(ei.size(1) for ei in topology['edge_index'])
            if total_edges > 0:
                edge_features = self.edge_decoder(
                    edge_h, topology['edge_index'], topology['n_particles']
                )
            else:
                edge_features = torch.zeros(batch_size, 0, 5, device=device)
        else:
            edge_features = None
        
        # 5. Conditionally decode hyperedge features (auxiliary - training only)
        if generate_hyperedges and topology['n_hyperedges'].sum() > 0:
            hyperedge_features = self.hyperedge_decoder(
                hyperedge_h, topology['hyperedge_mask'], topology['n_hyperedges']
            )
        else:
            hyperedge_features = None
        
        # 6. Decode jet-level features (ALWAYS generate) with deep projectors
        # Process through MLP layers
        jet_h = self.jet_mlp(latent_input)  # [batch_size, jet_hidden_dim]
        
        # Generate each jet feature using deep projectors (similar to particles)
        jet_feature_list = []
        for projector in self.jet_feature_projectors:
            feature = projector(jet_h).squeeze(-1)  # [batch_size]
            jet_feature_list.append(feature)
        
        # Stack features [batch_size, num_jet_features (3)]
        jet_features = torch.stack(jet_feature_list, dim=-1)
        
        return {
            'particle_features': particle_features,
            'edge_features': edge_features,  # None during inference
            'hyperedge_features': hyperedge_features,  # None during inference
            'jet_features': jet_features,  # [batch_size, 3] - jet_pt, jet_eta, jet_mass
            'topology': topology
        }


class TopologyDecoder(nn.Module):
    """Decode graph topology with Gumbel-Softmax"""
    
    def __init__(self, particle_hidden, edge_hidden, hyperedge_hidden, 
                 max_particles, use_gumbel_softmax=True):
        super().__init__()
        self.max_particles = max_particles
        self.use_gumbel_softmax = use_gumbel_softmax
        
        # Particle existence predictor
        self.particle_exist_mlp = nn.Sequential(
            nn.Linear(particle_hidden, particle_hidden),
            nn.GELU(),
            nn.Linear(particle_hidden, max_particles)
        )
        
        # Edge existence predictor (bilinear scoring)
        self.edge_score_proj = nn.Linear(particle_hidden, particle_hidden)
        
        # Hyperedge existence predictor
        self.hyperedge_exist_mlp = nn.Sequential(
            nn.Linear(hyperedge_hidden, hyperedge_hidden),
            nn.GELU(),
            nn.Linear(hyperedge_hidden, max_particles // 3)
        )
    
    def forward(self, particle_h, edge_h, hyperedge_h, temperature=1.0):
        """Generate topology"""
        batch_size = particle_h.size(0)
        device = particle_h.device
        
        # 1. Particle existence
        particle_logits = self.particle_exist_mlp(particle_h)
        if self.use_gumbel_softmax and self.training:
            particle_mask = F.gumbel_softmax(
                torch.stack([particle_logits, -particle_logits], dim=-1),
                tau=temperature, hard=True
            )[..., 0]
        else:
            particle_mask = (torch.sigmoid(particle_logits) > 0.5).float()
        
        n_particles = particle_mask.sum(dim=-1).long()
        
        # 2. Edge existence (sample edges between existing particles)
        edge_index_list = []
        for i in range(batch_size):
            n_p = n_particles[i].item()
            if n_p > 1:
                # Fully connected for simplicity (can be made sparse)
                edges = torch.combinations(torch.arange(n_p, device=device), r=2).T
                edge_index_list.append(edges)
            else:
                edge_index_list.append(torch.zeros((2, 0), dtype=torch.long, device=device))
        
        # 3. Hyperedge existence
        hyperedge_logits = self.hyperedge_exist_mlp(hyperedge_h)
        if self.use_gumbel_softmax and self.training:
            hyperedge_mask = F.gumbel_softmax(
                torch.stack([hyperedge_logits, -hyperedge_logits], dim=-1),
                tau=temperature, hard=True
            )[..., 0]
        else:
            hyperedge_mask = (torch.sigmoid(hyperedge_logits) > 0.5).float()
        
        n_hyperedges = hyperedge_mask.sum(dim=-1).long()
        
        return {
            'particle_mask': particle_mask,
            'n_particles': n_particles,
            'edge_index': edge_index_list,
            'hyperedge_mask': hyperedge_mask,
            'n_hyperedges': n_hyperedges
        }


class ParticleFeatureDecoder(nn.Module):
    """Decode particle features with edge/hyperedge context via cross-attention"""
    
    def __init__(self, hidden_dim, num_heads, dropout, edge_hidden=None, hyperedge_hidden=None, 
                 feature_proj_dropout=0.2, num_features=3, num_layers=4,
                 rapidity_index=1, phi_index=2, y_range=4.0, phi_range=3.141592653589793):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_features = num_features
        self.num_layers = num_layers
        
        # Physical constraints / indices for generated particle features
        self.rapidity_index = int(rapidity_index)
        self.phi_index = int(phi_index)
        self.y_range = float(y_range)
        self.phi_range = float(phi_range)
        
        self.lgat_layers = nn.ModuleList([
            LGATrLayer(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        # Projection layers to match dimensions for cross-attention
        # edge_hidden (96) -> particle_hidden (128)
        # hyperedge_hidden (64) -> particle_hidden (128)
        # CRITICAL: Initialize with small weights to prevent gradient explosion
        if edge_hidden is not None and edge_hidden != hidden_dim:
            self.edge_proj = nn.Linear(edge_hidden, hidden_dim)
            nn.init.xavier_uniform_(self.edge_proj.weight, gain=0.01)  # Small init
            nn.init.zeros_(self.edge_proj.bias)
        else:
            self.edge_proj = None
        
        if hyperedge_hidden is not None and hyperedge_hidden != hidden_dim:
            self.hyperedge_proj = nn.Linear(hyperedge_hidden, hidden_dim)
            nn.init.xavier_uniform_(self.hyperedge_proj.weight, gain=0.01)  # Small init
            nn.init.zeros_(self.hyperedge_proj.bias)
        else:
            self.hyperedge_proj = None
        
        # Cross-attention layers to incorporate edge/hyperedge context
        self.edge_cross_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.hyperedge_cross_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        
        # Layer norms for cross-attention
        self.edge_norm = nn.LayerNorm(hidden_dim)
        self.hyperedge_norm = nn.LayerNorm(hidden_dim)
        
        # Scaling factor for cross-attention (FIXED, not learnable)
        # Prevents cross-attention from overwhelming particle features
        self.cross_attn_scale = 0.1  # Fixed at 10%, not learnable
        
        # Feature predictors - DEEPER and WIDER for better expressivity
        # Dynamically create projectors based on num_features
        # 3-layer MLPs with hidden_dim (not hidden//2) to break plateau
        self.feature_projectors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(feature_proj_dropout),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(feature_proj_dropout),
                nn.Linear(hidden_dim // 2, 1)
            )
            for _ in range(num_features)
        ])
    
    def forward(self, particle_h, particle_mask, n_particles, edge_context=None, hyperedge_context=None):
        """
        Args:
            particle_h: [batch_size, particle_hidden_dim]
            particle_mask: [batch_size, max_particles]
            n_particles: [batch_size]
            edge_context: [batch_size, edge_hidden_dim] - edge embeddings (optional)
            hyperedge_context: [batch_size, hyperedge_hidden_dim] - hyperedge embeddings (optional)
        """
        batch_size = particle_h.size(0)
        max_p = particle_mask.size(1)
        device = particle_h.device
        
        # Expand to max_particles
        particle_h_expanded = particle_h.unsqueeze(1).expand(batch_size, max_p, -1)
        
        # Apply L-GATr layers with residual connections per sample
        for layer in self.lgat_layers:
            residual = particle_h_expanded  # Store for residual connection
            particle_h_expanded = particle_h_expanded.reshape(-1, self.hidden_dim)
            particle_h_expanded = layer(particle_h_expanded)
            particle_h_expanded = particle_h_expanded.reshape(batch_size, max_p, -1)
            # Residual connection: helps preserve particle-specific information
            particle_h_expanded = particle_h_expanded + residual
        
        # Cross-attention with edge context (if provided during training)
        # CRITICAL: Scale down attention output to prevent overwhelming particle features
        if edge_context is not None:
            # Project edge context to particle dimension if needed
            if self.edge_proj is not None:
                edge_context = self.edge_proj(edge_context)
            
            edge_ctx = edge_context.unsqueeze(1)  # [B, 1, particle_hidden_dim]
            attn_out, _ = self.edge_cross_attn(
                particle_h_expanded, edge_ctx, edge_ctx
            )
            # Scale down the attention contribution (starts at 0.1, learned during training)
            particle_h_expanded = self.edge_norm(particle_h_expanded + self.cross_attn_scale * attn_out)
        
        # Cross-attention with hyperedge context (if provided during training)
        if hyperedge_context is not None:
            # Project hyperedge context to particle dimension if needed
            if self.hyperedge_proj is not None:
                hyperedge_context = self.hyperedge_proj(hyperedge_context)
            
            he_ctx = hyperedge_context.unsqueeze(1)  # [B, 1, particle_hidden_dim]
            attn_out, _ = self.hyperedge_cross_attn(
                particle_h_expanded, he_ctx, he_ctx
            )
            # Scale down the attention contribution (same learnable scale)
            particle_h_expanded = self.hyperedge_norm(particle_h_expanded + self.cross_attn_scale * attn_out)
        
        # Generate features dynamically based on num_features
        # CRITICAL: Output in NORMALIZED space to match input data!
        features = []
        for projector in self.feature_projectors:
            feature = projector(particle_h_expanded).squeeze(-1)
            feature = feature * particle_mask  # Apply mask
            features.append(feature)
        
        # Stack features [batch_size, max_particles, num_features]
        particle_features = torch.stack(features, dim=-1)
        
        # Apply physical constraints to rapidity (y) and phi channels
        # Use tanh scaling to keep outputs differentiable and strictly within bounds
        if 0 <= self.rapidity_index < self.num_features:
            particle_features[..., self.rapidity_index] = torch.tanh(
                particle_features[..., self.rapidity_index]
            ) * self.y_range

        if 0 <= self.phi_index < self.num_features:
            particle_features[..., self.phi_index] = torch.tanh(
                particle_features[..., self.phi_index]
            ) * self.phi_range

        return particle_features


class EdgeFeatureDecoder(nn.Module):
    """Decode edge features with residual connections"""
    
    def __init__(self, hidden_dim, num_heads, dropout, num_features=5, num_layers=3, variation_scale=0.1):
        super().__init__()
        
        self.num_features = num_features
        self.hidden_dim = hidden_dim
        
        # MLP layers with residual connections (matching encoder)
        self.edge_mlp = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.LayerNorm(hidden_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.Dropout(dropout)
            )
            for _ in range(num_layers)
        ])
        
        # Final projection to edge features
        self.feature_proj = nn.Linear(hidden_dim, num_features)
        
        # Learnable noise scale for edge variation (configurable)
        # Controls diversity in edge features (default: 0.1 for subtle variation)
        self.variation_scale = nn.Parameter(torch.tensor(variation_scale))
    
    def forward(self, edge_h, edge_index_list, n_particles):
        """Generate edge features with residual connections"""
        batch_size = edge_h.size(0)
        device = edge_h.device
        
        # Apply MLP layers with residual connections
        for layer in self.edge_mlp:
            edge_h = edge_h + layer(edge_h)  # Residual connection
        
        # Project to features per batch element [batch_size, num_features]
        edge_features_base = self.feature_proj(edge_h)
        
        # Vectorized edge generation (no Python loops)
        # Count edges per jet
        edge_counts = torch.tensor([ei.size(1) for ei in edge_index_list], 
                                   dtype=torch.long, device=device)
        max_edges = edge_counts.max().item()
        
        if max_edges == 0:
            return torch.zeros(batch_size, 0, self.num_features, device=device)
        
        # Pre-allocate output tensor (single allocation, more efficient)
        output = torch.zeros(batch_size, max_edges, self.num_features, device=device)
        
        # Vectorized: Expand features and add noise in one go
        # Create mask for valid edges
        edge_mask = torch.arange(max_edges, device=device).unsqueeze(0) < edge_counts.unsqueeze(1)
        
        # Broadcast base features to all positions
        output = edge_features_base.unsqueeze(1).expand(-1, max_edges, -1).clone()
        
        # Add noise only to valid edges (vectorized, with learnable scale)
        noise = torch.randn_like(output) * torch.abs(self.variation_scale)  # abs() ensures positive
        output = torch.where(edge_mask.unsqueeze(-1), output + noise, output)
        
        return output  # [batch_size, max_edges, num_features]


class HyperedgeFeatureDecoder(nn.Module):
    """Decode hyperedge features"""
    
    def __init__(self, hidden_dim, num_heads, dropout, num_features=2, num_layers=2, variation_scale=0.5):
        super().__init__()
        
        self.num_features = num_features
        self.hidden_dim = hidden_dim
        
        # Use MLP instead of attention to save memory (for large hyperedge sets)
        self.mlp_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.LayerNorm(hidden_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.Dropout(dropout)
            )
            for _ in range(num_layers)
        ])
        
        self.feature_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, num_features)  # Configurable number of hyperedge features
        )
        
        # Per-hyperedge variation module (creates unique features per hyperedge)
        self.hyperedge_variation = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim),  # +1 for hyperedge index
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Learnable scale for variation (configurable, similar to particle decoder)
        # Controls how much per-hyperedge variation is applied (default: 0.5)
        self.variation_scale = nn.Parameter(torch.tensor(variation_scale))
    
    def forward(self, hyperedge_h, hyperedge_mask, n_hyperedges):
        """Generate hyperedge features"""
        batch_size = hyperedge_h.size(0)
        max_he = hyperedge_mask.size(1)
        device = hyperedge_h.device
        
        # Apply MLP with residual to base representation
        hyperedge_h_base = hyperedge_h
        for layer in self.mlp_layers:
            hyperedge_h_base = hyperedge_h_base + layer(hyperedge_h_base)
        
        # Pre-compute and cache indices (computed once per forward)
        # Normalized indices [0, 1/(max_he-1), ..., 1.0]
        hyperedge_indices_normalized = (
            torch.arange(max_he, device=device, dtype=torch.float32) / max(max_he - 1, 1)
        ).view(1, max_he, 1)  # [1, max_he, 1] for broadcasting
        
        # Expand base features to all hyperedges [batch, max_he, hidden]
        hyperedge_h_expanded = hyperedge_h_base.unsqueeze(1).expand(batch_size, max_he, -1)
        
        # Single concatenate + reshape + forward pass
        # Concatenate with normalized index [batch, max_he, hidden+1]
        hyperedge_input = torch.cat([
            hyperedge_h_expanded, 
            hyperedge_indices_normalized.expand(batch_size, -1, -1)
        ], dim=-1)
        
        # Single reshape and forward pass (more efficient than multiple reshapes)
        hyperedge_variation = self.hyperedge_variation(
            hyperedge_input.view(-1, self.hidden_dim + 1)
        ).view(batch_size, max_he, -1)
        
        # Combine base + scaled variation (learnable scale for flexibility)
        scale = torch.sigmoid(self.variation_scale)  # Keep in [0, 1]
        hyperedge_h_final = hyperedge_h_expanded + hyperedge_variation * scale
        
        # Generate features and apply mask in one operation
        hyperedge_features = self.feature_proj(hyperedge_h_final) * hyperedge_mask.unsqueeze(-1)
        
        return hyperedge_features


if __name__ == "__main__":
    # Test decoder
    config = {
        'latent_dim': 128,
        'particle_hidden': 64,
        'edge_hidden': 48,
        'hyperedge_hidden': 32,
        'max_particles': 150,
        'attention_heads': 4,
        'decoder': {
            'mlp_layers': 3,
            'dropout': 0.1,
            'use_gumbel_softmax': True,
            'gumbel_temperature': 1.0
        }
    }
    
    decoder = BipartiteDecoder(config)
    
    z = torch.randn(4, 128)
    jet_type = torch.tensor([0, 1, 2, 0])
    
    output = decoder(z, jet_type)
    
    print(f"Particle features: {output['particle_features'].shape}")
    print(f"Edge features: {output['edge_features'].shape}")
    print(f"Hyperedge features: {output['hyperedge_features'].shape}")
