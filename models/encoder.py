"""
═══════════════════════════════════════════════════════════════════════════════
ENCODER - BIPARTITE HYPERGRAPH TO VAE LATENT SPACE
═══════════════════════════════════════════════════════════════════════════════

Encodes jets (represented as bipartite hypergraphs) into VAE latent space.

Architecture:
-------------
    Particles [N, 4] → L-GATr → Scalars [N, hidden_s]
    Edges [M, 5] → MLP → [M, edge_hidden]
    Hyperedges [K, features] → MLP → [K, hyperedge_hidden]
                    ↓
              Cross-Attention (particles ↔ hyperedges)
                    ↓
            Global Pooling (mean over particles/edges/hyperedges)
                    ↓
            Fusion MLP → μ, log(σ²) (latent dim)

Key Components:
---------------
1. L-GATr Particle Encoder: Lorentz-equivariant processing of 4-momenta
2. Edge/Hyperedge MLPs: Process graph structure information
3. Cross-Attention: Fuses particle-level and hyperedge-level information
4. Global Pooling: Aggregates variable-length inputs to fixed-size latent
5. Latent Projection: Maps to VAE latent space (μ, logvar for reparameterization)

Outputs:
--------
mu: [batch_size, latent_dim] - Mean of latent distribution
logvar: [batch_size, latent_dim] - Log-variance of latent distribution

These are used for VAE reparameterization: z = μ + σ * ε, where ε ~ N(0, I)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .lgatr_wrapper import LGATrParticleEncoder


class BipartiteEncoder(nn.Module):
    """
    Encoder for bipartite hypergraph VAE.
    
    Architecture:
    1. Particle embedding (configurable dims) -> L-GATr blocks
    2. Edge embedding (configurable dims) -> Edge-aware transformer
    3. Hyperedge embedding (configurable dims) -> MLP blocks
    4. Bipartite cross-attention -> Fusion MLP -> Latent (μ, σ²)
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Dimensions
        edge_hidden = config['edge_hidden']
        hyperedge_hidden = config['hyperedge_hidden']
        latent_dim = config['latent_dim']
        dropout = config['encoder']['dropout']
        
        # 1. Particle encoder using official L-GATr library
        # L-GATr handles particle embedding internally, outputs scalars for latent projection
        self.particle_encoder = LGATrParticleEncoder(config)
        particle_output_dim = self.particle_encoder.get_output_dim()  # Get scalar output dimension
        
        # If L-GATr outputs scalars, use them; otherwise will need to extract from 4-vectors
        if particle_output_dim == 0:
            # L-GATr outputs only 4-vectors, need to convert to scalars for latent
            # Use a simple projection from 4D to hidden dim
            particle_features = config.get('particle_features', 4)
            particle_output_dim = config.get('particle_hidden', 128)  # Fallback for pooling
            self.particle_to_scalar = nn.Sequential(
                nn.Linear(particle_features, particle_output_dim),
                nn.LayerNorm(particle_output_dim),
                nn.GELU()
            )
        else:
            self.particle_to_scalar = None
        
        # 2. Edge encoder with residual connections
        edge_features = config.get('edge_features', 5)  # Configurable (default: 5)
        self.edge_embed = nn.Sequential(
            nn.Linear(edge_features, edge_hidden),
            nn.LayerNorm(edge_hidden),
            nn.GELU()
        )
        
        # Edge MLP layers with residual connections
        self.edge_mlp = nn.ModuleList([
            nn.Sequential(
                nn.Linear(edge_hidden, edge_hidden * 2),
                nn.LayerNorm(edge_hidden * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(edge_hidden * 2, edge_hidden),
                nn.Dropout(dropout)
            )
            for _ in range(config['encoder'].get('edge_layers', config['encoder'].get('edge_transformer_layers', 3)))
        ])
        
        # 3. Hyperedge encoder (memory-efficient - no self-attention for large hyperedge sets)
        hyperedge_features = config.get('hyperedge_features', 2)  # Configurable (default: 2)
        self.hyperedge_embed = nn.Sequential(
            nn.Linear(hyperedge_features, hyperedge_hidden),
            nn.LayerNorm(hyperedge_hidden),
            nn.GELU()
        )
        
        # Use simple MLP instead of attention for hyperedges (memory efficient)
        self.hyperedge_mlp = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hyperedge_hidden, hyperedge_hidden * 2),
                nn.LayerNorm(hyperedge_hidden * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hyperedge_hidden * 2, hyperedge_hidden),
                nn.Dropout(dropout)
            )
            for _ in range(config['encoder'].get('hyperedge_layers', config['encoder'].get('hyperedge_lgat_layers', 2)))
        ])
        
        # 4. Jet-level feature encoder with residual connections
        jet_features = config.get('jet_features', 3)  # Configurable (default: 3)
        jet_hidden = config.get('jet_hidden', latent_dim)  # Use jet_hidden from config
        self.jet_embed = nn.Sequential(
            nn.Linear(jet_features, jet_hidden),
            nn.LayerNorm(jet_hidden),
            nn.GELU()
        )
        
        # Jet MLP layers with residual connections
        self.jet_mlp = nn.ModuleList([
            nn.Sequential(
                nn.Linear(jet_hidden, jet_hidden * 2),
                nn.LayerNorm(jet_hidden * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(jet_hidden * 2, jet_hidden),
                nn.Dropout(dropout)
            )
            for _ in range(config['encoder'].get('jet_layers', 2))
        ])
        
        # 5. Bipartite cross-attention (use particle_output_dim instead of particle_hidden)
        num_heads = config.get('attention_heads', 4)  # Keep for backward compatibility
        self.cross_attention = BipartiteCrossAttention(
            particle_output_dim, hyperedge_hidden, num_heads, dropout
        )
        
        # 6. Fusion and latent projection with pre-projection for residual
        fusion_dim = particle_output_dim + hyperedge_hidden + edge_hidden + jet_hidden
        
        # Project fused features to latent_dim*2 for residual connection
        self.fusion_pre_proj = nn.Linear(fusion_dim, latent_dim * 2)
        
        # Main fusion MLP
        self.fusion_mlp = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim * 2),
            nn.LayerNorm(latent_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim * 2, latent_dim * 2)
        )
        
        # Latent parameters with layer norm for stability
        self.latent_norm = nn.LayerNorm(latent_dim * 2)
        self.mu_proj = nn.Linear(latent_dim * 2, latent_dim)
        self.logvar_proj = nn.Linear(latent_dim * 2, latent_dim)
    
    def forward(self, batch):
        """
        Args:
            batch: PyG Batch object with:
                - particle_x: [N_particles, 4] - 4-vectors [E, px, py, pz]
                - hyperedge_x: [N_hyperedges, 1 or 2]
                - edge_attr: [N_edges, 5]
                - edge_index: [2, N_edges]
                - batch: Batch assignment
        """
        # 1. Encode particles with official L-GATr
        # Need to convert PyG flat format [N_particles, 4] to batched [batch_size, max_particles, 4]
        
        # Create particle batch index first to determine structure
        particle_batch = self._create_particle_batch(batch)
        batch_size = batch.num_graphs
        
        # Find max particles in this batch for padding
        n_particles_list = batch.n_particles.tolist()
        max_particles = max(n_particles_list)
        
        # Reshape to batched format with padding
        device = batch.x.device
        batched_fourmomenta = torch.zeros(batch_size, max_particles, 4, device=device)
        batched_mask = torch.zeros(batch_size, max_particles, dtype=torch.bool, device=device)
        
        particle_idx = 0
        for i, n_p in enumerate(n_particles_list):
            batched_fourmomenta[i, :n_p] = batch.x[particle_idx:particle_idx + n_p]
            batched_mask[i, :n_p] = True
            particle_idx += n_p
        
        # Pass through L-GATr encoder
        lgatr_output = self.particle_encoder(batched_fourmomenta, mask=batched_mask)
        
        # Extract features - need to flatten back to [N_particles, features] for pooling
        if lgatr_output['scalars'] is not None:
            # L-GATr outputs scalar features [batch_size, max_particles, out_s_channels]
            particle_x_batched = lgatr_output['scalars']
        else:
            # L-GATr only outputs 4-vectors, project to scalars
            particle_x_batched = self.particle_to_scalar(lgatr_output['fourmomenta'])
        
        # Flatten back to [N_particles, features] for subsequent processing
        particle_x = []
        for i, n_p in enumerate(n_particles_list):
            particle_x.append(particle_x_batched[i, :n_p])
        particle_x = torch.cat(particle_x, dim=0)  # [N_particles, features]
        
        # Pool particles per graph
        particle_pooled = self._global_mean_pool(particle_x, particle_batch)
        
        # 2. Encode edges with residual connections (if present)
        if batch.edge_attr.size(0) > 0:
            # Embed edge features
            edge_x = self.edge_embed(batch.edge_attr)  # [E, edge_hidden]
            
            # Apply MLP layers with residual connections
            for layer in self.edge_mlp:
                edge_x = edge_x + layer(edge_x)  # Residual connection
            
            # Mean pooling of edges per graph
            edge_pooled = torch.zeros(batch.num_graphs, self.config['edge_hidden'], device=edge_x.device)
            cumulative_particles = 0
            cumulative_edges = 0
            
            for i in range(batch.num_graphs):
                n_particles = batch.n_particles[i].item()
                # Count edges in this graph
                mask = (batch.edge_index[0] >= cumulative_particles) & (batch.edge_index[0] < cumulative_particles + n_particles)
                n_edges = mask.sum().item()
                
                if n_edges > 0:
                    graph_edges = edge_x[cumulative_edges:cumulative_edges + n_edges]
                    edge_pooled[i] = graph_edges.mean(dim=0)
                    cumulative_edges += n_edges
                
                cumulative_particles += n_particles
        else:
            edge_pooled = torch.zeros(particle_pooled.size(0), self.config['edge_hidden'], 
                                     device=particle_x.device)
        
        # 3. Encode hyperedges (no self-attention to save memory)
        hyperedge_x = self.hyperedge_embed(batch.hyperedge_attr)
        for layer in self.hyperedge_mlp:
            # Residual connection
            hyperedge_x = hyperedge_x + layer(hyperedge_x)
        
        # Create batch assignment for hyperedges
        hyperedge_batch = self._create_hyperedge_batch(batch)
        hyperedge_pooled = self._global_mean_pool(hyperedge_x, hyperedge_batch)
        
        # 4. Encode jet-level features with residual connections
        # Extract jet features from y tensor: [jet_pt, jet_eta, jet_mass]
        jet_pt_index = self.config.get('training', {}).get('loss_config', {}).get('jet_pt_index', 1)
        jet_eta_index = self.config.get('training', {}).get('loss_config', {}).get('jet_eta_index', 2)
        jet_mass_index = self.config.get('training', {}).get('loss_config', {}).get('jet_mass_index', 3)
        
        # PyG concatenates y tensors during batching: [4, 4, 4, ...] -> [batch_size*4]
        # Reshape to [batch_size, 4] for proper indexing
        y = batch.y
        if y.dim() == 1:
            batch_size = batch.num_graphs
            y = y.view(batch_size, -1)  # [batch_size, 4]
        
        jet_features = torch.stack([
            y[:, jet_pt_index],
            y[:, jet_eta_index],
            y[:, jet_mass_index]
        ], dim=1)  # [batch_size, 3]
        
        jet_x = self.jet_embed(jet_features)
        for layer in self.jet_mlp:
            # Residual connection
            jet_x = jet_x + layer(jet_x)
        
        # 5. Cross-attention between particles and hyperedges
        cross_features = self.cross_attention(particle_pooled, hyperedge_pooled)
        
        # 6. Fusion with residual connection (now includes jet features)
        fused = torch.cat([particle_pooled, hyperedge_pooled, edge_pooled, jet_x], dim=-1)
        fused_proj = self.fusion_pre_proj(fused)  # Project to latent_dim*2
        fused = fused_proj + self.fusion_mlp(fused_proj)  # Residual: preserve direct mapping
        
        # 6. Latent parameters with normalization for stability
        fused = self.latent_norm(fused)  # Normalize before projection to prevent explosion
        mu = self.mu_proj(fused)
        logvar = self.logvar_proj(fused)
        
        # CRITICAL: Clamp to prevent numerical overflow in KL divergence
        # exp(-10) ≈ 0.000045, exp(10) ≈ 22026 (safe range)
        mu = torch.clamp(mu, min=-10.0, max=10.0)
        logvar = torch.clamp(logvar, min=-10.0, max=10.0)
        
        # Additional safety: replace NaN with zeros (emergency fallback)
        if torch.isnan(mu).any():
            mu = torch.where(torch.isnan(mu), torch.zeros_like(mu), mu)
        if torch.isnan(logvar).any():
            logvar = torch.where(torch.isnan(logvar), torch.zeros_like(logvar), logvar)
        
        return mu, logvar
    
    def _create_particle_batch(self, batch):
        """Create batch index for particles only"""
        particle_batch = []
        for i in range(batch.num_graphs):
            n_particles = batch.n_particles[i].item()
            particle_batch.extend([i] * n_particles)
        return torch.tensor(particle_batch, device=batch.x.device)
    
    def _global_mean_pool(self, x, batch):
        """Global mean pooling"""
        batch_size = batch.max().item() + 1
        out = torch.zeros(batch_size, x.size(1), device=x.device, dtype=x.dtype)
        out.scatter_add_(0, batch.unsqueeze(1).expand_as(x), x)
        count = torch.zeros(batch_size, device=x.device, dtype=x.dtype)
        count.scatter_add_(0, batch, torch.ones_like(batch, dtype=x.dtype))
        return out / count.unsqueeze(1).clamp(min=1)
    
    def _create_hyperedge_batch(self, batch):
        """Create batch assignment for hyperedges"""
        hyperedge_batch = []
        for i in range(batch.num_graphs):
            n_hyperedges = batch.n_hyperedges[i].item()
            hyperedge_batch.extend([i] * n_hyperedges)
        return torch.tensor(hyperedge_batch, device=batch.x.device)


class BipartiteCrossAttention(nn.Module):
    """Cross-attention between particle and hyperedge representations"""
    
    def __init__(self, particle_dim, hyperedge_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = particle_dim // num_heads
        
        self.q_proj = nn.Linear(particle_dim, particle_dim)
        self.k_proj = nn.Linear(hyperedge_dim, particle_dim)
        self.v_proj = nn.Linear(hyperedge_dim, particle_dim)
        self.o_proj = nn.Linear(particle_dim, particle_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
    
    def forward(self, particle_x, hyperedge_x):
        """
        Args:
            particle_x: [batch_size, particle_dim]
            hyperedge_x: [batch_size, hyperedge_dim]
        """
        batch_size = particle_x.size(0)
        
        q = self.q_proj(particle_x).view(batch_size, self.num_heads, self.head_dim)
        k = self.k_proj(hyperedge_x).view(batch_size, self.num_heads, self.head_dim)
        v = self.v_proj(hyperedge_x).view(batch_size, self.num_heads, self.head_dim)
        
        attn_scores = torch.einsum('bhd,bhd->bh', q, k) * self.scale
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        out = attn_weights.unsqueeze(-1) * v
        out = out.view(batch_size, -1)
        out = self.o_proj(out)
        
        return out


if __name__ == "__main__":
    from data.bipartite_dataset import BipartiteJetDataset, collate_bipartite_batch
    from torch.utils.data import DataLoader
    
    # Test encoder
    config = {
        'particle_hidden': 64,
        'edge_hidden': 48,
        'hyperedge_hidden': 32,
        'latent_dim': 128,
        'attention_heads': 4,
        'encoder': {
            'particle_lgat_layers': 3,
            'edge_transformer_layers': 2,
            'hyperedge_lgat_layers': 2,
            'dropout': 0.1
        }
    }
    
    dataset = BipartiteJetDataset(generate_dummy=True)
    loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_bipartite_batch)
    batch = next(iter(loader))
    
    encoder = BipartiteEncoder(config)
    mu, logvar = encoder(batch)
    
    print(f"Batch size: {batch.num_graphs}")
    print(f"Mu shape: {mu.shape}")
    print(f"Logvar shape: {logvar.shape}")
