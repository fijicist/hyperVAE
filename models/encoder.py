import torch
import torch.nn as nn
import torch.nn.functional as F
from .lgat_layers import LGATrLayer, EdgeAwareTransformerConv


class BipartiteEncoder(nn.Module):
    """
    Encoder for bipartite hypergraph VAE.
    
    Architecture:
    1. Particle embedding (4D) -> L-GATr blocks (3 layers)
    2. Edge embedding (5D) -> Edge-aware transformer (2 layers)
    3. Hyperedge embedding (2D) -> L-GATr blocks (2 layers)
    4. Bipartite cross-attention -> Fusion MLP -> Latent (μ, σ²)
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Dimensions
        particle_hidden = config['particle_hidden']
        edge_hidden = config['edge_hidden']
        hyperedge_hidden = config['hyperedge_hidden']
        latent_dim = config['latent_dim']
        num_heads = config['attention_heads']
        dropout = config['encoder']['dropout']
        
        # 1. Particle encoder
        particle_features = config.get('particle_features', 3)  # Default to 3 (pt, eta, phi)
        self.particle_embed = nn.Sequential(
            nn.Linear(particle_features, particle_hidden),  # pt, eta, phi
            nn.LayerNorm(particle_hidden),
            nn.GELU()
        )
        
        self.particle_lgat = nn.ModuleList([
            LGATrLayer(particle_hidden, num_heads, dropout)
            for _ in range(config['encoder']['particle_lgat_layers'])
        ])
        
        # 2. Edge encoder
        self.edge_embed = nn.Sequential(
            nn.Linear(5, edge_hidden),  # 5 edge features
            nn.LayerNorm(edge_hidden),
            nn.GELU()
        )
        
        self.edge_transformer = nn.ModuleList([
            EdgeAwareTransformerConv(edge_hidden, edge_hidden, edge_dim=5, num_heads=num_heads, dropout=dropout)
            for _ in range(config['encoder']['edge_transformer_layers'])
        ])
        
        # 3. Hyperedge encoder (memory-efficient - no self-attention for large hyperedge sets)
        self.hyperedge_embed = nn.Sequential(
            nn.Linear(2, hyperedge_hidden),  # 3pt_eec, 4pt_eec
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
            for _ in range(config['encoder']['hyperedge_lgat_layers'])
        ])
        
        # 4. Bipartite cross-attention
        self.cross_attention = BipartiteCrossAttention(
            particle_hidden, hyperedge_hidden, num_heads, dropout
        )
        
        # 5. Fusion and latent projection
        fusion_dim = particle_hidden + hyperedge_hidden + edge_hidden
        self.fusion_mlp = nn.Sequential(
            nn.Linear(fusion_dim, latent_dim * 2),
            nn.LayerNorm(latent_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim * 2, latent_dim * 2)
        )
        
        # Latent parameters
        self.mu_proj = nn.Linear(latent_dim * 2, latent_dim)
        self.logvar_proj = nn.Linear(latent_dim * 2, latent_dim)
    
    def forward(self, batch):
        """
        Args:
            batch: PyG Batch object with:
                - particle_x: [N_particles, 4]
                - hyperedge_x: [N_hyperedges, 2]
                - edge_attr: [N_edges, 5]
                - edge_index: [2, N_edges]
                - batch: Batch assignment
        """
        # 1. Encode particles
        particle_x = self.particle_embed(batch.particle_x)
        for layer in self.particle_lgat:
            particle_x = layer(particle_x)
        
        # Create particle batch index
        particle_batch = self._create_particle_batch(batch)
        
        # Pool particles per graph
        particle_pooled = self._global_mean_pool(particle_x, particle_batch)
        
        # 2. Encode edges (if present)
        if batch.edge_attr.size(0) > 0:
            # Embed edge features
            edge_x = self.edge_embed(batch.edge_attr)  # [E, edge_hidden]
            
            # Simple mean pooling of edges per graph
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
        hyperedge_x = self.hyperedge_embed(batch.hyperedge_x)
        for layer in self.hyperedge_mlp:
            # Residual connection
            hyperedge_x = hyperedge_x + layer(hyperedge_x)
        
        # Create batch assignment for hyperedges
        hyperedge_batch = self._create_hyperedge_batch(batch)
        hyperedge_pooled = self._global_mean_pool(hyperedge_x, hyperedge_batch)
        
        # 4. Cross-attention between particles and hyperedges
        cross_features = self.cross_attention(particle_pooled, hyperedge_pooled)
        
        # 5. Fusion
        fused = torch.cat([particle_pooled, hyperedge_pooled, edge_pooled], dim=-1)
        fused = self.fusion_mlp(fused)
        
        # 6. Latent parameters
        mu = self.mu_proj(fused)
        logvar = self.logvar_proj(fused)
        
        return mu, logvar
    
    def _create_particle_batch(self, batch):
        """Create batch index for particles only"""
        particle_batch = []
        for i in range(batch.num_graphs):
            n_particles = batch.n_particles[i].item()
            particle_batch.extend([i] * n_particles)
        return torch.tensor(particle_batch, device=batch.particle_x.device)
    
    def _global_mean_pool(self, x, batch):
        """Global mean pooling"""
        batch_size = batch.max().item() + 1
        out = torch.zeros(batch_size, x.size(1), device=x.device)
        out.scatter_add_(0, batch.unsqueeze(1).expand_as(x), x)
        count = torch.zeros(batch_size, device=x.device)
        count.scatter_add_(0, batch, torch.ones_like(batch, dtype=torch.float))
        return out / count.unsqueeze(1).clamp(min=1)
    
    def _create_hyperedge_batch(self, batch):
        """Create batch assignment for hyperedges"""
        hyperedge_batch = []
        for i in range(batch.num_graphs):
            n_hyperedges = batch.n_hyperedges[i].item()
            hyperedge_batch.extend([i] * n_hyperedges)
        return torch.tensor(hyperedge_batch, device=batch.particle_x.device)


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
