import torch
import torch.nn as nn
import torch.nn.functional as F
from .lgat_layers import LGATrLayer
from torch_geometric.nn import GATv2Conv


class BipartiteDecoder(nn.Module):
    """
    Decoder for bipartite hypergraph VAE.
    
    Architecture:
    1. Latent z + Jet Type -> MLP Expander
    2. Topology Decoder (Gumbel-Softmax sampling)
       - Particle existence mask
       - Edge/Hyperedge existence
    3. Feature Decoders (parallel)
       - Particle features: L-GATr -> (pt, eta, phi, m)
       - Edge features: GATv2Conv -> (ln_delta, ln_kt, ln_z, ln_m2, feat5)
       - Hyperedge features: L-GATr -> (3pt_eec, 4pt_eec)
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Dimensions
        latent_dim = config['latent_dim']
        particle_hidden = config['particle_hidden']
        edge_hidden = config['edge_hidden']
        hyperedge_hidden = config['hyperedge_hidden']
        max_particles = config['max_particles']
        num_heads = config['attention_heads']
        dropout = config['decoder']['dropout']
        jet_types = 3
        
        # 1. MLP Expander (latent + jet type -> hidden states)
        self.latent_expander = nn.Sequential(
            nn.Linear(latent_dim + jet_types, latent_dim * 2),
            nn.LayerNorm(latent_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim * 2, particle_hidden + edge_hidden + hyperedge_hidden)
        )
        
        # 2. Topology Decoder
        self.topology_decoder = TopologyDecoder(
            particle_hidden, edge_hidden, hyperedge_hidden, 
            max_particles, config['decoder']['use_gumbel_softmax']
        )
        
        # 3. Particle Feature Decoder
        self.particle_decoder = ParticleFeatureDecoder(
            particle_hidden, num_heads, dropout
        )
        
        # 4. Edge Feature Decoder
        self.edge_decoder = EdgeFeatureDecoder(
            edge_hidden, num_heads, dropout
        )
        
        # 5. Hyperedge Feature Decoder
        self.hyperedge_decoder = HyperedgeFeatureDecoder(
            hyperedge_hidden, num_heads, dropout
        )
        
        self.max_particles = max_particles
    
    def forward(self, z, jet_type, temperature=1.0):
        """
        Args:
            z: Latent vectors [batch_size, latent_dim]
            jet_type: Jet types [batch_size]
            temperature: Gumbel-softmax temperature
        
        Returns:
            Dictionary with generated features and topology
        """
        batch_size = z.size(0)
        device = z.device
        
        # One-hot encode jet type
        jet_type_onehot = F.one_hot(jet_type, num_classes=3).float()
        
        # 1. Expand latent
        latent_expanded = self.latent_expander(torch.cat([z, jet_type_onehot], dim=-1))
        
        particle_h, edge_h, hyperedge_h = torch.split(
            latent_expanded, 
            [self.config['particle_hidden'], self.config['edge_hidden'], 
             self.config['hyperedge_hidden']], 
            dim=-1
        )
        
        # 2. Decode topology
        topology = self.topology_decoder(particle_h, edge_h, hyperedge_h, temperature)
        
        # 3. Decode particle features
        particle_features = self.particle_decoder(
            particle_h, topology['particle_mask'], topology['n_particles']
        )
        
        # 4. Decode edge features (optional)
        # Note: edge_index is a list of tensors (one per graph)
        total_edges = sum(ei.size(1) for ei in topology['edge_index'])
        if total_edges > 0:
            edge_features = self.edge_decoder(
                edge_h, topology['edge_index'], topology['n_particles']
            )
        else:
            edge_features = torch.zeros(batch_size, 0, 5, device=device)
        
        # 5. Decode hyperedge features (optional)
        if topology['n_hyperedges'].sum() > 0:
            hyperedge_features = self.hyperedge_decoder(
                hyperedge_h, topology['hyperedge_mask'], topology['n_hyperedges']
            )
        else:
            hyperedge_features = torch.zeros(batch_size, 0, 2, device=device)
        
        return {
            'particle_features': particle_features,
            'edge_features': edge_features,
            'hyperedge_features': hyperedge_features,
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
    """Decode particle features with physics constraints"""
    
    def __init__(self, hidden_dim, num_heads, dropout):
        super().__init__()
        
        self.lgat_layers = nn.ModuleList([
            LGATrLayer(hidden_dim, num_heads, dropout)
            for _ in range(2)
        ])
        
        # Feature predictors with physics constraints (pt, eta, phi only - no mass)
        self.pt_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softplus()  # pt > 0
        )
        
        self.eta_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Tanh()  # eta ∈ [-2.5, 2.5]
        )
        
        self.phi_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Tanh()  # phi ∈ [-π, π]
        )
    
    def forward(self, particle_h, particle_mask, n_particles):
        """
        Args:
            particle_h: [batch_size, hidden_dim]
            particle_mask: [batch_size, max_particles]
            n_particles: [batch_size]
        """
        batch_size = particle_h.size(0)
        max_p = particle_mask.size(1)
        device = particle_h.device
        
        # Expand to max_particles
        particle_h_expanded = particle_h.unsqueeze(1).expand(batch_size, max_p, -1)
        
        # Apply L-GATr layers per sample
        for layer in self.lgat_layers:
            particle_h_expanded = particle_h_expanded.reshape(-1, particle_h.size(-1))
            particle_h_expanded = layer(particle_h_expanded)
            particle_h_expanded = particle_h_expanded.reshape(batch_size, max_p, -1)
        
        # Generate features (pt, eta, phi only - no mass)
        pt = self.pt_proj(particle_h_expanded).squeeze(-1) * 100  # Scale to typical pt range
        eta = self.eta_proj(particle_h_expanded).squeeze(-1) * 2.5
        phi = self.phi_proj(particle_h_expanded).squeeze(-1) * 3.14159
        
        # Apply mask
        pt = pt * particle_mask
        eta = eta * particle_mask
        phi = phi * particle_mask
        
        # Stack features (3D: pt, eta, phi)
        particle_features = torch.stack([pt, eta, phi], dim=-1)
        
        return particle_features


class EdgeFeatureDecoder(nn.Module):
    """Decode edge features"""
    
    def __init__(self, hidden_dim, num_heads, dropout):
        super().__init__()
        
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 5)  # 5 edge features
        )
    
    def forward(self, edge_h, edge_index_list, n_particles):
        """Generate edge features"""
        batch_size = edge_h.size(0)
        
        # Expand and generate features
        edge_features = self.edge_mlp(edge_h)
        
        # For now, return a placeholder (proper implementation needs edge indexing)
        return edge_features.unsqueeze(1).expand(batch_size, 10, 5)


class HyperedgeFeatureDecoder(nn.Module):
    """Decode hyperedge features (memory-efficient)"""
    
    def __init__(self, hidden_dim, num_heads, dropout):
        super().__init__()
        
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
            for _ in range(2)
        ])
        
        self.feature_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 2)  # 3pt_eec, 4pt_eec
        )
    
    def forward(self, hyperedge_h, hyperedge_mask, n_hyperedges):
        """Generate hyperedge features"""
        batch_size = hyperedge_h.size(0)
        max_he = hyperedge_mask.size(1)
        
        # Expand
        hyperedge_h_expanded = hyperedge_h.unsqueeze(1).expand(batch_size, max_he, -1)
        
        # Apply MLP with residual (memory-efficient)
        for layer in self.mlp_layers:
            hyperedge_h_expanded = hyperedge_h_expanded.reshape(-1, hyperedge_h.size(-1))
            hyperedge_h_expanded = hyperedge_h_expanded + layer(hyperedge_h_expanded)
            hyperedge_h_expanded = hyperedge_h_expanded.reshape(batch_size, max_he, -1)
        
        # Generate features
        hyperedge_features = self.feature_proj(hyperedge_h_expanded)
        
        # Apply mask
        hyperedge_features = hyperedge_features * hyperedge_mask.unsqueeze(-1)
        
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
