"""Data normalization utilities for jet features"""
import torch
import numpy as np


class JetFeatureNormalizer:
    """
    Normalize jet features to prevent gradient explosion.
    
    Particle features:
    - pt: log-transform then standardize (handles exponential distribution)
    - eta: already in reasonable range [-2.5, 2.5]
    - phi: already in reasonable range [-π, π]
    
    Edge features: standardize
    Hyperedge features: standardize
    """
    
    def __init__(self):
        self.particle_mean = None
        self.particle_std = None
        self.edge_mean = None
        self.edge_std = None
        self.hyperedge_mean = None
        self.hyperedge_std = None
    
    def fit(self, dataset):
        """Compute normalization statistics from dataset"""
        print("Computing normalization statistics...")
        
        # Collect all features
        all_pt = []
        all_eta = []
        all_phi = []
        all_edges = []
        all_hyperedges = []
        
        for i in range(len(dataset)):
            data = dataset[i]
            particle_x = data.particle_x
            
            # Particles
            all_pt.append(particle_x[:, 0])
            all_eta.append(particle_x[:, 1])
            all_phi.append(particle_x[:, 2])
            
            # Edges
            if data.edge_attr.size(0) > 0:
                all_edges.append(data.edge_attr)
            
            # Hyperedges
            if data.hyperedge_x.size(0) > 0:
                all_hyperedges.append(data.hyperedge_x)
        
        # Compute statistics
        all_pt = torch.cat(all_pt)
        all_eta = torch.cat(all_eta)
        all_phi = torch.cat(all_phi)
        
        # Log-transform pt (add epsilon to avoid log(0))
        log_pt = torch.log(all_pt + 1e-8)
        
        # Particle normalization (log_pt, eta, phi)
        self.particle_mean = torch.stack([
            log_pt.mean(),
            all_eta.mean(),
            all_phi.mean()
        ])
        self.particle_std = torch.stack([
            log_pt.std(),
            all_eta.std(),
            all_phi.std()
        ])
        
        # Clamp std to avoid division by zero
        self.particle_std = torch.clamp(self.particle_std, min=1e-6)
        
        # Edge normalization
        if len(all_edges) > 0:
            all_edges = torch.cat(all_edges, dim=0)
            self.edge_mean = all_edges.mean(dim=0)
            self.edge_std = all_edges.std(dim=0).clamp(min=1e-6)
        
        # Hyperedge normalization
        if len(all_hyperedges) > 0:
            all_hyperedges = torch.cat(all_hyperedges, dim=0)
            self.hyperedge_mean = all_hyperedges.mean(dim=0)
            self.hyperedge_std = all_hyperedges.std(dim=0).clamp(min=1e-6)
        
        print(f"Particle normalization:")
        print(f"  log(pt): mean={self.particle_mean[0]:.3f}, std={self.particle_std[0]:.3f}")
        print(f"  eta: mean={self.particle_mean[1]:.3f}, std={self.particle_std[1]:.3f}")
        print(f"  phi: mean={self.particle_mean[2]:.3f}, std={self.particle_std[2]:.3f}")
    
    def normalize_particles(self, particle_x):
        """Normalize particle features (pt, eta, phi)"""
        if self.particle_mean is None:
            return particle_x
        
        # Log-transform pt
        pt = particle_x[:, 0:1]
        log_pt = torch.log(pt + 1e-8)
        
        # Stack and normalize
        features = torch.cat([log_pt, particle_x[:, 1:3]], dim=-1)
        normalized = (features - self.particle_mean.to(features.device)) / self.particle_std.to(features.device)
        
        return normalized
    
    def denormalize_particles(self, normalized_x):
        """Denormalize particle features back to original space"""
        if self.particle_mean is None:
            return normalized_x
        
        # Denormalize
        features = normalized_x * self.particle_std.to(normalized_x.device) + self.particle_mean.to(normalized_x.device)
        
        # Exp-transform back to pt
        pt = torch.exp(features[:, 0:1])
        eta = features[:, 1:2]
        phi = features[:, 2:3]
        
        return torch.cat([pt, eta, phi], dim=-1)
    
    def normalize_edges(self, edge_attr):
        """Normalize edge features"""
        if self.edge_mean is None or edge_attr.size(0) == 0:
            return edge_attr
        
        return (edge_attr - self.edge_mean.to(edge_attr.device)) / self.edge_std.to(edge_attr.device)
    
    def normalize_hyperedges(self, hyperedge_x):
        """Normalize hyperedge features"""
        if self.hyperedge_mean is None or hyperedge_x.size(0) == 0:
            return hyperedge_x
        
        return (hyperedge_x - self.hyperedge_mean.to(hyperedge_x.device)) / self.hyperedge_std.to(hyperedge_x.device)
    
    def save(self, path):
        """Save normalization statistics"""
        torch.save({
            'particle_mean': self.particle_mean,
            'particle_std': self.particle_std,
            'edge_mean': self.edge_mean,
            'edge_std': self.edge_std,
            'hyperedge_mean': self.hyperedge_mean,
            'hyperedge_std': self.hyperedge_std
        }, path)
        print(f"Saved normalization stats to {path}")
    
    def load(self, path):
        """Load normalization statistics"""
        stats = torch.load(path)
        self.particle_mean = stats['particle_mean']
        self.particle_std = stats['particle_std']
        self.edge_mean = stats.get('edge_mean')
        self.edge_std = stats.get('edge_std')
        self.hyperedge_mean = stats.get('hyperedge_mean')
        self.hyperedge_std = stats.get('hyperedge_std')
        print(f"Loaded normalization stats from {path}")
