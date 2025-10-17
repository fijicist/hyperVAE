import torch
import torch.nn as nn
import torch.nn.functional as F
from .encoder import BipartiteEncoder
from .decoder import BipartiteDecoder


class BipartiteHyperVAE(nn.Module):
    """
    Bipartite Hypergraph Variational Autoencoder for Jet Generation.
    
    Combines Lorentz-equivariant attention for physics-aware particle encoding
    with edge-aware attention for particle pairs and hyperedge correlations.
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.encoder = BipartiteEncoder(config['model'])
        self.decoder = BipartiteDecoder(config['model'])
        
        # Loss weights
        self.loss_weights = config['training']['loss_weights']
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, batch, temperature=1.0):
        """
        Forward pass through VAE.
        
        Args:
            batch: PyG Batch with bipartite graph data
            temperature: Gumbel-softmax temperature
        
        Returns:
            Dictionary with reconstructed features and latent parameters
        """
        # Encode
        mu, logvar = self.encoder(batch)
        
        # Sample latent
        z = self.reparameterize(mu, logvar)
        
        # Decode
        output = self.decoder(z, batch.jet_type, temperature)
        
        output['mu'] = mu
        output['logvar'] = logvar
        output['z'] = z
        
        return output
    
    def compute_loss(self, batch, output, epoch=0):
        """
        Compute total VAE loss.
        
        Loss = MSE(particles) + MSE(edges) + MSE(hyperedges) + BCE(topology) + KL
        """
        losses = {}
        
        # 1. Particle feature reconstruction loss
        particle_loss = self._particle_reconstruction_loss(
            batch, output['particle_features'], output['topology']['particle_mask']
        )
        losses['particle'] = particle_loss
        
        # 2. Edge feature reconstruction loss (if edges present)
        if batch.edge_attr.size(0) > 0:
            edge_loss = self._edge_reconstruction_loss(batch, output['edge_features'])
            losses['edge'] = edge_loss
        else:
            losses['edge'] = torch.tensor(0.0, device=batch.particle_x.device)
        
        # 3. Hyperedge feature reconstruction loss
        hyperedge_loss = self._hyperedge_reconstruction_loss(
            batch, output['hyperedge_features'], output['topology']['hyperedge_mask']
        )
        losses['hyperedge'] = hyperedge_loss
        
        # 4. Topology loss
        topology_loss = self._topology_loss(batch, output['topology'])
        losses['topology'] = topology_loss
        
        # 5. KL divergence
        kl_loss = self._kl_divergence(output['mu'], output['logvar'])
        
        # KL annealing with maximum cap
        kl_warmup_epochs = self.config['training'].get('kl_warmup_epochs', 100)
        kl_max_weight = self.config['training'].get('kl_max_weight', 1.0)
        kl_weight = min(kl_max_weight, epoch / kl_warmup_epochs) if kl_warmup_epochs > 0 else kl_max_weight
        losses['kl'] = kl_loss
        losses['kl_raw'] = kl_loss.item()  # Log raw KL for monitoring
        
        # Total loss
        total_loss = (
            self.loss_weights['particle_features'] * losses['particle'] +
            self.loss_weights['edge_features'] * losses['edge'] +
            self.loss_weights['hyperedge_features'] * losses['hyperedge'] +
            self.loss_weights['topology'] * losses['topology'] +
            self.loss_weights['kl_divergence'] * kl_weight * losses['kl']
        )
        
        losses['total'] = total_loss
        losses['kl_weight'] = kl_weight
        
        return losses
    
    def _particle_reconstruction_loss(self, batch, pred_features, pred_mask):
        """MSE loss for particle features (pt, eta, phi)"""
        # Extract true features
        true_features = batch.particle_x  # [N_particles, 3]
        
        # Create batch assignment and reconstruct per-graph features
        batch_size = batch.num_graphs
        loss = 0.0
        
        cumulative_particles = 0
        for i in range(batch_size):
            n_true = batch.n_particles[i].item()
            true_particles = true_features[cumulative_particles:cumulative_particles + n_true]
            
            pred_particles = pred_features[i, :n_true]
            
            # MSE on existing particles
            loss += F.mse_loss(pred_particles, true_particles)
            
            cumulative_particles += n_true
        
        return loss / batch_size
    
    def _edge_reconstruction_loss(self, batch, pred_features):
        """MSE loss for edge features"""
        # Simplified: average over all edge features
        # Proper implementation would align predicted and true edges
        true_features = batch.edge_attr
        
        # For now, compute distribution matching loss
        true_mean = true_features.mean(dim=0)
        pred_mean = pred_features.reshape(-1, 5).mean(dim=0)
        
        loss = F.mse_loss(pred_mean, true_mean)
        return loss
    
    def _hyperedge_reconstruction_loss(self, batch, pred_features, pred_mask):
        """MSE loss for hyperedge features"""
        true_features = batch.hyperedge_x
        
        # Distribution matching
        true_mean = true_features.mean(dim=0)
        pred_mean = pred_features.reshape(-1, 2).mean(dim=0)
        
        loss = F.mse_loss(pred_mean, true_mean)
        return loss
    
    def _topology_loss(self, batch, topology):
        """BCE loss for topology prediction"""
        # Number of particles loss
        true_n_particles = batch.n_particles.float()
        pred_n_particles = topology['n_particles'].float()
        
        particle_count_loss = F.mse_loss(pred_n_particles, true_n_particles)
        
        # Number of hyperedges loss
        true_n_hyperedges = batch.n_hyperedges.float()
        pred_n_hyperedges = topology['n_hyperedges'].float()
        
        hyperedge_count_loss = F.mse_loss(pred_n_hyperedges, true_n_hyperedges)
        
        return particle_count_loss + hyperedge_count_loss
    
    def _kl_divergence(self, mu, logvar):
        """KL divergence between q(z|x) and p(z) = N(0,I)"""
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
        return kl_loss.mean()
    
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
    print(f"  Topology loss: {losses['topology'].item():.4f}")
    print(f"  KL loss: {losses['kl'].item():.4f}")
    
    # Test generation
    print("\nTesting generation...")
    generated = model.generate(4, torch.tensor([0, 1, 2, 0]), device='cpu')
    print(f"Generated particle features: {generated['particle_features'].shape}")
