import torch
import numpy as np
from torch.utils.data import Dataset, Subset
from torch_geometric.data import Data, Batch
import os


class BipartiteJetDataset(Dataset):
    """
    Dataset for hypergraph representation of jets.
    
    Expected format (PyG Data object):
    - x: [N_particles, 3] - Node features (pt, eta, phi)
    - edge_index: [2, N_edges] - Edge connectivity
    - edge_attr: [N_edges, 5] - Edge features (ln_delta, ln_kt, ln_z, ln_m2, feat5)
    - hyperedge_index: [2, N_hyperedge_connections] - Hyperedge connectivity
    - hyperedge_attr: [N_unique_hyperedges, 2] - Hyperedge features (3pt_eec, 4pt_eec)
    - y: [1] - Jet type (0=quark, 1=gluon, 2=top)
    """
    
    def __init__(self, data_path=None, max_particles=150, generate_dummy=False):
        """
        Args:
            data_path: Path to .pt file with jet data (PyG Data list)
            max_particles: Maximum particles per jet
            generate_dummy: If True, generate dummy data for testing
        """
        self.max_particles = max_particles
        
        if generate_dummy:
            self.data = self._generate_dummy_data(num_samples=1000)
        elif data_path:
            self.data = self._load_from_pt(data_path)
        else:
            raise ValueError("Provide data_path or set generate_dummy=True")
    
    def _load_from_pt(self, data_path):
        """Load jet data from .pt file"""
        print(f"Loading jet data from {data_path}...")
        
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")
        
        # Load PyG Data list
        data_list = torch.load(data_path)
        
        if not isinstance(data_list, list):
            raise ValueError(f"Expected list of Data objects, got {type(data_list)}")
        
        print(f"Loaded {len(data_list)} jets")
        
        # Validate first sample
        if len(data_list) > 0:
            sample = data_list[0]
            print(f"Sample jet structure:")
            print(f"  x (particles): {sample.x.shape}")
            print(f"  edge_index: {sample.edge_index.shape}")
            print(f"  edge_attr: {sample.edge_attr.shape}")
            if hasattr(sample, 'hyperedge_index'):
                print(f"  hyperedge_index: {sample.hyperedge_index.shape}")
            if hasattr(sample, 'hyperedge_attr'):
                print(f"  hyperedge_attr: {sample.hyperedge_attr.shape}")
            print(f"  y (jet type): {sample.y}")
        
        # Convert to our format
        converted_data = []
        for data in data_list:
            converted = self._convert_to_bipartite(data)
            converted_data.append(converted)
        
        return converted_data
    
    def _convert_to_bipartite(self, data):
        """
        Convert PyG Data to our bipartite format.
        
        Args:
            data: PyG Data object with x, edge_index, edge_attr, hyperedge_index, hyperedge_attr, y
        
        Returns:
            Dictionary with bipartite graph data
        """
        # Extract particle features (x) - only pt, eta, phi (no mass)
        particle_features = data.x.float()  # [N_particles, 3]
        if particle_features.shape[1] == 4:
            # If data has 4 features, keep only first 3 (pt, eta, phi)
            particle_features = particle_features[:, :3]
        elif particle_features.shape[1] != 3:
            raise ValueError(f"Expected 3 or 4 particle features, got {particle_features.shape[1]}")
        
        # Extract edge features
        edge_features = data.edge_attr.float() if hasattr(data, 'edge_attr') and data.edge_attr is not None else torch.zeros((0, 5))
        
        # Extract hyperedge features
        if hasattr(data, 'hyperedge_attr') and data.hyperedge_attr is not None:
            hyperedge_features = data.hyperedge_attr.float()  # [N_hyperedges, 2]
        else:
            hyperedge_features = torch.zeros((0, 2))
        
        # Build bipartite edge index (particle -> hyperedge)
        # hyperedge_index format: [2, N_connections] where each column is [particle_id, hyperedge_id]
        if hasattr(data, 'hyperedge_index') and data.hyperedge_index is not None:
            edge_index_bipartite = data.hyperedge_index.long()
        else:
            edge_index_bipartite = torch.zeros((2, 0), dtype=torch.long)
        
        # Jet type
        jet_type = data.y.long() if hasattr(data, 'y') else torch.tensor([0], dtype=torch.long)
        if jet_type.dim() == 2:
            jet_type = jet_type.squeeze(1)
        
        return {
            'particle_features': particle_features,
            'edge_features': edge_features,
            'hyperedge_features': hyperedge_features,
            'edge_index_bipartite': edge_index_bipartite,
            'jet_type': jet_type,
            'n_particles': particle_features.shape[0],
            'n_hyperedges': hyperedge_features.shape[0]
        }
    
    def _generate_dummy_data(self, num_samples):
        """Generate dummy jet data for testing (matching your .pt format)"""
        print(f"Generating {num_samples} dummy jet samples...")
        data = []
        
        for i in range(num_samples):
            # Random number of particles (15-50 typical for jets)
            n_particles = np.random.randint(15, 50)
            
            # Particle features (pt, eta, phi) - no mass
            pt = np.random.exponential(20, n_particles).astype(np.float32)
            eta = np.random.uniform(-2.5, 2.5, n_particles).astype(np.float32)
            phi = np.random.uniform(-np.pi, np.pi, n_particles).astype(np.float32)
            particle_features = torch.from_numpy(np.stack([pt, eta, phi], axis=1))  # [N, 3]
            
            # Number of hyperedges (pairs and triplets/quadruplets)
            n_edges = min(n_particles * (n_particles - 1) // 2, 200)  # Limit edges
            n_hyperedges = min(n_particles // 3, 30)  # Limit hyperedges
            
            # Edge features (5D: ln_delta, ln_kt, ln_z, ln_m2, feat5)
            edge_features = torch.randn(n_edges, 5)
            
            # Hyperedge features (2D: 3pt_eec, 4pt_eec)
            hyperedge_features = torch.randn(n_hyperedges, 2)
            
            # Bipartite graph structure
            # Edge connections: particle -> hyperedge
            # Randomly connect particles to hyperedges (3-4 particles per hyperedge)
            edge_index_bipartite = []
            for h_idx in range(n_hyperedges):
                n_particles_in_hyperedge = np.random.randint(3, 5)
                particles_in_hyperedge = np.random.choice(
                    n_particles, n_particles_in_hyperedge, replace=False
                )
                for p_idx in particles_in_hyperedge:
                    edge_index_bipartite.append([p_idx, h_idx])
            
            if len(edge_index_bipartite) > 0:
                edge_index_bipartite = torch.tensor(edge_index_bipartite).T.long()
            else:
                edge_index_bipartite = torch.zeros((2, 0), dtype=torch.long)
            
            # Jet type (0: quark, 1: gluon, 2: top)
            jet_type = torch.tensor([i % 3], dtype=torch.long)
            
            data.append({
                'particle_features': particle_features,
                'edge_features': edge_features,
                'hyperedge_features': hyperedge_features,
                'edge_index_bipartite': edge_index_bipartite,
                'jet_type': jet_type,
                'n_particles': n_particles,
                'n_hyperedges': n_hyperedges
            })
        
        return data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """Return PyG Data object for bipartite graph"""
        sample = self.data[idx]
        
        # Create bipartite graph
        # Node features: [particles; hyperedges]
        particle_x = sample['particle_features']
        hyperedge_x = sample['hyperedge_features']
        
        # Pad to max_particles for batching
        n_particles = sample['n_particles']
        n_hyperedges = sample['n_hyperedges']
        
        # Create data object
        data = Data(
            particle_x=particle_x,
            hyperedge_x=hyperedge_x,
            edge_attr=sample['edge_features'],
            edge_index=sample['edge_index_bipartite'],
            jet_type=sample['jet_type'],
            n_particles=torch.tensor([n_particles]),
            n_hyperedges=torch.tensor([n_hyperedges]),
            num_nodes=n_particles + n_hyperedges  # Total nodes in bipartite graph
        )
        
        return data


def collate_bipartite_batch(data_list):
    """Custom collate function for bipartite graphs"""
    batch = Batch.from_data_list(data_list)
    return batch


def create_train_val_test_split(dataset, train_frac=0.8, val_frac=0.1, test_frac=0.1, seed=42):
    """
    Split dataset into train/val/test subsets.
    
    Args:
        dataset: BipartiteJetDataset instance
        train_frac: Fraction for training (default: 0.8)
        val_frac: Fraction for validation (default: 0.1)
        test_frac: Fraction for testing (default: 0.1)
        seed: Random seed for reproducibility
    
    Returns:
        train_dataset, val_dataset, test_dataset
    """
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6, "Fractions must sum to 1.0"
    
    # Set seed for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Get total size
    total_size = len(dataset)
    indices = np.random.permutation(total_size)
    
    # Calculate split sizes
    train_size = int(total_size * train_frac)
    val_size = int(total_size * val_frac)
    test_size = total_size - train_size - val_size
    
    # Split indices
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    # Create subsets
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)
    
    print(f"\nDataset split:")
    print(f"  Total: {total_size} jets")
    print(f"  Train: {len(train_dataset)} jets ({train_frac*100:.1f}%)")
    print(f"  Val:   {len(val_dataset)} jets ({val_frac*100:.1f}%)")
    print(f"  Test:  {len(test_dataset)} jets ({test_frac*100:.1f}%)")
    
    return train_dataset, val_dataset, test_dataset


if __name__ == "__main__":
    # Test dataset
    dataset = BipartiteJetDataset(generate_dummy=True)
    print(f"Dataset size: {len(dataset)}")
    
    sample = dataset[0]
    print(f"Sample jet:")
    print(f"  Particles: {sample.particle_x.shape}")
    print(f"  Hyperedges: {sample.hyperedge_x.shape}")
    print(f"  Edge features: {sample.edge_attr.shape}")
    print(f"  Bipartite edges: {sample.edge_index.shape}")
    print(f"  Jet type: {sample.jet_type}")
    
    # Test batching
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_bipartite_batch)
    batch = next(iter(loader))
    print(f"\nBatched data:")
    print(f"  Batch: {batch}")
    print(f"  Total particles: {batch.particle_x.shape[0]}")
    print(f"  Total hyperedges: {batch.hyperedge_x.shape[0]}")
