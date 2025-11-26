"""
═══════════════════════════════════════════════════════════════════════════════
BIPARTITE JET DATASET - PYTORCH GEOMETRIC DATA LOADER
═══════════════════════════════════════════════════════════════════════════════

Purpose:
--------
Load and preprocess jet data in PyTorch Geometric bipartite hypergraph format.

Data Format (PyG Data object):
-------------------------------
Each jet is represented as a PyG Data object with:

Nodes (Particles):
  - particle_x: [N_particles, 4] - 4-momenta [E, px, py, pz] (normalized)
  - n_particles: int - Number of particles in this jet

Edges (Pairwise particle connections):
  - edge_index: [2, N_edges] - Particle pairs (bipartite adjacency)
  - edge_attr: [N_edges, 5] - Edge features [ln_delta, ln_kt, ln_z, ln_m2, feat5]

Hyperedges (Higher-order particle groupings):
  - hyperedge_index: [2, N_connections] - Particle-hyperedge incidence
  - hyperedge_x: [N_hyperedges, 2] - Hyperedge features [3pt_eec, 4pt_eec]
  - n_hyperedges: int - Number of hyperedges in this jet

Jet-level:
  - y: [num_features] - [jet_type, jet_pt, jet_eta, jet_mass, ...]
  - jet_type: int - Jet flavor (0=quark, 1=gluon, 2=top)

Usage:
------
Load from file:
    dataset = BipartiteJetDataset(data_path='jets.pt')

Generate dummy data for testing:
    dataset = BipartiteJetDataset(generate_dummy=True, num_samples=1000)

Split into train/val/test:
    train, val, test = create_train_val_test_split(dataset)

DataLoader with collation:
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=32, collate_fn=collate_bipartite_batch)

Data Preprocessing:
-------------------
- 4-momenta: Z-score normalized per feature (E, px, py, pz)
- Edge features: Log-transformed and normalized
- Jet features: Standardized to [0, 1] or [-1, 1] range
- Variable length: Handled via PyG Batch with n_particles/n_edges/n_hyperedges

Collation:
----------
collate_bipartite_batch(): Batches multiple jets into single PyG Batch object
  - Concatenates all particles/edges/hyperedges across batch
  - Maintains per-jet indexing via n_particles, n_edges, n_hyperedges
  - Adjusts edge_index and hyperedge_index offsets for batching

═══════════════════════════════════════════════════════════════════════════════
"""
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
    
    def __init__(self, data_path=None, max_particles=150, generate_dummy=False, num_samples=1000):
        """
        Args:
            data_path: Path to .pt file with jet data (PyG Data list)
            max_particles: Maximum particles per jet
            generate_dummy: If True, generate dummy data for testing
            num_samples: Number of dummy samples to generate (only used if generate_dummy=True)
        """
        self.max_particles = max_particles
        
        # Normalization statistics (extracted from graph_constructor.py output)
        # These are the same for all samples since normalization was done globally
        self.particle_norm_stats = None
        self.jet_norm_stats = None
        self.edge_norm_stats = None
        self.hyperedge_norm_stats = None
        
        if generate_dummy:
            self.data = self._generate_dummy_data(num_samples=num_samples)
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
        data_list = torch.load(data_path, weights_only=False)
        
        if not isinstance(data_list, list):
            raise ValueError(f"Expected list of Data objects, got {type(data_list)}")
        
        print(f"Loaded {len(data_list)} jets")
        
        # Extract normalization statistics from first graph (they're the same for all)
        if len(data_list) > 0:
            first_graph = data_list[0]
            
            # Store normalization statistics at dataset level
            if hasattr(first_graph, 'particle_norm_stats'):
                self.particle_norm_stats = first_graph.particle_norm_stats
                print(f"Extracted particle normalization statistics")
            if hasattr(first_graph, 'jet_norm_stats'):
                self.jet_norm_stats = first_graph.jet_norm_stats
                print(f"Extracted jet normalization statistics")
            if hasattr(first_graph, 'edge_norm_stats'):
                self.edge_norm_stats = first_graph.edge_norm_stats
                print(f"Extracted edge normalization statistics")
            if hasattr(first_graph, 'hyperedge_norm_stats'):
                self.hyperedge_norm_stats = first_graph.hyperedge_norm_stats
                print(f"Extracted hyperedge normalization statistics")
        
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
        # Extract particle features (x) - keep all features
        particle_features = data.x.float()  # [N_particles, num_features]
        # Accept any number of features (3, 4, or more)
        
        # Extract edge features
        if hasattr(data, 'edge_attr') and data.edge_attr is not None:
            edge_features = data.edge_attr.float()
        else:
            edge_features = torch.zeros((0, 5))  # Default to 5 features if none
        
        # Extract hyperedge features
        if hasattr(data, 'hyperedge_attr') and data.hyperedge_attr is not None:
            hyperedge_features = data.hyperedge_attr.float()
        else:
            hyperedge_features = torch.zeros((0, 1))  # Default to 1 feature if none
        
        # Extract edge_index (pairwise particle edges)
        if hasattr(data, 'edge_index') and data.edge_index is not None:
            edge_index = data.edge_index.long()
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
        
        # Build bipartite edge index (particle -> hyperedge)
        # hyperedge_index format: [2, N_connections] where each column is [particle_id, hyperedge_id]
        if hasattr(data, 'hyperedge_index') and data.hyperedge_index is not None:
            hyperedge_index = data.hyperedge_index.long()
        else:
            hyperedge_index = torch.zeros((2, 0), dtype=torch.long)
        
        # Jet type and jet features
        # y tensor contains: [jet_type, jet_pt, jet_eta, jet_mass, ...]
        if hasattr(data, 'y') and data.y is not None:
            y_tensor = data.y.float() if data.y.dtype != torch.long else data.y.float()
            # Keep as 1D [4] to match graph_constructor.py format
            if y_tensor.dim() > 1:
                y_tensor = y_tensor.squeeze(0)  # [4]
            
            # Extract jet_type (first element, convert to long)
            jet_type = torch.tensor([int(y_tensor[0].item())], dtype=torch.long)
        else:
            y_tensor = torch.tensor([0.0, 0.0, 0.0, 0.0], dtype=torch.float)  # Default [4]
            jet_type = torch.tensor([0], dtype=torch.long)
        
        return {
            'particle_features': particle_features,
            'edge_index': edge_index,  # Pairwise particle edges
            'edge_features': edge_features,
            'hyperedge_index': hyperedge_index,  # Hyperedge connections
            'hyperedge_features': hyperedge_features,
            'jet_type': jet_type,
            'y': y_tensor,  # [4] tensor: [jet_type, log(pt), eta, log(mass)]
            'n_particles': particle_features.shape[0],
            'n_hyperedges': hyperedge_features.shape[0]
        }
    
    def _generate_dummy_data(self, num_samples):
        """Generate dummy jet data for testing (matching graph_constructor.py format)"""
        print(f"Generating {num_samples} dummy jet samples...")
        data = []
        
        for i in range(num_samples):
            # Random number of particles (15-50 typical for jets)
            n_particles = np.random.randint(15, 50)
            
            # Particle 4-momentum features [E, px, py, pz] - matching graph_constructor.py
            # Generate physically valid 4-momenta: E² = px² + py² + pz² + m²
            px = np.random.normal(0, 10, n_particles).astype(np.float32)
            py = np.random.normal(0, 10, n_particles).astype(np.float32)
            pz = np.random.normal(0, 15, n_particles).astype(np.float32)
            m = np.random.uniform(0.1, 1.0, n_particles).astype(np.float32)  # Small masses
            E = np.sqrt(px**2 + py**2 + pz**2 + m**2)  # Satisfy energy-momentum relation
            particle_features = torch.from_numpy(np.stack([E, px, py, pz], axis=1))  # [N, 4]
            
            # Number of edges and hyperedges
            n_edges = min(n_particles * (n_particles - 1) // 2, 200)  # Limit edges
            n_hyperedges = min(n_particles // 3, 30)  # Limit hyperedges
            
            # Edge features (5D: 2pt_EEC, ln_delta, ln_kT, ln_z, ln_m²) - matching graph_constructor.py
            edge_features = torch.randn(n_edges, 5)
            
            # Edge index (fully connected for dummy data)
            edge_pairs = []
            for i_p in range(min(n_particles, 20)):  # Limit for dummy data
                for j_p in range(i_p + 1, min(n_particles, 20)):
                    edge_pairs.append([i_p, j_p])
            if len(edge_pairs) > 0:
                edge_index = torch.tensor(edge_pairs[:n_edges]).T.long()  # [2, E]
            else:
                edge_index = torch.zeros((2, 0), dtype=torch.long)
            
            # Hyperedge features (N-point EEC) - matching graph_constructor.py
            hyperedge_features = torch.randn(n_hyperedges, 1) if n_hyperedges > 0 else torch.zeros((0, 1))
            
            # Hyperedge index (particle-hyperedge connections)
            hyperedge_index_list = []
            for h_idx in range(n_hyperedges):
                n_particles_in_hyperedge = np.random.randint(3, 5)
                particles_in_hyperedge = np.random.choice(
                    n_particles, n_particles_in_hyperedge, replace=False
                )
                for p_idx in particles_in_hyperedge:
                    hyperedge_index_list.append([p_idx, h_idx])
            
            if len(hyperedge_index_list) > 0:
                hyperedge_index = torch.tensor(hyperedge_index_list).T.long()
            else:
                hyperedge_index = torch.zeros((2, 0), dtype=torch.long)
            
            # Jet type (0: quark, 1: gluon, 2: top)
            jet_type = i % 3
            
            # Generate jet-level features: [jet_type, log(pt), eta, log(mass)] - matching graph_constructor.py
            jet_pt = np.sum(np.sqrt(px**2 + py**2))  # Sum of particle pts
            # Compute eta safely: eta = arctanh(pz/E), clip to avoid NaN
            pz_over_E = np.clip(pz / (E + 1e-8), -0.99, 0.99)  # Clip to valid range for arctanh
            jet_eta = np.mean(np.arctanh(pz_over_E))  # Average eta
            jet_mass = np.random.uniform(50, 200)  # Random jet mass
            y_tensor = torch.tensor([jet_type, np.log(jet_pt + 1), jet_eta, np.log(jet_mass + 1)], dtype=torch.float)
            
            data.append({
                'particle_features': particle_features,
                'edge_index': edge_index,
                'edge_features': edge_features,
                'hyperedge_index': hyperedge_index,
                'hyperedge_features': hyperedge_features,
                'jet_type': torch.tensor([jet_type], dtype=torch.long),
                'y': y_tensor,  # [4] tensor
                'n_particles': n_particles,
                'n_hyperedges': n_hyperedges
            })
        
        return data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """Return PyG Data object matching graph_constructor.py format"""
        sample = self.data[idx]
        
        # Create data object with field names matching graph_constructor.py
        # x: particle 4-momenta [N, 4]
        # edge_index: edge connectivity [2, E]
        # edge_attr: edge features [E, 5]
        # hyperedge_index: hyperedge connectivity [2, H] (optional)
        # hyperedge_attr: hyperedge features [K, F] (optional)
        # y: jet features [4] - [jet_type, log(pt), eta, log(mass)]
        
        data = Data(
            x=sample['particle_features'],  # Changed from particle_x to x
            edge_index=sample['edge_index'],  # Changed from edge_index_bipartite
            edge_attr=sample['edge_features'],
            hyperedge_index=sample.get('hyperedge_index'),  # Optional
            hyperedge_attr=sample['hyperedge_features'],  # Changed from hyperedge_x
            y=sample['y'],  # [4] tensor
            jet_type=sample['jet_type'],  # Keep for backward compatibility
            n_particles=torch.tensor([sample['n_particles']]),
            n_hyperedges=torch.tensor([sample['n_hyperedges']])
        )
        
        return data


def collate_bipartite_batch(data_list):
    """
    Custom collate function for bipartite graphs.
    
    Note: Normalization statistics are added in train.py via a wrapper function
    that accesses the dataset object directly.
    
    Args:
        data_list: List of PyG Data objects
    
    Returns:
        PyG Batch object
    """
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
