import torch
import yaml
import argparse
import numpy as np
from tqdm import tqdm
from torch_geometric.data import Data

from models.hypervae import BipartiteHyperVAE


def generate_jets(model, num_samples, jet_type_dist, device, batch_size=32):
    """
    Generate jets using the trained model.
    
    Args:
        model: Trained BipartiteHyperVAE model
        num_samples: Number of jets to generate
        jet_type_dist: Distribution of jet types [quark_frac, gluon_frac, top_frac]
        device: Device to generate on
        batch_size: Batch size for generation
    
    Returns:
        Dictionary with generated jet features
    """
    model.eval()
    
    # Sample jet types according to distribution
    jet_types = np.random.choice(
        [0, 1, 2], 
        size=num_samples, 
        p=jet_type_dist
    )
    
    all_particle_features = []
    all_edge_features = []
    all_hyperedge_features = []
    all_jet_types = []
    all_n_particles = []
    all_n_hyperedges = []
    
    with torch.no_grad():
        for i in tqdm(range(0, num_samples, batch_size), desc="Generating jets"):
            batch_jet_types = jet_types[i:i+batch_size]
            current_batch_size = len(batch_jet_types)
            
            # Sample from prior N(0, I)
            z = torch.randn(current_batch_size, model.config['model']['latent_dim'], device=device)
            jet_type_tensor = torch.tensor(batch_jet_types, dtype=torch.long, device=device)
            
            # Generate
            output = model.decoder(z, jet_type_tensor, temperature=0.5)
            
            # Extract features
            particle_features = output['particle_features'].cpu().numpy()
            edge_features = output['edge_features'].cpu().numpy()
            hyperedge_features = output['hyperedge_features'].cpu().numpy()
            n_particles = output['topology']['n_particles'].cpu().numpy()
            n_hyperedges = output['topology']['n_hyperedges'].cpu().numpy()
            
            # Store
            for j in range(current_batch_size):
                n_p = int(n_particles[j])
                n_he = int(n_hyperedges[j])
                
                # Only store non-zero particles
                if n_p > 0:
                    all_particle_features.append(particle_features[j, :n_p])
                    all_n_particles.append(n_p)
                else:
                    all_particle_features.append(np.zeros((1, 4)))
                    all_n_particles.append(1)
                
                # Store edge and hyperedge features
                all_edge_features.append(edge_features[j])
                all_hyperedge_features.append(hyperedge_features[j, :n_he] if n_he > 0 else np.zeros((1, 2)))
                all_n_hyperedges.append(n_he)
                all_jet_types.append(batch_jet_types[j])
    
    return {
        'particle_features': all_particle_features,
        'edge_features': all_edge_features,
        'hyperedge_features': all_hyperedge_features,
        'jet_types': np.array(all_jet_types),
        'n_particles': np.array(all_n_particles),
        'n_hyperedges': np.array(all_n_hyperedges)
    }


def save_generated_jets(generated_data, output_path):
    """Save generated jets to .pt file (PyG Data format)"""
    print(f"\nSaving generated jets to {output_path}...")
    
    data_list = []
    
    for i in range(len(generated_data['jet_types'])):
        # Get features
        particle_feat = torch.tensor(generated_data['particle_features'][i], dtype=torch.float32)
        edge_feat = torch.tensor(generated_data['edge_features'][i], dtype=torch.float32)
        hyperedge_feat = torch.tensor(generated_data['hyperedge_features'][i], dtype=torch.float32)
        
        # Create dummy edge_index (2-point connections between particles)
        n_particles = particle_feat.shape[0]
        if n_particles > 1:
            # Create pairwise edges
            edge_pairs = []
            for j in range(min(n_particles, 10)):  # Limit edges
                for k in range(j+1, min(n_particles, 10)):
                    edge_pairs.append([j, k])
            if edge_pairs:
                edge_index = torch.tensor(edge_pairs, dtype=torch.long).T
            else:
                edge_index = torch.zeros((2, 0), dtype=torch.long)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
        
        # Create dummy hyperedge_index (particle -> hyperedge connections)
        n_hyperedges = hyperedge_feat.shape[0]
        hyperedge_connections = []
        for h_idx in range(n_hyperedges):
            # Connect 3-4 random particles to each hyperedge
            n_conn = min(np.random.randint(3, 5), n_particles)
            particles = np.random.choice(n_particles, n_conn, replace=False)
            for p_idx in particles:
                hyperedge_connections.append([p_idx, h_idx])
        
        if hyperedge_connections:
            hyperedge_index = torch.tensor(hyperedge_connections, dtype=torch.long).T
        else:
            hyperedge_index = torch.zeros((2, 0), dtype=torch.long)
        
        # Create PyG Data object
        data = Data(
            x=particle_feat,
            edge_index=edge_index,
            edge_attr=edge_feat[:edge_index.shape[1]] if edge_feat.shape[0] > 0 else torch.zeros((edge_index.shape[1], 5)),
            hyperedge_index=hyperedge_index,
            hyperedge_attr=hyperedge_feat,
            y=torch.tensor([generated_data['jet_types'][i]], dtype=torch.long)
        )
        
        data_list.append(data)
    
    # Save as .pt file
    torch.save(data_list, output_path)
    print(f"Saved {len(data_list)} generated jets in PyG format")


def print_statistics(generated_data):
    """Print statistics of generated jets"""
    print("\n" + "="*60)
    print("Generated Jet Statistics")
    print("="*60)
    
    # Jet type distribution
    jet_types = generated_data['jet_types']
    print(f"\nJet Type Distribution:")
    print(f"  Quark: {(jet_types == 0).sum()} ({(jet_types == 0).mean()*100:.1f}%)")
    print(f"  Gluon: {(jet_types == 1).sum()} ({(jet_types == 1).mean()*100:.1f}%)")
    print(f"  Top:   {(jet_types == 2).sum()} ({(jet_types == 2).mean()*100:.1f}%)")
    
    # Number of particles
    n_particles = generated_data['n_particles']
    print(f"\nParticles per Jet:")
    print(f"  Mean: {n_particles.mean():.2f}")
    print(f"  Std:  {n_particles.std():.2f}")
    print(f"  Min:  {n_particles.min()}")
    print(f"  Max:  {n_particles.max()}")
    
    # Particle features statistics
    all_particles = np.concatenate(generated_data['particle_features'], axis=0)
    print(f"\nParticle Feature Statistics:")
    print(f"  Total particles: {all_particles.shape[0]}")
    print(f"  pt   - Mean: {all_particles[:, 0].mean():.2f}, Std: {all_particles[:, 0].std():.2f}")
    print(f"  eta  - Mean: {all_particles[:, 1].mean():.2f}, Std: {all_particles[:, 1].std():.2f}")
    print(f"  phi  - Mean: {all_particles[:, 2].mean():.2f}, Std: {all_particles[:, 2].std():.2f}")
    print(f"  mass - Mean: {all_particles[:, 3].mean():.2f}, Std: {all_particles[:, 3].std():.2f}")
    
    # Hyperedges
    n_hyperedges = generated_data['n_hyperedges']
    print(f"\nHyperedges per Jet:")
    print(f"  Mean: {n_hyperedges.mean():.2f}")
    print(f"  Std:  {n_hyperedges.std():.2f}")


def main(args):
    # Load config
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    config = checkpoint['config']
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() and args.gpu else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    model = BipartiteHyperVAE(config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
    
    # Jet type distribution
    jet_type_dist = [args.quark_frac, args.gluon_frac, args.top_frac]
    jet_type_dist = np.array(jet_type_dist) / sum(jet_type_dist)
    print(f"\nJet type distribution: Quark={jet_type_dist[0]:.2f}, "
          f"Gluon={jet_type_dist[1]:.2f}, Top={jet_type_dist[2]:.2f}")
    
    # Generate jets
    print(f"\nGenerating {args.num_samples} jets...")
    generated_data = generate_jets(
        model, 
        args.num_samples, 
        jet_type_dist, 
        device,
        batch_size=args.batch_size
    )
    
    # Print statistics
    print_statistics(generated_data)
    
    # Save
    if args.output:
        save_generated_jets(generated_data, args.output)
    
    print("\nGeneration completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate jets using trained HyperVAE')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--output', type=str, default='generated_jets.pt', help='Output .pt file')
    parser.add_argument('--num-samples', type=int, default=10000, help='Number of jets to generate')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for generation')
    parser.add_argument('--quark-frac', type=float, default=0.33, help='Fraction of quark jets')
    parser.add_argument('--gluon-frac', type=float, default=0.33, help='Fraction of gluon jets')
    parser.add_argument('--top-frac', type=float, default=0.34, help='Fraction of top jets')
    parser.add_argument('--gpu', action='store_true', help='Use GPU if available')
    
    args = parser.parse_args()
    main(args)
