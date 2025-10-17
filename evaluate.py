import torch
import numpy as np
from scipy.stats import wasserstein_distance
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm


def load_jets_from_pt(file_path, max_jets=None):
    """Load jet data from .pt file (PyG Data format)"""
    print(f"Loading jets from {file_path}...")
    
    data_list = torch.load(file_path)
    n_jets = len(data_list) if max_jets is None else min(max_jets, len(data_list))
    
    particle_features = []
    jet_types = []
    n_particles = []
    
    for i in range(n_jets):
        data = data_list[i]
        
        # Extract particle features (x)
        pf = data.x.cpu().numpy()
        if pf.shape[0] > 0:
            particle_features.append(pf)
            n_particles.append(pf.shape[0])
        
        # Extract jet type
        if hasattr(data, 'y') and data.y is not None:
            jet_type = data.y.cpu().numpy()
            if jet_type.ndim > 0:
                jet_type = jet_type[0]
            jet_types.append(jet_type)
        else:
            jet_types.append(0)
    
    return {
        'particle_features': particle_features,
        'jet_types': np.array(jet_types),
        'n_particles': np.array(n_particles)
    }


def compute_wasserstein_distances(real_data, generated_data):
    """
    Compute Wasserstein distances between real and generated distributions.
    
    Args:
        real_data: Dictionary with real jet data
        generated_data: Dictionary with generated jet data
    
    Returns:
        Dictionary with Wasserstein distances for each feature
    """
    print("\nComputing Wasserstein distances...")
    
    # Concatenate all particle features
    real_particles = np.concatenate(real_data['particle_features'], axis=0)
    gen_particles = np.concatenate(generated_data['particle_features'], axis=0)
    
    # Wasserstein distances for particle features
    w_distances = {}
    feature_names = ['pt', 'eta', 'phi']
    
    for i, name in enumerate(feature_names):
        real_feat = real_particles[:, i]
        gen_feat = gen_particles[:, i]
        
        # Remove zeros (masked particles)
        real_feat = real_feat[real_feat > 0] if name == 'pt' else real_feat
        gen_feat = gen_feat[gen_feat > 0] if name == 'pt' else gen_feat
        
        w_dist = wasserstein_distance(real_feat, gen_feat)
        w_distances[name] = w_dist
    
    # Wasserstein distance for number of particles
    w_distances['n_particles'] = wasserstein_distance(
        real_data['n_particles'], 
        generated_data['n_particles']
    )
    
    return w_distances


def compute_structural_metrics(real_data, generated_data):
    """Compute structural metrics for jets"""
    print("\nComputing structural metrics...")
    
    metrics = {}
    
    # Average number of particles
    metrics['mean_n_particles_real'] = real_data['n_particles'].mean()
    metrics['mean_n_particles_gen'] = generated_data['n_particles'].mean()
    metrics['std_n_particles_real'] = real_data['n_particles'].std()
    metrics['std_n_particles_gen'] = generated_data['n_particles'].std()
    
    # Jet type distribution
    for jet_type in [0, 1, 2]:
        jet_type_name = ['quark', 'gluon', 'top'][jet_type]
        metrics[f'frac_{jet_type_name}_real'] = (real_data['jet_types'] == jet_type).mean()
        metrics[f'frac_{jet_type_name}_gen'] = (generated_data['jet_types'] == jet_type).mean()
    
    return metrics


def plot_feature_distributions(real_data, generated_data, output_dir='plots'):
    """Plot feature distributions for comparison"""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nGenerating distribution plots in {output_dir}...")
    
    # Concatenate all particle features
    real_particles = np.concatenate(real_data['particle_features'], axis=0)
    gen_particles = np.concatenate(generated_data['particle_features'], axis=0)
    
    feature_names = ['pt', 'eta', 'phi']
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for i, name in enumerate(feature_names):
        ax = axes[i]
        
        real_feat = real_particles[:, i]
        gen_feat = gen_particles[:, i]
        
        # Remove zeros for pt
        if name == 'pt':
            real_feat = real_feat[real_feat > 0]
            gen_feat = gen_feat[gen_feat > 0]
        
        # Plot histograms
        bins = 50
        ax.hist(real_feat, bins=bins, alpha=0.5, label='Real', density=True, color='blue')
        ax.hist(gen_feat, bins=bins, alpha=0.5, label='Generated', density=True, color='orange')
        
        ax.set_xlabel(name)
        ax.set_ylabel('Density')
        ax.set_title(f'{name.capitalize()} Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/particle_features.png', dpi=150)
    plt.close()
    
    # Plot number of particles
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    bins = np.arange(0, max(real_data['n_particles'].max(), generated_data['n_particles'].max()) + 2)
    ax.hist(real_data['n_particles'], bins=bins, alpha=0.5, label='Real', density=True, color='blue')
    ax.hist(generated_data['n_particles'], bins=bins, alpha=0.5, label='Generated', density=True, color='orange')
    
    ax.set_xlabel('Number of Particles')
    ax.set_ylabel('Density')
    ax.set_title('Number of Particles per Jet')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/n_particles.png', dpi=150)
    plt.close()
    
    print(f"Saved plots to {output_dir}/")


def print_evaluation_results(w_distances, metrics):
    """Print evaluation results"""
    print("\n" + "="*60)
    print("Evaluation Results")
    print("="*60)
    
    print("\nWasserstein Distances:")
    print("-" * 40)
    for feature, distance in w_distances.items():
        print(f"  {feature:15s}: {distance:.6f}")
    
    print("\nStructural Metrics:")
    print("-" * 40)
    print(f"  Mean particles (real): {metrics['mean_n_particles_real']:.2f}")
    print(f"  Mean particles (gen):  {metrics['mean_n_particles_gen']:.2f}")
    print(f"  Std particles (real):  {metrics['std_n_particles_real']:.2f}")
    print(f"  Std particles (gen):   {metrics['std_n_particles_gen']:.2f}")
    
    print("\n  Jet Type Distribution:")
    for jet_type in ['quark', 'gluon', 'top']:
        real_frac = metrics[f'frac_{jet_type}_real']
        gen_frac = metrics[f'frac_{jet_type}_gen']
        print(f"    {jet_type.capitalize():6s} - Real: {real_frac:.3f}, Gen: {gen_frac:.3f}")


def main(args):
    # Load data
    real_data = load_jets_from_pt(args.real_data, max_jets=args.max_jets)
    generated_data = load_jets_from_pt(args.generated_data, max_jets=args.max_jets)
    
    print(f"Loaded {len(real_data['particle_features'])} real jets")
    print(f"Loaded {len(generated_data['particle_features'])} generated jets")
    
    # Compute Wasserstein distances
    w_distances = compute_wasserstein_distances(real_data, generated_data)
    
    # Compute structural metrics
    metrics = compute_structural_metrics(real_data, generated_data)
    
    # Print results
    print_evaluation_results(w_distances, metrics)
    
    # Plot distributions
    if args.plot:
        plot_feature_distributions(real_data, generated_data, output_dir=args.plot_dir)
    
    print("\nEvaluation completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate generated jets')
    parser.add_argument('--real-data', type=str, required=True, help='Path to real jet data .pt file')
    parser.add_argument('--generated-data', type=str, required=True, help='Path to generated jet data .pt file')
    parser.add_argument('--max-jets', type=int, default=10000, help='Maximum number of jets to evaluate')
    parser.add_argument('--plot', action='store_true', help='Generate distribution plots')
    parser.add_argument('--plot-dir', type=str, default='plots', help='Directory to save plots')
    
    args = parser.parse_args()
    main(args)
