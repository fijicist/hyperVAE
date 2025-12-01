import torch
import numpy as np
from scipy.stats import wasserstein_distance
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm


def load_jets_from_pt(file_path, max_jets=None):
    """Load jet data from .pt file (PyG Data format)"""
    print(f"Loading jets from {file_path}...")
    
    loaded_data = torch.load(file_path, weights_only=False)
    
    # Handle both old format (list) and new format (dict with 'graphs' key)
    if isinstance(loaded_data, dict) and 'graphs' in loaded_data:
        data_list = loaded_data['graphs']
    else:
        data_list = loaded_data
    
    n_jets = len(data_list) if max_jets is None else min(max_jets, len(data_list))
    
    particle_features = []
    edge_features = []
    hyperedge_features = []
    jet_types = []
    jet_features = []  # New: store jet features (jet_pt, jet_eta, jet_mass)
    n_particles = []
    n_edges = []
    n_hyperedges = []
    
    for i in range(n_jets):
        data = data_list[i]
        
        # Extract particle features (x)
        pf = data.x.cpu().numpy()
        if pf.shape[0] > 0:
            particle_features.append(pf)
            n_particles.append(pf.shape[0])
        
        # Extract edge features (edge_attr)
        if hasattr(data, 'edge_attr') and data.edge_attr is not None:
            ef = data.edge_attr.cpu().numpy()
            if ef.shape[0] > 0:
                edge_features.append(ef)
                n_edges.append(ef.shape[0])
        
        # Extract hyperedge features (hyperedge_attr)
        if hasattr(data, 'hyperedge_attr') and data.hyperedge_attr is not None:
            hf = data.hyperedge_attr.cpu().numpy()
            if hf.shape[0] > 0:
                hyperedge_features.append(hf)
                n_hyperedges.append(hf.shape[0])
        
        # Extract jet type and jet features from y
        if hasattr(data, 'y') and data.y is not None:
            y = data.y.cpu().numpy()
            if y.ndim == 0 or len(y) == 1:
                # Only jet type
                jet_types.append(float(y) if y.ndim == 0 else y[0])
                jet_features.append(np.array([]))
            else:
                # y = [jet_type, jet_pt, jet_eta, jet_mass, ...]
                jet_types.append(y[0])
                jet_features.append(y[1:])  # jet_pt, jet_eta, jet_mass, etc.
        else:
            jet_types.append(0)
            jet_features.append(np.array([]))
    
    return {
        'particle_features': particle_features,
        'edge_features': edge_features,
        'hyperedge_features': hyperedge_features,
        'jet_types': np.array(jet_types),
        'jet_features': jet_features,  # New: jet-level features
        'n_particles': np.array(n_particles),
        'n_edges': np.array(n_edges) if n_edges else np.array([]),
        'n_hyperedges': np.array(n_hyperedges) if n_hyperedges else np.array([])
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
    
    # Detect feature dimension
    num_particle_features = real_particles.shape[1]
    
    if num_particle_features == 3:
        particle_feature_names = ['pt', 'eta', 'phi']
    elif num_particle_features == 4:
        particle_feature_names = ['E', 'px', 'py', 'pz']
    else:
        particle_feature_names = [f'particle_feat_{i}' for i in range(num_particle_features)]
    
    for i, name in enumerate(particle_feature_names):
        real_feat = real_particles[:, i]
        gen_feat = gen_particles[:, i]
        
        # Remove NaN and Inf values only (keep normalized values including negatives)
        real_feat = real_feat[np.isfinite(real_feat)]
        gen_feat = gen_feat[np.isfinite(gen_feat)]
        
        if len(real_feat) > 0 and len(gen_feat) > 0:
            w_dist = wasserstein_distance(real_feat, gen_feat)
            w_distances[f'particle_{name}'] = w_dist
    
    # Wasserstein distance for number of particles
    w_distances['n_particles'] = wasserstein_distance(
        real_data['n_particles'], 
        generated_data['n_particles']
    )
    
    # Wasserstein distances for jet features (jet_pt, jet_eta, jet_mass)
    if len(real_data['jet_features']) > 0 and len(generated_data['jet_features']) > 0:
        # Filter out empty arrays and concatenate
        real_jet_feats = [jf for jf in real_data['jet_features'] if len(jf) > 0]
        gen_jet_feats = [jf for jf in generated_data['jet_features'] if len(jf) > 0]
        
        if len(real_jet_feats) > 0 and len(gen_jet_feats) > 0:
            real_jet_array = np.array(real_jet_feats)
            gen_jet_array = np.array(gen_jet_feats)
            
            num_jet_features = real_jet_array.shape[1]
            jet_feature_names = ['jet_pt', 'jet_eta', 'jet_mass'][:num_jet_features]
            
            for i, name in enumerate(jet_feature_names):
                if i < real_jet_array.shape[1] and i < gen_jet_array.shape[1]:
                    real_feat = real_jet_array[:, i]
                    gen_feat = gen_jet_array[:, i]
                    
                    # Remove NaN and Inf values
                    real_feat = real_feat[np.isfinite(real_feat)]
                    gen_feat = gen_feat[np.isfinite(gen_feat)]
                    
                    if len(real_feat) > 0 and len(gen_feat) > 0:
                        w_dist = wasserstein_distance(real_feat, gen_feat)
                        w_distances[name] = w_dist
    
    # Wasserstein distances for edge features (ln_delta, ln_kt, ln_z, ln_m2, 2pt_eec)
    if len(real_data['edge_features']) > 0 and len(generated_data['edge_features']) > 0:
        real_edges = np.concatenate(real_data['edge_features'], axis=0)
        gen_edges = np.concatenate(generated_data['edge_features'], axis=0)
        edge_feature_names = ['ln_delta', 'ln_kt', 'ln_z', 'ln_m2', '2pt_eec']
        
        for i, name in enumerate(edge_feature_names):
            if i < real_edges.shape[1] and i < gen_edges.shape[1]:
                real_feat = real_edges[:, i]
                gen_feat = gen_edges[:, i]
                
                # Remove NaN and Inf values
                real_feat = real_feat[np.isfinite(real_feat)]
                gen_feat = gen_feat[np.isfinite(gen_feat)]
                
                if len(real_feat) > 0 and len(gen_feat) > 0:
                    w_dist = wasserstein_distance(real_feat, gen_feat)
                    w_distances[f'edge_{name}'] = w_dist
        
        # Wasserstein distance for number of edges
        if len(real_data['n_edges']) > 0 and len(generated_data['n_edges']) > 0:
            w_distances['n_edges'] = wasserstein_distance(
                real_data['n_edges'], 
                generated_data['n_edges']
            )
    
    # Wasserstein distances for hyperedge features (3pt_eec, 4pt_eec)
    if len(real_data['hyperedge_features']) > 0 and len(generated_data['hyperedge_features']) > 0:
        real_hyperedges = np.concatenate(real_data['hyperedge_features'], axis=0)
        gen_hyperedges = np.concatenate(generated_data['hyperedge_features'], axis=0)
        
        hyperedge_feature_names = ['3pt_eec', '4pt_eec']
        
        for i, name in enumerate(hyperedge_feature_names):
            if i < real_hyperedges.shape[1] and i < gen_hyperedges.shape[1]:
                real_feat = real_hyperedges[:, i]
                gen_feat = gen_hyperedges[:, i]
                
                # Remove NaN and Inf values
                real_feat = real_feat[np.isfinite(real_feat)]
                gen_feat = gen_feat[np.isfinite(gen_feat)]
                
                if len(real_feat) > 0 and len(gen_feat) > 0:
                    w_dist = wasserstein_distance(real_feat, gen_feat)
                    w_distances[f'hyperedge_{name}'] = w_dist
        
        # Wasserstein distance for number of hyperedges
        if len(real_data['n_hyperedges']) > 0 and len(generated_data['n_hyperedges']) > 0:
            w_distances['n_hyperedges'] = wasserstein_distance(
                real_data['n_hyperedges'], 
                generated_data['n_hyperedges']
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
    
    # ============================================================
    # Particle Features
    # ============================================================
    real_particles = np.concatenate(real_data['particle_features'], axis=0)
    gen_particles = np.concatenate(generated_data['particle_features'], axis=0)
    
    # Detect feature dimension
    num_particle_features = real_particles.shape[1]
    
    if num_particle_features == 3:
        particle_feature_names = ['pt', 'eta', 'phi']
    elif num_particle_features == 4:
        particle_feature_names = ['E', 'px', 'py', 'pz']
    else:
        particle_feature_names = [f'feat_{i}' for i in range(num_particle_features)]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for i, name in enumerate(particle_feature_names):
        if i >= len(axes):
            break
        ax = axes[i]
        
        real_feat = real_particles[:, i]
        gen_feat = gen_particles[:, i]
        
        # Remove NaN and Inf values only (keep normalized values including negatives)
        real_feat = real_feat[np.isfinite(real_feat)]
        gen_feat = gen_feat[np.isfinite(gen_feat)]
        
        # Compute shared range for better comparison
        # Use percentiles to exclude outliers and focus on main distribution
        all_feat = np.concatenate([real_feat, gen_feat])
        vmin, vmax = np.percentile(all_feat, [1, 99])
        
        # Plot histograms with shared bins
        bins = np.linspace(vmin, vmax, 50)
        ax.hist(real_feat, bins=bins, alpha=0.5, label='Real', density=True, color='blue')
        ax.hist(gen_feat, bins=bins, alpha=0.5, label='Generated', density=True, color='orange')
        
        ax.set_xlabel(name)
        ax.set_ylabel('Density')
        ax.set_title(f'{name.capitalize()} Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Hide the 4th subplot if we only have 3 or 4 particle features
    if len(particle_feature_names) < 4:
        axes[3].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/particle_features.png', dpi=150)
    plt.close()
    
    # ============================================================
    # Jet Features (jet_pt, jet_eta, jet_mass, n_particles)
    # ============================================================
    if len(real_data['jet_features']) > 0 and len(generated_data['jet_features']) > 0:
        # Filter out empty arrays
        real_jet_feats = [jf for jf in real_data['jet_features'] if len(jf) > 0]
        gen_jet_feats = [jf for jf in generated_data['jet_features'] if len(jf) > 0]
        
        if len(real_jet_feats) > 0 and len(gen_jet_feats) > 0:
            real_jet_array = np.array(real_jet_feats)
            gen_jet_array = np.array(gen_jet_feats)
            
            num_jet_features = min(real_jet_array.shape[1], gen_jet_array.shape[1])
            jet_feature_names = ['jet_pt', 'jet_eta', 'jet_mass'][:num_jet_features]
            
            # Create 2x2 subplot: 3 jet features + n_particles
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            axes = axes.flatten()
            
            for i, name in enumerate(jet_feature_names):
                if i >= len(axes) - 1:  # Reserve last subplot for n_particles
                    break
                ax = axes[i]
                
                real_feat = real_jet_array[:, i]
                gen_feat = gen_jet_array[:, i]
                
                # Remove NaN and Inf values
                real_feat = real_feat[np.isfinite(real_feat)]
                gen_feat = gen_feat[np.isfinite(gen_feat)]
                
                if len(real_feat) > 0 and len(gen_feat) > 0:
                    # Compute shared range using percentiles
                    all_feat = np.concatenate([real_feat, gen_feat])
                    vmin, vmax = np.percentile(all_feat, [1, 99])
                    bins = np.linspace(vmin, vmax, 50)
                    
                    ax.hist(real_feat, bins=bins, alpha=0.5, label='Real', density=True, color='blue')
                    ax.hist(gen_feat, bins=bins, alpha=0.5, label='Generated', density=True, color='orange')
                    
                    ax.set_xlabel(name)
                    ax.set_ylabel('Density')
                    ax.set_title(f'{name} Distribution')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
            
            # Plot number of particles in the 4th subplot
            ax = axes[3]
            bins = np.arange(0, max(real_data['n_particles'].max(), generated_data['n_particles'].max()) + 2)
            ax.hist(real_data['n_particles'], bins=bins, alpha=0.5, label='Real', density=True, color='blue')
            ax.hist(generated_data['n_particles'], bins=bins, alpha=0.5, label='Generated', density=True, color='orange')
            ax.set_xlabel('Number of Particles')
            ax.set_ylabel('Density')
            ax.set_title('Number of Particles per Jet')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f'{output_dir}/jet_features.png', dpi=150)
            plt.close()
    
    # ============================================================
    # Edge Features (ln_delta, ln_kt, ln_z, ln_m2, 2pt_eec)
    # ============================================================
    if len(real_data['edge_features']) > 0 and len(generated_data['edge_features']) > 0:
        real_edges = np.concatenate(real_data['edge_features'], axis=0)
        gen_edges = np.concatenate(generated_data['edge_features'], axis=0)
        
        edge_feature_names = ['ln_delta', 'ln_kt', 'ln_z', 'ln_m2', '2pt_eec']
        n_edge_features = min(real_edges.shape[1], gen_edges.shape[1], len(edge_feature_names))
        
        # Create subplots for edge features
        n_cols = 3
        n_rows = (n_edge_features + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        axes = axes.flatten()
        
        for i in range(n_edge_features):
            ax = axes[i]
            name = edge_feature_names[i]
            
            real_feat = real_edges[:, i]
            gen_feat = gen_edges[:, i]
            
            # Remove NaN and Inf values
            real_feat = real_feat[np.isfinite(real_feat)]
            gen_feat = gen_feat[np.isfinite(gen_feat)]
            
            if len(real_feat) > 0 and len(gen_feat) > 0:
                # Compute shared range using percentiles
                all_feat = np.concatenate([real_feat, gen_feat])
                vmin, vmax = np.percentile(all_feat, [1, 99])
                bins = np.linspace(vmin, vmax, 50)
                
                ax.hist(real_feat, bins=bins, alpha=0.5, label='Real', density=True, color='blue')
                ax.hist(gen_feat, bins=bins, alpha=0.5, label='Generated', density=True, color='orange')
                
                ax.set_xlabel(name)
                ax.set_ylabel('Density')
                ax.set_title(f'{name} Distribution')
                ax.legend()
                ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(n_edge_features, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/edge_features.png', dpi=150)
        plt.close()
        
        # Plot number of edges
        if len(real_data['n_edges']) > 0 and len(generated_data['n_edges']) > 0:
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            max_edges = max(real_data['n_edges'].max(), generated_data['n_edges'].max())
            bins = np.linspace(0, max_edges, 50)
            ax.hist(real_data['n_edges'], bins=bins, alpha=0.5, label='Real', density=True, color='blue')
            ax.hist(generated_data['n_edges'], bins=bins, alpha=0.5, label='Generated', density=True, color='orange')
            ax.set_xlabel('Number of Edges')
            ax.set_ylabel('Density')
            ax.set_title('Number of Edges per Jet')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f'{output_dir}/n_edges.png', dpi=150)
            plt.close()
    
    # ============================================================
    # Hyperedge Features (3pt_eec, 4pt_eec)
    # ============================================================
    if len(real_data['hyperedge_features']) > 0 and len(generated_data['hyperedge_features']) > 0:
        real_hyperedges = np.concatenate(real_data['hyperedge_features'], axis=0)
        gen_hyperedges = np.concatenate(generated_data['hyperedge_features'], axis=0)
        
        hyperedge_feature_names = ['3pt_eec', '4pt_eec']
        n_hyperedge_features = min(real_hyperedges.shape[1], gen_hyperedges.shape[1], len(hyperedge_feature_names))
        
        # Create subplots for hyperedge features
        fig, axes = plt.subplots(1, n_hyperedge_features, figsize=(8 * n_hyperedge_features, 6))
        if n_hyperedge_features == 1:
            axes = [axes]
        
        for i in range(n_hyperedge_features):
            ax = axes[i]
            name = hyperedge_feature_names[i]
            
            real_feat = real_hyperedges[:, i]
            gen_feat = gen_hyperedges[:, i]
            
            # Remove NaN and Inf values
            real_feat = real_feat[np.isfinite(real_feat)]
            gen_feat = gen_feat[np.isfinite(gen_feat)]
            
            if len(real_feat) > 0 and len(gen_feat) > 0:
                # Compute shared range using percentiles
                all_feat = np.concatenate([real_feat, gen_feat])
                vmin, vmax = np.percentile(all_feat, [1, 99])
                bins = np.linspace(vmin, vmax, 50)
                
                ax.hist(real_feat, bins=bins, alpha=0.5, label='Real', density=True, color='blue')
                ax.hist(gen_feat, bins=bins, alpha=0.5, label='Generated', density=True, color='orange')
                
                ax.set_xlabel(name)
                ax.set_ylabel('Density')
                ax.set_title(f'{name} Distribution')
                ax.legend()
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/hyperedge_features.png', dpi=150)
        plt.close()
        
        # Plot number of hyperedges
        if len(real_data['n_hyperedges']) > 0 and len(generated_data['n_hyperedges']) > 0:
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            max_hyperedges = max(real_data['n_hyperedges'].max(), generated_data['n_hyperedges'].max())
            bins = np.linspace(0, max_hyperedges, 50)
            ax.hist(real_data['n_hyperedges'], bins=bins, alpha=0.5, label='Real', density=True, color='blue')
            ax.hist(generated_data['n_hyperedges'], bins=bins, alpha=0.5, label='Generated', density=True, color='orange')
            ax.set_xlabel('Number of Hyperedges')
            ax.set_ylabel('Density')
            ax.set_title('Number of Hyperedges per Jet')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f'{output_dir}/n_hyperedges.png', dpi=150)
            plt.close()
    
    print(f"Saved plots to {output_dir}/")


def print_evaluation_results(w_distances, metrics):
    """Print evaluation results"""
    print("\n" + "="*60)
    print("Evaluation Results")
    print("="*60)
    
    print("\nWasserstein Distances:")
    print("-" * 40)
    
    # Particle features - detect if 3D or 4D
    particle_keys = [k for k in w_distances.keys() if k.startswith('particle_')]
    if particle_keys:
        print("  Particle Features:")
        for key in sorted(particle_keys):
            print(f"    {key:20s}: {w_distances[key]:.6f}")
    
    # Jet features
    jet_keys = ['jet_pt', 'jet_eta', 'jet_mass']
    jet_found = [k for k in jet_keys if k in w_distances]
    if jet_found:
        print("\n  Jet Features:")
        for key in jet_found:
            print(f"    {key:20s}: {w_distances[key]:.6f}")
    
    # Number of particles
    if 'n_particles' in w_distances:
        print(f"\n  {'n_particles':20s}: {w_distances['n_particles']:.6f}")
    
    # Edge features
    edge_keys = [k for k in w_distances.keys() if k.startswith('edge_')]
    if edge_keys:
        print("\n  Edge Features:")
        for key in edge_keys:
            print(f"    {key:20s}: {w_distances[key]:.6f}")
        if 'n_edges' in w_distances:
            print(f"    {'n_edges':20s}: {w_distances['n_edges']:.6f}")
    
    # Hyperedge features
    hyperedge_keys = [k for k in w_distances.keys() if k.startswith('hyperedge_')]
    if hyperedge_keys:
        print("\n  Hyperedge Features:")
        for key in hyperedge_keys:
            print(f"    {key:20s}: {w_distances[key]:.6f}")
        if 'n_hyperedges' in w_distances:
            print(f"    {'n_hyperedges':20s}: {w_distances['n_hyperedges']:.6f}")
    
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
