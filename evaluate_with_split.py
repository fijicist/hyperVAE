#!/usr/bin/env python3
"""
Evaluation script that uses the saved test split from training.
"""

import torch
import argparse
from data.bipartite_dataset import BipartiteJetDataset
from torch.utils.data import Subset

# Import evaluation functions from evaluate.py
from evaluate import load_jets_from_pt, compute_wasserstein_distances, compute_structural_metrics, plot_feature_distributions, print_evaluation_results


def load_test_split(data_path, test_indices_path):
    """Load test split using saved indices"""
    print(f"Loading full dataset from: {data_path}")
    full_dataset = BipartiteJetDataset(data_path=data_path)
    
    print(f"Loading test indices from: {test_indices_path}")
    test_indices = torch.load(test_indices_path)
    
    test_dataset = Subset(full_dataset, test_indices)
    print(f"Test set size: {len(test_dataset)} jets")
    
    return test_dataset


def dataset_to_dict(dataset):
    """Convert dataset/subset to dictionary format for evaluation"""
    particle_features = []
    jet_types = []
    n_particles = []
    
    for i in range(len(dataset)):
        data = dataset[i]
        
        # Extract particle features
        pf = data.particle_x.cpu().numpy()
        if pf.shape[0] > 0:
            particle_features.append(pf)
            n_particles.append(pf.shape[0])
        
        # Extract jet type
        if hasattr(data, 'jet_type') and data.jet_type is not None:
            jet_type = data.jet_type.cpu().numpy()
            if jet_type.ndim > 0:
                jet_type = jet_type[0]
            jet_types.append(jet_type)
        else:
            jet_types.append(0)
    
    import numpy as np
    return {
        'particle_features': particle_features,
        'jet_types': np.array(jet_types),
        'n_particles': np.array(n_particles)
    }


def main(args):
    # Load test split
    test_dataset = load_test_split(args.data_path, args.test_indices)
    
    # Load generated data
    print(f"\nLoading generated jets from: {args.generated_data}")
    generated_data = load_jets_from_pt(args.generated_data, max_jets=args.max_jets)
    
    # Convert test dataset to dict format
    print("Converting test dataset...")
    real_data = dataset_to_dict(test_dataset)
    
    print(f"\nLoaded {len(real_data['particle_features'])} real jets (test set)")
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
    parser = argparse.ArgumentParser(description='Evaluate generated jets against test split')
    parser.add_argument('--data-path', type=str, required=True, help='Path to original data .pt file')
    parser.add_argument('--test-indices', type=str, required=True, help='Path to test_indices.pt file')
    parser.add_argument('--generated-data', type=str, required=True, help='Path to generated jets .pt file')
    parser.add_argument('--max-jets', type=int, default=10000, help='Maximum number of jets to evaluate')
    parser.add_argument('--plot', action='store_true', help='Generate distribution plots')
    parser.add_argument('--plot-dir', type=str, default='plots', help='Directory to save plots')
    
    args = parser.parse_args()
    main(args)
