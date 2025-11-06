#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
DATA VALIDATION SCRIPT - VERIFY DATASET FORMAT FOR HYPERVAE
═══════════════════════════════════════════════════════════════════════════════

Purpose:
--------
Validate that your .pt dataset file is in the correct PyTorch Geometric format
for training HyperVAE. Checks data structure, feature dimensions, and provides
statistics about your dataset.

Usage:
------
    python validate_data.py data/real/jets.pt

Expected Data Format:
---------------------
The .pt file should contain a list of PyG Data objects, where each Data object
represents one jet with the following attributes:

Required Fields:
  - x: [N_particles, 4] - Particle 4-momenta [E, px, py, pz] (normalized)
  - edge_index: [2, N_edges] - Pairwise particle connections (fully-connected)
  - edge_attr: [N_edges, 5] - Edge features [2pt_EEC, ln_delta, ln_kT, ln_z, ln_m²]
  - y: [4] - Jet-level features [jet_type, log(pt), eta, log(mass)]

Optional Fields (for hypergraph structure):
  - hyperedge_index: [N_particles, N_hyperedges] - Particle-hyperedge incidence matrix
  - hyperedge_attr: [N_hyperedges, features] - N-point EEC features

Normalization Statistics (for denormalization):
  - particle_norm_stats: dict - Statistics for E, px, py, pz
  - jet_norm_stats: dict - Statistics for jet pt, eta, mass
  - edge_norm_stats: dict - Statistics for edge features
  - hyperedge_norm_stats: dict - Statistics for hyperedge features

Validation Checks:
------------------
1. File format (list of PyG Data objects)
2. Required fields present
3. Feature dimensions correct
4. Jet type values valid (0, 1, 2)
5. Normalized 4-momenta (E, px, py, pz)
6. Dataset statistics and distribution

Output:
-------
Prints detailed validation report with:
- Per-jet structure verification
- Dataset statistics (particles, edges, hyperedges per jet)
- Jet type distribution
- Feature dimension checks
- Warnings for potential issues

═══════════════════════════════════════════════════════════════════════════════
"""

import torch
import argparse
import numpy as np
from torch_geometric.data import Data


def validate_data_file(file_path):
    """
    Validate that the .pt file is in the correct format for HyperVAE training.
    
    Args:
        file_path: Path to .pt file containing list of PyG Data objects
    
    Returns:
        bool: True if validation passes, False otherwise
    """
    print("═══════════════════════════════════════════════════════════════")
    print(f"Validating: {file_path}")
    print("═══════════════════════════════════════════════════════════════")
    
    try:
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # STEP 1: Load Data
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        print("\n[1/5] Loading data file...")
        data_list = torch.load(file_path)
        print(f"      ✓ File loaded successfully")
        
        # Check if it's a list
        if not isinstance(data_list, list):
            print(f"      ✗ ERROR: Expected list of Data objects, got {type(data_list)}")
            print(f"         The file should contain: list of PyG Data objects")
            return False
        
        print(f"      ✓ Contains {len(data_list)} jets")
        
        if len(data_list) == 0:
            print(f"      ✗ ERROR: Empty dataset")
            return False
        
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # STEP 2: Check Data Structure
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        print("\n[2/5] Checking data structure...")
        n_samples = min(3, len(data_list))
        errors = []
        warnings = []
        
        for i in range(n_samples):
            print(f"\n      Jet {i}:")
            data = data_list[i]
            
            if not isinstance(data, Data):
                print(f"         ✗ ERROR: Expected PyG Data object, got {type(data)}")
                errors.append(f"Jet {i}: Wrong type {type(data)}")
                continue
            
            # ─────────────────────────────────────────────────────────────
            # Check x (4-momenta) - REQUIRED
            # ─────────────────────────────────────────────────────────────
            if hasattr(data, 'x') and data.x is not None:
                print(f"         ✓ x (4-momenta): {data.x.shape}")
                
                # Check if it's 4D (E, px, py, pz)
                if data.x.shape[1] == 4:
                    print(f"            ✓ Correct format: [N, 4] for [E, px, py, pz]")
                else:
                    warnings.append(f"Jet {i}: x has {data.x.shape[1]} features (expected 4)")
                    print(f"            ⚠ Warning: Has {data.x.shape[1]} features (expected 4)")
                
                # Check for normalization (values should be roughly in [-3, 3] for Z-score)
                if data.x.shape[0] > 0:
                    pmin, pmax = data.x.min().item(), data.x.max().item()
                    if pmin < -10 or pmax > 10:
                        warnings.append(f"Jet {i}: x may not be normalized (range [{pmin:.2f}, {pmax:.2f}])")
                        print(f"            ⚠ Values may not be normalized: [{pmin:.2f}, {pmax:.2f}]")
                    else:
                        print(f"            ✓ Values look normalized: [{pmin:.2f}, {pmax:.2f}]")
            else:
                print(f"         ✗ ERROR: Missing 'x' (particle 4-momenta)")
                errors.append(f"Jet {i}: Missing x")
            
            # ─────────────────────────────────────────────────────────────
            # Check normalization statistics - REQUIRED for denormalization
            # ─────────────────────────────────────────────────────────────
            if hasattr(data, 'particle_norm_stats'):
                print(f"         ✓ particle_norm_stats: {list(data.particle_norm_stats.keys())}")
            else:
                warnings.append(f"Jet {i}: Missing particle_norm_stats (needed for denormalization)")
                print(f"         ⚠ Missing 'particle_norm_stats' (needed for generation)")
            
            if hasattr(data, 'jet_norm_stats'):
                print(f"         ✓ jet_norm_stats: {list(data.jet_norm_stats.keys())}")
            else:
                warnings.append(f"Jet {i}: Missing jet_norm_stats")
                print(f"         ⚠ Missing 'jet_norm_stats'")
            
            if hasattr(data, 'edge_norm_stats'):
                print(f"         ✓ edge_norm_stats: {list(data.edge_norm_stats.keys())}")
            else:
                warnings.append(f"Jet {i}: Missing edge_norm_stats")
                print(f"         ⚠ Missing 'edge_norm_stats'")

            
            # ─────────────────────────────────────────────────────────────
            # Check y (jet-level features) - REQUIRED
            # ─────────────────────────────────────────────────────────────
            if hasattr(data, 'y') and data.y is not None:
                y_shape = data.y.shape if hasattr(data.y, 'shape') else (len(data.y),)
                print(f"         ✓ y (jet features): {y_shape}")
                
                # Check jet_type (first element should be 0, 1, or 2)
                if y_shape[0] >= 1:
                    jet_type = data.y[0].item() if hasattr(data.y[0], 'item') else data.y[0]
                    jet_type_int = int(jet_type)
                    type_names = {0: 'Quark', 1: 'Gluon', 2: 'Top'}
                    type_name = type_names.get(jet_type_int, 'Unknown')
                    print(f"         ✓ jet_type (y[0]): {jet_type_int} ({type_name})")
                    if jet_type_int not in [0, 1, 2]:
                        warnings.append(f"Jet {i}: y[0] (jet_type) = {jet_type_int} not in [0,1,2]")
                        print(f"            ⚠ Invalid jet_type (expected 0, 1, or 2)")
                
                # Expected: [jet_type, log(pt), eta, log(mass)]
                if y_shape[0] == 4:
                    print(f"            ✓ Format: [jet_type, log(pt), eta, log(mass)]")
                else:
                    warnings.append(f"Jet {i}: y has {y_shape[0]} features (expected 4)")
                    print(f"            ⚠ Has {y_shape[0]} features (expected 4)")
            else:
                print(f"         ✗ ERROR: Missing 'y' (jet-level features)")
                errors.append(f"Jet {i}: Missing y tensor")

            
            # ─────────────────────────────────────────────────────────────
            # Check optional graph structure fields
            # ─────────────────────────────────────────────────────────────
            if hasattr(data, 'edge_index') and data.edge_index is not None:
                print(f"         ○ edge_index (optional): {data.edge_index.shape}")
                if data.edge_index.shape[0] != 2:
                    warnings.append(f"Jet {i}: edge_index shape {data.edge_index.shape} (expected [2, N])")
            else:
                print(f"         ○ edge_index: Not present (optional)")
            
            if hasattr(data, 'edge_attr') and data.edge_attr is not None:
                print(f"         ○ edge_attr (optional): {data.edge_attr.shape}")
            else:
                print(f"         ○ edge_attr: Not present (optional)")
            
            if hasattr(data, 'hyperedge_index') and data.hyperedge_index is not None:
                print(f"         ○ hyperedge_index (optional): {data.hyperedge_index.shape}")
            else:
                print(f"         ○ hyperedge_index: Not present (optional)")
            
            if hasattr(data, 'hyperedge_attr') and data.hyperedge_attr is not None:
                print(f"         ○ hyperedge_attr (optional): {data.hyperedge_attr.shape}")
            else:
                print(f"         ○ hyperedge features: Not present (optional)")

        
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # STEP 3: Dataset Statistics
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        print("\n[3/5] Computing dataset statistics...")
        
        # Gather statistics
        n_particles_list = []
        n_edges_list = []
        n_hyperedges_list = []
        
        for data in data_list:
            # Particles
            if hasattr(data, 'x') and data.x is not None:
                n_particles_list.append(data.x.shape[0])
            else:
                n_particles_list.append(0)
            
            # Edges
            if hasattr(data, 'edge_index') and data.edge_index is not None:
                n_edges_list.append(data.edge_index.shape[1])
            else:
                n_edges_list.append(0)
            
            # Hyperedges
            if hasattr(data, 'hyperedge_attr') and data.hyperedge_attr is not None:
                n_hyperedges_list.append(data.hyperedge_attr.shape[0])
            else:
                n_hyperedges_list.append(0)
        
        print(f"\n      Particles per jet:")
        print(f"         Mean:   {np.mean(n_particles_list):.1f}")
        print(f"         Median: {np.median(n_particles_list):.1f}")
        print(f"         Min:    {np.min(n_particles_list)}")
        print(f"         Max:    {np.max(n_particles_list)}")
        
        if any(n > 0 for n in n_edges_list):
            print(f"\n      Edges per jet:")
            print(f"         Mean: {np.mean(n_edges_list):.1f}")
            print(f"         Max:  {np.max(n_edges_list)}")
        
        if any(n > 0 for n in n_hyperedges_list):
            print(f"\n      Hyperedges per jet:")
            print(f"         Mean: {np.mean(n_hyperedges_list):.1f}")
            print(f"         Max:  {np.max(n_hyperedges_list)}")
        
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # STEP 4: Jet Type Distribution
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        print("\n[4/5] Checking jet type distribution...")
        
        jet_types = []
        for data in data_list:
            if hasattr(data, 'y') and data.y is not None:
                jt = data.y[0].item() if hasattr(data.y[0], 'item') else data.y[0]
                jet_types.append(int(jt))
        
        if jet_types:
            print(f"\n      Jet type distribution:")
            for jt in [0, 1, 2]:
                count = jet_types.count(jt)
                frac = count / len(jet_types) if jet_types else 0
                label = ['Quark', 'Gluon', 'Top'][jt]
                print(f"         {label:6s} ({jt}): {count:6d} ({frac*100:5.1f}%)")
        
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # STEP 5: Final Report
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        print("\n[5/5] Validation summary...")
        
        print(f"\n      Errors:   {len(errors)}")
        for err in errors:
            print(f"         ✗ {err}")
        
        print(f"\n      Warnings: {len(warnings)}")
        for warn in warnings[:10]:  # Show first 10
            print(f"         ⚠ {warn}")
        if len(warnings) > 10:
            print(f"         ... and {len(warnings) - 10} more warnings")
        
        print("\n═══════════════════════════════════════════════════════════════")
        
        if errors:
            print("✗ VALIDATION FAILED")
            print("═══════════════════════════════════════════════════════════════")
            print("\nPlease fix the errors above before training.")
            print("\nRequired format:")
            print("  - List of PyG Data objects")
            print("  - Each Data must have: particle_x [N, 4], n_particles, y, jet_type")
            return False
        else:
            print("✓ VALIDATION PASSED")
            print("═══════════════════════════════════════════════════════════════")
            
            if warnings:
                print(f"\n⚠ Note: {len(warnings)} warnings (non-critical)")
                print("  Your data should work, but review warnings above.")
            
            print(f"\nYour dataset is ready for training!")
            print(f"Total jets: {len(data_list)}")
            print(f"Average particles/jet: {np.mean(n_particles_list):.1f}")
            print(f"\nNext steps:")
            print(f"  1. Train: python train.py --config config.yaml --data-path {file_path}")
            print(f"  2. Monitor: tensorboard --logdir runs")
            return True
        
    except Exception as e:
        print(f"\n✗ EXCEPTION: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Validate .pt dataset file')
    parser.add_argument('data_path', type=str, help='Path to .pt data file')
    
    args = parser.parse_args()
    
    success = validate_data_file(args.data_path)
    exit(0 if success else 1)
