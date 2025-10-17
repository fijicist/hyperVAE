#!/usr/bin/env python3
"""
Script to validate your .pt dataset file format.
"""

import torch
import argparse
from torch_geometric.data import Data


def validate_data_file(file_path):
    """Validate that the .pt file is in correct format"""
    print(f"Validating data file: {file_path}")
    print("="*60)
    
    try:
        # Load data
        print("\n1. Loading data...")
        data_list = torch.load(file_path)
        print(f"   ✓ Loaded successfully")
        
        # Check if it's a list
        if not isinstance(data_list, list):
            print(f"   ✗ ERROR: Expected list, got {type(data_list)}")
            return False
        
        print(f"   ✓ Data is a list with {len(data_list)} jets")
        
        # Check first few samples
        print("\n2. Checking data structure...")
        n_samples = min(5, len(data_list))
        
        for i in range(n_samples):
            print(f"\n   Jet {i}:")
            data = data_list[i]
            
            if not isinstance(data, Data):
                print(f"      ✗ ERROR: Expected PyG Data object, got {type(data)}")
                return False
            
            # Check required fields
            required_fields = ['x', 'edge_index', 'edge_attr', 'y']
            optional_fields = ['hyperedge_index', 'hyperedge_attr']
            
            # x (particle features)
            if hasattr(data, 'x') and data.x is not None:
                print(f"      ✓ x (particles): {data.x.shape}")
                if data.x.shape[1] not in [3, 4]:
                    print(f"         ⚠ Warning: Expected 3 features (pt,eta,phi), got {data.x.shape[1]}")
                elif data.x.shape[1] == 4:
                    print(f"         ℹ Info: Has 4 features - will use first 3 (pt,eta,phi)")
            else:
                print(f"      ✗ ERROR: Missing 'x' (particle features)")
                return False
            
            # edge_index
            if hasattr(data, 'edge_index') and data.edge_index is not None:
                print(f"      ✓ edge_index: {data.edge_index.shape}")
                if data.edge_index.shape[0] != 2:
                    print(f"         ⚠ Warning: Expected shape [2, N], got {data.edge_index.shape}")
            else:
                print(f"      ⚠ Warning: Missing 'edge_index'")
            
            # edge_attr
            if hasattr(data, 'edge_attr') and data.edge_attr is not None:
                print(f"      ✓ edge_attr: {data.edge_attr.shape}")
                if data.edge_attr.shape[1] != 5:
                    print(f"         ⚠ Warning: Expected 5 features, got {data.edge_attr.shape[1]}")
            else:
                print(f"      ⚠ Warning: Missing 'edge_attr'")
            
            # hyperedge_index
            if hasattr(data, 'hyperedge_index') and data.hyperedge_index is not None:
                print(f"      ✓ hyperedge_index: {data.hyperedge_index.shape}")
            else:
                print(f"      ⚠ Warning: Missing 'hyperedge_index'")
            
            # hyperedge_attr
            if hasattr(data, 'hyperedge_attr') and data.hyperedge_attr is not None:
                print(f"      ✓ hyperedge_attr: {data.hyperedge_attr.shape}")
                if data.hyperedge_attr.shape[1] != 2:
                    print(f"         ⚠ Warning: Expected 2 features (3pt_eec, 4pt_eec), got {data.hyperedge_attr.shape[1]}")
            else:
                print(f"      ⚠ Warning: Missing 'hyperedge_attr'")
            
            # y (label)
            if hasattr(data, 'y') and data.y is not None:
                print(f"      ✓ y (jet type): {data.y}")
                jet_type = data.y.item() if data.y.numel() == 1 else data.y[0].item()
                if jet_type not in [0, 1, 2]:
                    print(f"         ⚠ Warning: Expected jet type in [0,1,2], got {jet_type}")
            else:
                print(f"      ⚠ Warning: Missing 'y' (jet type)")
        
        # Statistics
        print("\n3. Dataset statistics:")
        n_particles_list = [data.x.shape[0] for data in data_list]
        n_edges_list = [data.edge_index.shape[1] if hasattr(data, 'edge_index') and data.edge_index is not None else 0 
                       for data in data_list]
        n_hyperedges_list = [data.hyperedge_attr.shape[0] if hasattr(data, 'hyperedge_attr') and data.hyperedge_attr is not None else 0 
                            for data in data_list]
        
        print(f"   Particles per jet:")
        print(f"      Mean: {sum(n_particles_list) / len(n_particles_list):.1f}")
        print(f"      Min:  {min(n_particles_list)}")
        print(f"      Max:  {max(n_particles_list)}")
        
        print(f"   Edges per jet:")
        print(f"      Mean: {sum(n_edges_list) / len(n_edges_list):.1f}")
        print(f"      Min:  {min(n_edges_list)}")
        print(f"      Max:  {max(n_edges_list)}")
        
        print(f"   Hyperedges per jet:")
        print(f"      Mean: {sum(n_hyperedges_list) / len(n_hyperedges_list):.1f}")
        print(f"      Min:  {min(n_hyperedges_list)}")
        print(f"      Max:  {max(n_hyperedges_list)}")
        
        # Jet types
        if all(hasattr(data, 'y') and data.y is not None for data in data_list):
            jet_types = [data.y.item() if data.y.numel() == 1 else data.y[0].item() for data in data_list]
            print(f"\n   Jet type distribution:")
            for jt in [0, 1, 2]:
                count = jet_types.count(jt)
                frac = count / len(jet_types)
                label = ['Quark', 'Gluon', 'Top'][jt]
                print(f"      {label} ({jt}): {count} ({frac*100:.1f}%)")
        
        print("\n" + "="*60)
        print("✓ Data validation passed!")
        print("="*60)
        print("\nYour data is ready for training!")
        print(f"Run: python train.py --data-path {file_path}")
        
        return True
        
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Validate .pt dataset file')
    parser.add_argument('data_path', type=str, help='Path to .pt data file')
    
    args = parser.parse_args()
    
    success = validate_data_file(args.data_path)
    exit(0 if success else 1)
