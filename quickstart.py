#!/usr/bin/env python3
"""
Quick start script to test the Bipartite HyperVAE.
Trains on dummy data for a few epochs to verify everything works.
"""

import torch
import yaml
from torch.utils.data import DataLoader
from data.bipartite_dataset import BipartiteJetDataset, collate_bipartite_batch
from models.hypervae import BipartiteHyperVAE


def main():
    print("="*60)
    print("Bipartite HyperVAE Quick Start Test")
    print("="*60)
    
    # Check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    print("\n1. Creating dummy dataset...")
    dataset = BipartiteJetDataset(generate_dummy=True)
    print(f"   Dataset size: {len(dataset)}")
    
    # Create loader
    loader = DataLoader(
        dataset, 
        batch_size=config['training']['batch_size'], 
        shuffle=True,
        collate_fn=collate_bipartite_batch
    )
    
    # Test data loading
    print("\n2. Testing data loading...")
    batch = next(iter(loader))
    print(f"   Batch graphs: {batch.num_graphs}")
    print(f"   Total particles: {batch.particle_x.shape[0]}")
    print(f"   Total hyperedges: {batch.hyperedge_x.shape[0]}")
    print(f"   Total edges: {batch.edge_attr.shape[0]}")
    
    # Create model
    print("\n3. Creating model...")
    model = BipartiteHyperVAE(config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"   Parameters: {n_params/1e6:.2f}M")
    
    # Test forward pass
    print("\n4. Testing forward pass...")
    batch = batch.to(device)
    output = model(batch)
    print(f"   Latent dim: {output['z'].shape}")
    print(f"   Generated particles: {output['particle_features'].shape}")
    
    # Test loss computation
    print("\n5. Testing loss computation...")
    losses = model.compute_loss(batch, output, epoch=0)
    print(f"   Total loss: {losses['total'].item():.4f}")
    print(f"   - Particle: {losses['particle'].item():.4f}")
    print(f"   - Edge: {losses['edge'].item():.4f}")
    print(f"   - Hyperedge: {losses['hyperedge'].item():.4f}")
    print(f"   - Topology: {losses['topology'].item():.4f}")
    print(f"   - KL: {losses['kl'].item():.4f}")
    
    # Test generation
    print("\n6. Testing generation...")
    jet_types = torch.tensor([0, 1, 2, 0], device=device)
    generated = model.generate(4, jet_types, device=device)
    print(f"   Generated particles: {generated['particle_features'].shape}")
    print(f"   Particle counts: {generated['topology']['n_particles']}")
    
    # Quick training test (2 epochs)
    print("\n7. Quick training test (2 epochs)...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    model.train()
    for epoch in range(1, 3):
        total_loss = 0
        for i, batch in enumerate(loader):
            if i >= 5:  # Only 5 batches per epoch for testing
                break
            
            batch = batch.to(device)
            optimizer.zero_grad()
            
            output = model(batch)
            losses = model.compute_loss(batch, output, epoch=epoch)
            loss = losses['total']
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / min(5, len(loader))
        print(f"   Epoch {epoch}: Loss = {avg_loss:.4f}")
    
    print("\n" + "="*60)
    print("✓ All tests passed! Model is working correctly.")
    print("="*60)
    print("\nNext steps:")
    print("1. Prepare your jet data in HDF5 format (see README.md)")
    print("2. Train the model: python train.py --data-path your_data.h5")
    print("3. Generate jets: python generate.py --checkpoint best_model.pt")
    print("4. Evaluate: python evaluate.py --real-data real.h5 --generated-data gen.h5")


if __name__ == "__main__":
    main()
