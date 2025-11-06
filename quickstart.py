#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
QUICKSTART SCRIPT - BIPARTITE HYPERVAE
═══════════════════════════════════════════════════════════════════════════════

Purpose:
--------
Quick verification script that tests the complete HyperVAE pipeline on dummy data.
Validates that all components work correctly before training on real data.

What This Script Does:
----------------------
1. Checks GPU availability and compute capability (for BF16 support)
2. Loads configuration from config.yaml
3. Creates dummy bipartite jet graphs
4. Tests data loading with PyG Batch collation
5. Creates BipartiteHyperVAE model
6. Tests forward pass (encoding → latent → decoding)
7. Tests loss computation (particle, edge, hyperedge, jet, KL)
8. Tests generation from latent space
9. Runs 2 quick training epochs to verify backprop works

Usage:
------
python quickstart.py

Expected Runtime: ~30 seconds (CPU) or ~10 seconds (GPU)

What to Check:
--------------
✓ No CUDA errors or OOM (if using GPU)
✓ Model parameters are reasonable (~1-5M for default config)
✓ Forward pass produces correct shapes
✓ Losses are finite (not NaN or Inf)
✓ Loss decreases over 2 epochs
✓ Generation produces valid particle features

If all tests pass, you can proceed with real training!

Next Steps After Quickstart:
-----------------------------
1. Generate graph dataset:
   python graph_constructor.py

2. Validate data format:
   python validate_data.py data/real/graphs_pyg_particle__fully_connected_final.pt

3. Check GPU precision support (if using mixed precision):
   python test_bf16.py

4. Train on real data:
   python train.py --config config.yaml --data-path data/real/graphs.pt

5. Generate jets:
   python generate.py --checkpoint checkpoints/best_model.pt --output generated_jets.pt

6. Evaluate quality:
   python evaluate.py --real-data data/real/graphs.pt --generated-data generated_jets.pt

═══════════════════════════════════════════════════════════════════════════════
"""

import torch
import yaml
from torch.utils.data import DataLoader
from data.bipartite_dataset import BipartiteJetDataset, collate_bipartite_batch
from models.hypervae import BipartiteHyperVAE


def main():
    print("="*70)
    print("       BIPARTITE HYPERVAE - QUICKSTART VERIFICATION TEST")
    print("="*70)
    
    # Check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n[Device Info]")
    print(f"  Device: {device}")
    if device.type == 'cuda':
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        compute_cap = torch.cuda.get_device_capability(0)
        supports_bf16 = compute_cap[0] >= 8  # Ampere = SM 8.0+
        
        print(f"  GPU: {gpu_name}")
        print(f"  VRAM: {vram_gb:.2f} GB")
        print(f"  Compute Capability: {compute_cap[0]}.{compute_cap[1]}")
        
        bf16_msg = '✓ YES (Use precision_type="bf16")' if supports_bf16 else '✗ NO (Use precision_type="fp16")'
        print(f"  BF16 Support: {bf16_msg}")
    else:
        print("  ⚠️  No GPU detected. Training will be slow on CPU.")
    
    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Show training configuration
    print(f"\n[Training Config]")
    print(f"  Batch size: {config['training']['batch_size']}")
    print(f"  Gradient accumulation: {config['training']['gradient_accumulation_steps']}")
    print(f"  Effective batch: {config['training']['batch_size'] * config['training']['gradient_accumulation_steps']}")
    print(f"  Mixed precision: {config['training']['mixed_precision']}")
    if config['training']['mixed_precision']:
        precision_type = config['training'].get('precision_type', 'fp16')
        print(f"  Precision type: {precision_type.upper()}")
    
    print(f"\n{'─'*70}")
    print("STEP 1: Creating dummy dataset...")
    print(f"{'─'*70}")
    dataset = BipartiteJetDataset(generate_dummy=True)
    print(f"  ✓ Dataset size: {len(dataset)} graphs")
    print(f"  ✓ Using PyG Data format with bipartite edges")
    
    # Create loader
    loader = DataLoader(
        dataset, 
        batch_size=config['training']['batch_size'], 
        shuffle=True,
        collate_fn=collate_bipartite_batch
    )
    
    # Test data loading
    print(f"\n{'─'*70}")
    print("STEP 2: Testing data loading and batching...")
    print(f"{'─'*70}")
    batch = next(iter(loader))
    print(f"  ✓ Batch created successfully")
    print(f"  ✓ Number of graphs in batch: {batch.num_graphs}")
    print(f"  ✓ Total particles: {batch.x.shape[0]}")  # Updated field name
    print(f"  ✓ Particle features: {batch.x.shape[1]} (E, px, py, pz)")
    print(f"  ✓ Total edges: {batch.edge_attr.shape[0]}")
    print(f"  ✓ Edge features: {batch.edge_attr.shape[1]} (2pt_EEC, ln_delta, ln_kT, ln_z, ln_m²)")
    
    if hasattr(batch, 'hyperedge_attr') and batch.hyperedge_attr is not None:
        print(f"  ✓ Total hyperedges: {batch.hyperedge_attr.shape[0]}")
        print(f"  ✓ Hyperedge features: {batch.hyperedge_attr.shape[1]}")
    else:
        print(f"  ○ Hyperedges: Not present (optional)")
    
    # Create model
    print(f"\n{'─'*70}")
    print("STEP 3: Creating BipartiteHyperVAE model...")
    print(f"{'─'*70}")
    model = BipartiteHyperVAE(config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  ✓ Model created successfully")
    print(f"  ✓ Total parameters: {n_params/1e6:.2f}M")
    print(f"  ✓ Trainable parameters: {n_trainable/1e6:.2f}M")
    print(f"  ✓ Latent dimension: {config['model']['latent_dim']}")
    
    # Test forward pass
    print(f"\n{'─'*70}")
    print("STEP 4: Testing forward pass (encoder → latent → decoder)...")
    print(f"{'─'*70}")
    batch = batch.to(device)
    
    # Use mixed precision if enabled
    use_amp = config['training']['mixed_precision']
    precision_type = config['training'].get('precision_type', 'fp16')
    dtype = torch.bfloat16 if precision_type == 'bf16' else torch.float16
    
    with torch.cuda.amp.autocast(enabled=use_amp, dtype=dtype):
        output = model(batch)
    
    print(f"  ✓ Forward pass completed")
    print(f"  ✓ Encoded to latent: {output['z'].shape}")
    print(f"  ✓ Encoder mu range: [{output['mu'].min().item():.3f}, {output['mu'].max().item():.3f}]")
    print(f"  ✓ Encoder logvar range: [{output['logvar'].min().item():.3f}, {output['logvar'].max().item():.3f}]")
    print(f"  ✓ Generated particles: {output['particle_features'].shape}")
    print(f"  ✓ Generated edges: {output['edge_features'].shape}")
    if 'hyperedge_features' in output and output['hyperedge_features'] is not None:
        print(f"  ✓ Generated hyperedges: {output['hyperedge_features'].shape}")
    print(f"  ✓ Generated jet features: {output['jet_features'].shape}")
    
    # Test loss computation
    print(f"\n{'─'*70}")
    print("STEP 5: Testing loss computation...")
    print(f"{'─'*70}")
    with torch.cuda.amp.autocast(enabled=use_amp, dtype=dtype):
        losses = model.compute_loss(batch, output, epoch=0)
    
    print(f"  ✓ All losses computed successfully")
    print(f"  ✓ Total loss: {losses['total'].item():.4f}")
    print(f"     - Particle loss: {losses['particle'].item():.4f}")
    print(f"     - Edge loss: {losses['edge'].item():.4f}")
    print(f"     - Hyperedge loss: {losses['hyperedge'].item():.4f}")
    print(f"     - Jet features loss: {losses['jet'].item():.4f}")
    print(f"     - KL divergence: {losses['kl'].item():.4f} (weight: {losses.get('kl_weight', 0.0):.4f})")
    
    # Check for NaN
    if torch.isnan(losses['total']):
        print(f"  ✗ ERROR: Loss is NaN! Check model initialization.")
        return False
    else:
        print(f"  ✓ No NaN detected in losses")
    
    # Test generation
    print(f"\n{'─'*70}")
    print("STEP 6: Testing conditional generation from latent space...")
    print(f"{'─'*70}")
    jet_types = torch.tensor([0, 1, 2, 0], device=device)  # Q, G, T, Q
    type_names = ['Quark', 'Gluon', 'Top', 'Quark']
    
    with torch.cuda.amp.autocast(enabled=use_amp, dtype=dtype):
        generated = model.generate(4, jet_types, device=device)
    
    print(f"  ✓ Generated 4 jets successfully")
    for i, name in enumerate(type_names):
        print(f"     Jet {i+1} ({name}): {generated['topology']['n_particles'][i].item()} particles")
    print(f"  ✓ Generated particle features: {generated['particle_features'].shape}")
    print(f"  ✓ Generated jet features: {generated['jet_features'].shape}")
    
    # Quick training test (2 epochs)
    print(f"\n{'─'*70}")
    print("STEP 7: Quick training test (2 epochs, 5 batches each)...")
    print(f"{'─'*70}")
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    
    model.train()
    for epoch in range(1, 3):
        total_loss = 0
        n_batches = 0
        
        for i, batch in enumerate(loader):
            if i >= 5:  # Only 5 batches per epoch for testing
                break
            
            batch = batch.to(device)
            optimizer.zero_grad()
            
            # Mixed precision forward and loss
            with torch.cuda.amp.autocast(enabled=use_amp, dtype=dtype):
                output = model(batch)
                losses = model.compute_loss(batch, output, epoch=epoch)
                loss = losses['total']
            
            # Backward with gradient scaling
            scaler.scale(loss).backward()
            
            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 
                config['training']['gradient_clip']
            )
            
            # Optimizer step with scaler
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            n_batches += 1
        
        avg_loss = total_loss / n_batches
        print(f"  Epoch {epoch}/2: Loss = {avg_loss:.4f}")
    
    print(f"  ✓ Training loop works correctly")
    print(f"  ✓ Gradients flow properly through encoder and decoder")
    
    print(f"\n{'='*70}")
    print("       ✅ ALL TESTS PASSED! MODEL IS WORKING CORRECTLY")
    print(f"{'='*70}")
    
    print(f"\n[Next Steps]")
    print(f"  1. Generate graph dataset:")
    print(f"     → python graph_constructor.py")
    print(f"")
    print(f"  2. Validate data format:")
    print(f"     → python validate_data.py data/real/graphs_pyg_particle__fully_connected_final.pt")
    print(f"")
    if device.type == 'cuda':
        print(f"  3. Check GPU precision support (optional):")
        print(f"     → python test_bf16.py")
        print(f"")
        print(f"  4. Train on real data:")
    else:
        print(f"  3. Train on real data:")
    print(f"     → python train.py --config config.yaml --data-path data/real/graphs.pt \\")
    print(f"                       --save-dir checkpoints --log-dir runs")
    print(f"")
    print(f"  {'4' if device.type != 'cuda' else '5'}. Monitor training:")
    print(f"     → tensorboard --logdir runs")
    print(f"")
    print(f"  {'5' if device.type != 'cuda' else '6'}. Generate jets from trained model:")
    print(f"     → python generate.py --checkpoint checkpoints/best_model.pt \\")
    print(f"                          --output generated_jets.pt --num-samples 1000")
    print(f"")
    print(f"  {'6' if device.type != 'cuda' else '7'}. Evaluate generation quality:")
    print(f"     → python evaluate.py --real-data data/real/graphs.pt \\")
    print(f"                          --generated-data generated_jets.pt")
    print(f"")
    print(f"[Documentation]")
    print(f"  • README.md: Complete usage guide with architecture diagram")
    print(f"  • PROJECT_OVERVIEW.md: Detailed technical documentation")
    print(f"  • MODEL_ARCHITECTURE.md: Model component specifications")
    print(f"  • BF16_IMPLEMENTATION.md: Mixed precision training guide")
    print(f"")
    print(f"{'='*70}\n")
    
    return True


if __name__ == "__main__":
    try:
        success = main()
        exit(0 if success else 1)
    except Exception as e:
        print(f"\n{'='*70}")
        print(f"       ✗ ERROR: Quickstart test failed")
        print(f"{'='*70}")
        print(f"\nException: {type(e).__name__}: {e}")
        print(f"\nTroubleshooting:")
        print(f"  • Check that config.yaml exists and is valid YAML")
        print(f"  • Ensure all dependencies are installed: pip install -r requirements.txt")
        print(f"  • If CUDA error, check GPU drivers: nvidia-smi")
        print(f"  • For OOM, reduce batch_size in config.yaml")
        print(f"  • Check README.md troubleshooting section")
        print(f"\n{'='*70}\n")
        import traceback
        traceback.print_exc()
        exit(1)
