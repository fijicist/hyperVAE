"""
═══════════════════════════════════════════════════════════════════════════════
TRAINING SCRIPT - BIPARTITE HYPERVAE FOR JET GENERATION
═══════════════════════════════════════════════════════════════════════════════

Purpose:
--------
Train a Variational Autoencoder (VAE) with L-GATr for jet generation using:
- Mixed-precision FP16 training (memory-efficient for GTX 1650 Ti)
- Gradient accumulation (simulate larger batch sizes)
- Gradient clipping (stability for squared distance gradients)
- KL annealing (prevent posterior collapse)
- Checkpoint saving & resuming
- TensorBoard logging

Usage:
------
Basic training:
    python train.py --config config.yaml --data-path data/jets.pt \
                    --save-dir checkpoints --log-dir runs

Resume from checkpoint:
    python train.py --config config.yaml --resume --data-path data/jets.pt

Key Features:
-------------
1. MIXED PRECISION (FP16/BF16):
   - FP16 (float16): Reduces memory by 2×, faster on Volta/Turing GPUs (V100, T4, RTX 20xx)
   - BF16 (bfloat16): Better stability, recommended for Ampere+ GPUs (A100, RTX 30xx/40xx)
   - Uses GradScaler to prevent underflow (FP16) or for consistency (BF16)
   - Configure via: mixed_precision=true, precision_type="fp16" or "bf16"

2. GRADIENT ACCUMULATION:
   - Simulates batch_size × accumulation_steps effective batch
   - Example: batch=64, steps=4 → effective batch=256
   - Critical for memory-constrained GPUs

3. GRADIENT CLIPPING:
   - Clamps gradient norm to prevent explosion
   - Especially important with squared distance loss (stronger gradients)
   - Typical: gradient_clip=1.5 for squared distance

4. KL ANNEALING:
   - β(epoch) = min(1.0, epoch / warmup_epochs) × kl_max_weight
   - Starts at β=0 (focus on reconstruction)
   - Gradually increases to kl_max_weight (enforce prior matching)
   - Prevents posterior collapse (KL → 0 issue)

5. CHECKPOINT SAVING:
   - Saves best model (lowest val loss)
   - Periodic checkpoints every 20 epochs
   - Stores: model, optimizer, scheduler, scaler, epoch, losses
   - Resume training with --resume flag

6. DATA SPLITTING:
   - 80% train / 10% val / 10% test (automatic)
   - Deterministic split (seed=42)
   - Handles PyG Batch format with variable-length graphs

Training Loop:
--------------
For each epoch:
    1. Train one epoch (forward → backward → update)
    2. Validate on validation set
    3. Save checkpoint if val_loss improved
    4. Log metrics to TensorBoard
    5. Update learning rate (ReduceLROnPlateau)

NaN Handling:
-------------
- Detects NaN/Inf in loss at every batch
- Prints which loss component caused NaN
- Skips bad batches (allows training to continue)
- Logs encoder mu/logvar for debugging

Arguments:
----------
--config: Path to YAML config file (required)
--data-path: Path to PyTorch dataset file (.pt) (required)
--save-dir: Directory for checkpoints (default: checkpoints)
--log-dir: Directory for TensorBoard logs (default: runs)
--resume: Resume from latest checkpoint in save-dir (flag)
--gpu: Use GPU if available (default: True)

Config File Structure:
----------------------
See config.yaml for full configuration. Key sections:

training:
  batch_size: 64                      # Per-GPU batch size
  learning_rate: 0.0001              # Adam LR
  epochs: 200                        # Total epochs
  gradient_accumulation_steps: 4     # Effective batch = 64×4=256
  gradient_clip: 1.5                 # Gradient clipping threshold
  mixed_precision: true              # Use FP16/BF16
  precision_type: "fp16"             # "fp16" (Volta/Turing) or "bf16" (Ampere+)
  kl_warmup_epochs: 100              # KL annealing warmup
  kl_max_weight: 0.05                # Final KL weight

model:
  latent_dim: 128                    # VAE latent dimension
  max_particles: 100                 # Max particles per jet

loss:
  use_squared_distance: true         # Use d² instead of d (recommended)
  particle_distance_metric: euclidean_4d
  w_energy: 2.0                      # Energy weight in 4D distance

Outputs:
--------
Checkpoints saved to save_dir/:
- best_model.pt: Best validation loss
- checkpoint_epoch{N}.pt: Periodic checkpoints

TensorBoard logs saved to log_dir/:
- train/loss, train/particle_loss, train/kl_loss, etc.
- val/loss, val/particle_loss, etc.
- View with: tensorboard --logdir runs

Typical Training Curves:
-------------------------
Healthy training should show:
- Total loss: 50-100 → 5-20 over 100 epochs (depends on dataset)
- Particle loss: 1.5-2.0 → 0.5-1.0 (Chamfer distance, squared)
- KL loss: 0.01 → 10-50 (anneals from near-zero to stable value)
- Val loss: Tracks train loss, may plateau higher (generalization gap)

Troubleshooting:
----------------
Issue: Loss not decreasing
- Check gradient_clip (too low = no progress, too high = instability)
- Check KL weight (if KL >> recon, model ignores decoder)
- Check use_squared_distance=true (euclidean has vanishing gradients)

Issue: NaN in training
- Check encoder clamping (mu, logvar in [-10, 10])
- Check KL computation (should clamp per-dim KL)
- Reduce learning rate or increase gradient_clip

Issue: OOM (Out of Memory)
- Reduce batch_size (64 → 32)
- Reduce hidden_mv_channels in L-GATr (32 → 16)
- Increase gradient_accumulation_steps (maintain effective batch)

Issue: Val loss much higher than train loss
- Model overfitting, increase regularization
- Reduce model capacity (latent_dim, hidden_mv_channels)
- Increase KL weight (kl_max_weight)

═══════════════════════════════════════════════════════════════════════════════
"""
import torch
import torch.optim as optim
from lion_pytorch import Lion
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import yaml
import os
from tqdm import tqdm
import argparse

from data.bipartite_dataset import BipartiteJetDataset, collate_bipartite_batch, create_train_val_test_split
from models.hypervae import BipartiteHyperVAE


def train_epoch(model, loader, optimizer, scaler, config, epoch, writer, device):
    """Train for one epoch"""
    model.train()
    
    # Initialize loss accumulators as tensors on GPU 
    total_loss = torch.tensor(0.0, device=device)
    losses_dict = {
        'particle': torch.tensor(0.0, device=device),
        'edge_distribution': torch.tensor(0.0, device=device),
        'hyperedge_distribution': torch.tensor(0.0, device=device),
        'jet': torch.tensor(0.0, device=device),
        'kl': torch.tensor(0.0, device=device),
        'consistency': torch.tensor(0.0, device=device),
        'kl_weight': 0.0  # Keep as float, not a loss
    }
    
    # Track gradient norms
    grad_norms = []
    clip_count = 0  # Count how many times gradients were clipped
    
    # Gradient accumulation
    accumulation_steps = config['training']['gradient_accumulation_steps']
    optimizer.zero_grad()
    
    # Determine precision dtype
    precision_type = config['training'].get('precision_type', 'fp16')
    use_mixed_precision = config['training']['mixed_precision']
    dtype = torch.bfloat16 if precision_type == 'bf16' else torch.float16
    
    pbar = tqdm(loader, desc=f"Epoch {epoch}")
    for i, batch in enumerate(pbar):
        batch = batch.to(device)
        
        # Gumbel temperature annealing from config
        initial_temp = config['training'].get('initial_temperature', 5.0)
        final_temp = config['training'].get('final_temperature', 0.5)
        temp_decay = config['training'].get('temperature_decay', 0.98)
        
        # Exponential decay: temp = final + (initial - final) * decay^epoch
        temperature = final_temp + (initial_temp - final_temp) * (temp_decay ** epoch)
        temperature = max(final_temp, temperature)  # Clamp to final temp
        
        # Mixed precision training with configurable dtype
        with torch.cuda.amp.autocast(enabled=use_mixed_precision, dtype=dtype):
            output = model(batch, temperature=temperature)
            losses = model.compute_loss(batch, output, epoch=epoch)
            loss = losses['total'] / accumulation_steps
        
        # NaN detection
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"\n⚠️  NaN/Inf detected at epoch {epoch}, batch {i}")
            
            # Helper function to safely get value
            def get_value(x):
                if isinstance(x, torch.Tensor):
                    if x.numel() == 1:
                        return x.item()
                    return x
                return x
            
            print(f"  Particle loss: {get_value(losses['particle'])}")
            print(f"  Edge distribution loss: {get_value(losses.get('edge_distribution', 0.0))}")
            print(f"  Hyperedge distribution loss: {get_value(losses.get('hyperedge_distribution', 0.0))}")
            print(f"  Jet loss: {get_value(losses['jet'])}")
            print(f"  KL loss: {get_value(losses['kl'])}")
            
            # Check mu and logvar for NaN
            if torch.isnan(output['mu']).any():
                print(f"  ⚠️  NaN in encoder mu!")
            if torch.isnan(output['logvar']).any():
                print(f"  ⚠️  NaN in encoder logvar!")
            if torch.isnan(output['particle_features']).any():
                print(f"  ⚠️  NaN in particle predictions!")
            
            print(f"  Skipping batch...")
            optimizer.zero_grad()
            continue
        
        # Backward with gradient scaling
        scaler.scale(loss).backward()
        
        # Gradient accumulation
        if (i + 1) % accumulation_steps == 0:
            # Gradient clipping
            scaler.unscale_(optimizer)
            
            # Compute gradient norm before clipping (OPTIMIZED - avoid multiple .item() calls)
            # Stack all gradient norms into a single tensor, then compute on GPU
            grad_norms_tensor = torch.stack([
                p.grad.data.norm(2) 
                for p in model.parameters() 
                if p.grad is not None
            ])
            total_norm = torch.norm(grad_norms_tensor, 2).item()  # Single GPU→CPU sync
            
            # Track gradient norm
            grad_norms.append(total_norm)
            
            # Check if clipping will occur
            gradient_clip_threshold = config['training']['gradient_clip']
            if total_norm > gradient_clip_threshold:
                clip_count += 1
            
            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 
                gradient_clip_threshold
            )
            
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        # Logging - accumulate on GPU to avoid sync (OPTIMIZED)
        # Only convert to CPU once at the end of epoch
        if isinstance(losses['total'], torch.Tensor):
            total_loss += losses['total'].detach()
        else:
            total_loss += losses['total']
            
        for key in losses_dict.keys():
            if key in losses and key != 'kl_weight':
                if isinstance(losses[key], torch.Tensor):
                    losses_dict[key] += losses[key].detach()
                else:
                    losses_dict[key] += losses[key]
        
        # Update kl_weight separately (it's a scalar, not accumulated)
        if 'kl_weight' in losses:
            losses_dict['kl_weight'] = losses['kl_weight']
        
        # Update progress bar less frequently to avoid GPU→CPU sync overhead
        # Only update every 10 batches instead of every batch
        if i % 10 == 0:  # Update every 10 batches instead of every batch
            # Helper function to safely get scalar value
            def get_scalar(x):
                return x.item() if isinstance(x, torch.Tensor) else x
            
            pbar.set_postfix({
                'loss': get_scalar(losses['total']),
                'part': get_scalar(losses['particle']),
                'kl': get_scalar(losses['kl']),
                'kl_w': losses.get('kl_weight', 0.0)
            })
    
    # Average losses - convert to CPU only once at the end
    n_batches = len(loader)
    total_loss = (total_loss / n_batches).item()  # Single GPU→CPU transfer
    for key in losses_dict.keys():
        if key != 'kl_weight':
            losses_dict[key] = (losses_dict[key] / n_batches).item()  # Single transfer per loss
    
    # Compute gradient statistics
    avg_grad_norm = sum(grad_norms) / len(grad_norms) if grad_norms else 0.0
    max_grad_norm = max(grad_norms) if grad_norms else 0.0
    clip_frequency = clip_count / len(grad_norms) if grad_norms else 0.0
    
    # TensorBoard logging
    writer.add_scalar('Loss/train_total', total_loss, epoch)
    for key, value in losses_dict.items():
        writer.add_scalar(f'Loss/train_{key}', value, epoch)
    
    # Log gradient statistics
    writer.add_scalar('Gradients/norm_avg', avg_grad_norm, epoch)
    writer.add_scalar('Gradients/norm_max', max_grad_norm, epoch)
    writer.add_scalar('Gradients/clip_frequency', clip_frequency, epoch)
    writer.add_scalar('Gradients/clip_threshold', config['training']['gradient_clip'], epoch)
    
    # Log KL weight for monitoring
    if epoch > 0:
        writer.add_scalar('KL/weight', losses.get('kl_weight', 0.0), epoch)
    
    # Return losses and gradient statistics
    grad_stats = {
        'avg_norm': avg_grad_norm,
        'max_norm': max_grad_norm,
        'clip_freq': clip_frequency
    }
    
    return total_loss, losses_dict, grad_stats


def validate(model, loader, config, epoch, writer, device):
    """Validate model"""
    model.eval()
    total_loss = 0.0
    losses_dict = {
        'particle': 0.0,
        'edge_distribution': 0.0,
        'hyperedge_distribution': 0.0,
        'jet': 0.0,
        'kl': 0.0,
        'consistency': 0.0
    }
    
    # Determine precision dtype
    precision_type = config['training'].get('precision_type', 'fp16')
    use_mixed_precision = config['training']['mixed_precision']
    dtype = torch.bfloat16 if precision_type == 'bf16' else torch.float16
    
    # Use final (converged) temperature for validation
    # This represents the model's behavior when fully trained
    val_temperature = config['training'].get('final_temperature', 0.5)
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Validation"):
            batch = batch.to(device)
            
            # Use mixed precision for validation too
            with torch.cuda.amp.autocast(enabled=use_mixed_precision, dtype=dtype):
                output = model(batch, temperature=val_temperature)
                losses = model.compute_loss(batch, output, epoch=epoch)
            
            total_loss += losses['total'].item() if isinstance(losses['total'], torch.Tensor) else losses['total']
            for key in losses_dict.keys():
                if key in losses:
                    value = losses[key]
                    # Handle both tensor and float values
                    losses_dict[key] += value.item() if isinstance(value, torch.Tensor) else value
    
    # Average losses
    n_batches = len(loader)
    total_loss /= n_batches
    for key in losses_dict.keys():
        losses_dict[key] /= n_batches
    
    # TensorBoard logging
    writer.add_scalar('Loss/val_total', total_loss, epoch)
    for key, value in losses_dict.items():
        writer.add_scalar(f'Loss/val_{key}', value, epoch)
    
    return total_loss, losses_dict  # Return both total and individual losses


def main(args):
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Create datasets
    if args.data_path:
        print(f"\nLoading data from: {args.data_path}")
        full_dataset = BipartiteJetDataset(data_path=args.data_path)
        
        # Check if separate validation/test files provided
        if args.val_data_path and args.test_data_path:
            # Use separate files
            print(f"Loading validation data from: {args.val_data_path}")
            val_dataset = BipartiteJetDataset(data_path=args.val_data_path)
            print(f"Loading test data from: {args.test_data_path}")
            test_dataset = BipartiteJetDataset(data_path=args.test_data_path)
            train_dataset = full_dataset
        elif args.val_data_path:
            # Use single validation file, split remaining for train/test
            print(f"Loading validation data from: {args.val_data_path}")
            val_dataset = BipartiteJetDataset(data_path=args.val_data_path)
            # Split train data into train and test
            train_size = int(len(full_dataset) * 0.89)  # 80/(80+10) to keep proportions
            test_size = len(full_dataset) - train_size
            train_dataset, test_dataset = torch.utils.data.random_split(
                full_dataset, [train_size, test_size]
            )
            print(f"Split train data: {train_size} train, {test_size} test")
        else:
            # Auto-split single file into train/val/test (80/10/10)
            print("Auto-splitting data into train/val/test (80%/10%/10%)")
            train_dataset, val_dataset, test_dataset = create_train_val_test_split(
                full_dataset,
                train_frac=args.train_frac,
                val_frac=args.val_frac,
                test_frac=args.test_frac,
                seed=args.split_seed
            )
            
            # Save test dataset path for later evaluation
            if args.save_test_indices:
                test_indices_path = os.path.join(args.save_dir, 'test_indices.pt')
                torch.save(test_dataset.indices, test_indices_path)
                print(f"Saved test indices to: {test_indices_path}")
    else:
        print("No data path provided. Generating dummy data for testing...")
        train_dataset = BipartiteJetDataset(generate_dummy=True)
        val_dataset = BipartiteJetDataset(generate_dummy=True)
    
    # Create custom collate function that adds normalization statistics
    def collate_with_stats(data_list):
        """Wrapper around collate_bipartite_batch that adds normalization statistics"""
        batch = collate_bipartite_batch(data_list)
        
        # Get the base dataset (unwrap if it's a Subset)
        base_dataset = train_dataset
        if isinstance(train_dataset, torch.utils.data.Subset):
            base_dataset = train_dataset.dataset
        
        # Attach normalization statistics from dataset to batch
        if hasattr(base_dataset, 'particle_norm_stats') and base_dataset.particle_norm_stats is not None:
            batch.particle_norm_stats = base_dataset.particle_norm_stats
        if hasattr(base_dataset, 'jet_norm_stats') and base_dataset.jet_norm_stats is not None:
            batch.jet_norm_stats = base_dataset.jet_norm_stats
        if hasattr(base_dataset, 'edge_norm_stats') and base_dataset.edge_norm_stats is not None:
            batch.edge_norm_stats = base_dataset.edge_norm_stats
        if hasattr(base_dataset, 'hyperedge_norm_stats') and base_dataset.hyperedge_norm_stats is not None:
            batch.hyperedge_norm_stats = base_dataset.hyperedge_norm_stats
        
        return batch
    
    # Create loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['data']['num_workers'],
        pin_memory=config['data']['pin_memory'],
        collate_fn=collate_with_stats,
        prefetch_factor=config['data']['prefetch_factor']
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=config['data']['pin_memory'],
        collate_fn=collate_with_stats
    )
    
    # Create model
    model = BipartiteHyperVAE(config).to(device)
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
    # Optimizer
    # optimizer = optim.AdamW(
    #     model.parameters(),
    #     lr=config['training']['learning_rate'],
    #     weight_decay=config['training']['weight_decay']
    # )
    
    optimizer = Lion(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'], 
        betas=(0.95, 0.98), 
        use_triton=True
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=config['training']['warmup_epochs'],
        T_mult=2
    )
    
    # Mixed precision scaler (works with both FP16 and BF16)
    use_mixed_precision = config['training']['mixed_precision']
    precision_type = config['training'].get('precision_type', 'fp16')
    scaler = torch.cuda.amp.GradScaler(enabled=use_mixed_precision)
    
    if use_mixed_precision:
        print(f"\nMixed precision training enabled: {precision_type.upper()}")
        if precision_type == 'bf16':
            print("   Using BF16 (bfloat16) - Recommended for Ampere+ GPUs (A100, RTX 30xx/40xx)")
            print("   Benefits: Better numerical stability, wider dynamic range than FP16")
        else:
            print("   Using FP16 (float16) - Recommended for Volta/Turing GPUs (V100, T4, RTX 20xx)")
            print("   Benefits: 2× memory savings, faster on older GPUs")
    else:
        print("\nMixed precision training disabled - Using FP32")
    
    # Resume from checkpoint if specified
    start_epoch = 1
    if args.resume:
        if os.path.exists(args.resume):
            print(f"\nLoading checkpoint from: {args.resume}")
            checkpoint = torch.load(args.resume, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            
            # Load scheduler state if available
            if 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            # Load scaler state if available
            if 'scaler_state_dict' in checkpoint:
                scaler.load_state_dict(checkpoint['scaler_state_dict'])
            
            print(f"✅ Resumed training from epoch {checkpoint['epoch']}")
            print(f"   Starting at epoch {start_epoch}")
        else:
            print(f"⚠️  Checkpoint not found: {args.resume}")
            print(f"   Starting training from scratch")
    
    # TensorBoard
    writer = SummaryWriter(log_dir=args.log_dir)
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(start_epoch, config['training']['epochs'] + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{config['training']['epochs']}")
        print(f"{'='*60}")
        
        # Train
        train_loss, losses_dict, grad_stats = train_epoch(
            model, train_loader, optimizer, scaler, config, epoch, writer, device
        )
        
        # Validate - use config parameter
        validate_every = config['training'].get('validate_every', 5)
        if epoch % validate_every == 0:       # Validate
            val_loss, val_losses_dict = validate(model, val_loader, config, epoch, writer, device)
            print(f"\nEpoch {epoch}:")
            print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            
            # Print gradient statistics
            print(f"  Grad Norm: avg={grad_stats['avg_norm']:.2f}, "
                  f"max={grad_stats['max_norm']:.2f}, "
                  f"clipped={grad_stats['clip_freq']*100:.1f}%")
            
            # Print training losses
            print(f"  Train - Part: {losses_dict['particle']:.4f}, "
                  f"Edge Dist: {losses_dict['edge_distribution']:.4f}, "
                  f"Hyper Dist: {losses_dict['hyperedge_distribution']:.4f}, "
                  f"Jet: {losses_dict['jet']:.4f}, "
                  f"Cons: {losses_dict['consistency']:.4f}, "
                  f"KL: {losses_dict['kl']:.4f} (w={losses_dict.get('kl_weight', 0.0):.4f})")
            
            # Print validation losses
            print(f"  Val   - Part: {val_losses_dict['particle']:.4f}, "
                  f"Edge Dist: {val_losses_dict['edge_distribution']:.4f}, "
                  f"Hyper Dist: {val_losses_dict['hyperedge_distribution']:.4f}, "
                  f"Jet: {val_losses_dict['jet']:.4f}, "
                  f"Cons: {val_losses_dict['consistency']:.4f}, "
                  f"KL: {val_losses_dict['kl']:.4f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'config': config
                }, os.path.join(args.save_dir, 'best_model.pt'))
                print(f"Saved best model (val_loss: {val_loss:.4f})")
        else:
            # Print gradient statistics for every epoch even if not validating
            print(f"  Grad Norm: avg={grad_stats['avg_norm']:.2f}, "
                  f"max={grad_stats['max_norm']:.2f}, "
                  f"clipped={grad_stats['clip_freq']*100:.1f}%")
            
            # Print training losses for every epoch even if not validating
            print(f"  Train - Part: {losses_dict['particle']:.4f}, "
                  f"Edge Dist: {losses_dict['edge_distribution']:.4f}, "
                  f"Hyper Dist: {losses_dict['hyperedge_distribution']:.4f}, "
                  f"Jet: {losses_dict['jet']:.4f}, "
                  f"Cons: {losses_dict['consistency']:.4f}, "
                  f"KL: {losses_dict['kl']:.4f} (w={losses_dict.get('kl_weight', 0.0):.4f})")
        
        # Save checkpoint
        save_every = config['training'].get('save_every', 20)
        if epoch % save_every == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'config': config
            }, os.path.join(args.save_dir, f'checkpoint_epoch{epoch}.pt'))
        
        # Learning rate scheduling
        scheduler.step()
        writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
    
    writer.close()
    print("\nTraining completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Bipartite HyperVAE')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--data-path', type=str, default=None, help='Path to data .pt file (will be auto-split if no val/test provided)')
    parser.add_argument('--val-data-path', type=str, default=None, help='Path to validation data .pt file (optional)')
    parser.add_argument('--test-data-path', type=str, default=None, help='Path to test data .pt file (optional)')
    parser.add_argument('--save-dir', type=str, default='checkpoints', help='Directory to save models')
    parser.add_argument('--log-dir', type=str, default='runs', help='Directory for TensorBoard logs')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume training from')
    
    # Data splitting arguments
    parser.add_argument('--train-frac', type=float, default=0.8, help='Training fraction (default: 0.8)')
    parser.add_argument('--val-frac', type=float, default=0.1, help='Validation fraction (default: 0.1)')
    parser.add_argument('--test-frac', type=float, default=0.1, help='Test fraction (default: 0.1)')
    parser.add_argument('--split-seed', type=int, default=42, help='Random seed for data splitting')
    parser.add_argument('--save-test-indices', action='store_true', help='Save test indices for reproducibility')
    
    args = parser.parse_args()
    
    # Create directories
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    main(args)
