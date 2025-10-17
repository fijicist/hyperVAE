import torch
import torch.optim as optim
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
    total_loss = 0.0
    losses_dict = {
        'particle': 0.0,
        'edge': 0.0,
        'hyperedge': 0.0,
        'topology': 0.0,
        'kl': 0.0
    }
    
    # Gradient accumulation
    accumulation_steps = config['training']['gradient_accumulation_steps']
    optimizer.zero_grad()
    
    pbar = tqdm(loader, desc=f"Epoch {epoch}")
    for i, batch in enumerate(pbar):
        batch = batch.to(device)
        
        # Gumbel temperature annealing
        temperature = max(0.5, 1.0 - epoch / 100)
        
        # Mixed precision training
        with torch.cuda.amp.autocast(enabled=config['training']['mixed_precision']):
            output = model(batch, temperature=temperature)
            losses = model.compute_loss(batch, output, epoch=epoch)
            loss = losses['total'] / accumulation_steps
        
        # Backward with gradient scaling
        scaler.scale(loss).backward()
        
        # Gradient accumulation
        if (i + 1) % accumulation_steps == 0:
            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 
                config['training']['gradient_clip']
            )
            
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        # Logging
        total_loss += losses['total'].item()
        for key in losses_dict.keys():
            if key in losses:
                losses_dict[key] += losses[key].item()
        
        pbar.set_postfix({
            'loss': losses['total'].item(),
            'kl': losses['kl'].item(),
            'kl_w': losses.get('kl_weight', 0.0)
        })
    
    # Average losses
    n_batches = len(loader)
    total_loss /= n_batches
    for key in losses_dict.keys():
        losses_dict[key] /= n_batches
    
    # TensorBoard logging
    writer.add_scalar('Loss/train_total', total_loss, epoch)
    for key, value in losses_dict.items():
        writer.add_scalar(f'Loss/train_{key}', value, epoch)
    
    # Log KL weight for monitoring
    if epoch > 0:
        writer.add_scalar('KL/weight', losses.get('kl_weight', 0.0), epoch)
    
    return total_loss


def validate(model, loader, config, epoch, writer, device):
    """Validate model"""
    model.eval()
    total_loss = 0.0
    losses_dict = {
        'particle': 0.0,
        'edge': 0.0,
        'hyperedge': 0.0,
        'topology': 0.0,
        'kl': 0.0
    }
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Validation"):
            batch = batch.to(device)
            
            output = model(batch, temperature=0.5)
            losses = model.compute_loss(batch, output, epoch=epoch)
            
            total_loss += losses['total'].item()
            for key in losses_dict.keys():
                if key in losses:
                    losses_dict[key] += losses[key].item()
    
    # Average losses
    n_batches = len(loader)
    total_loss /= n_batches
    for key in losses_dict.keys():
        losses_dict[key] /= n_batches
    
    # TensorBoard logging
    writer.add_scalar('Loss/val_total', total_loss, epoch)
    for key, value in losses_dict.items():
        writer.add_scalar(f'Loss/val_{key}', value, epoch)
    
    return total_loss


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
    
    # Create loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['data']['num_workers'],
        pin_memory=config['data']['pin_memory'],
        collate_fn=collate_bipartite_batch,
        prefetch_factor=config['data']['prefetch_factor']
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=config['data']['pin_memory'],
        collate_fn=collate_bipartite_batch
    )
    
    # Create model
    model = BipartiteHyperVAE(config).to(device)
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=config['training']['warmup_epochs'],
        T_mult=2
    )
    
    # Mixed precision scaler
    scaler = torch.cuda.amp.GradScaler(enabled=config['training']['mixed_precision'])
    
    # TensorBoard
    writer = SummaryWriter(log_dir=args.log_dir)
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(1, config['training']['epochs'] + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{config['training']['epochs']}")
        print(f"{'='*60}")
        
        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, scaler, config, epoch, writer, device
        )
        
        # Validate
        if epoch % 5 == 0:
            val_loss = validate(model, val_loader, config, epoch, writer, device)
            print(f"\nTrain Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            
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
        
        # Save checkpoint
        if epoch % 20 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
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
