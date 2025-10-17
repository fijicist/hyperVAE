# Training Tips for Small Datasets

## Your Situation

- **Dataset**: 250 jets (200 train, 25 val, 25 test)
- **Issue**: Loss not decreasing, KL divergence exploding (8 → 73)
- **VRAM Usage**: 1.5GB / 4GB (underutilized)

## Problem Analysis

### 1. KL Divergence Exploding ❌
```
Epoch 23: kl=8.19
Epoch 89: kl=12.9
Epoch 108: kl=42.8
Epoch 193: kl=72.9  ← Too high!
```

**Root cause**: KL weight growing too fast without cap.

### 2. Loss Not Decreasing ❌
```
loss=8.18e+8, loss=8.46e+8, loss=9.89e+8
```

**Root causes**:
- Features not normalized (pt~20, eta~2, phi~3)
- Loss weights too high for unnormalized features
- Small dataset overfitting

### 3. Small Dataset ⚠️
- 250 jets → 200 training samples
- VAEs typically need 1000+ samples
- High variance in gradients

## Solutions Applied

### ✅ 1. Fixed KL Divergence

**Before:**
```yaml
kl_divergence: 0.001
kl_warmup_epochs: 50
# No maximum cap!
```

**After:**
```yaml
kl_divergence: 0.00001     # 100x smaller base weight
kl_warmup_epochs: 100      # Slower warmup
kl_max_weight: 0.0001      # Cap at 0.0001 (never exceed)
```

**Why**: KL was growing unbounded. Now it caps at 0.0001 × KL_loss.

### ✅ 2. Features Already Normalized

Your data already has normalized features (confirmed by user).
- Node features (pt, η, φ): Pre-normalized
- Edge features (5D): Pre-normalized  
- Hyperedge features (2D EEC): Pre-normalized

**No additional normalization needed.** ✓

### ✅ 3. Rebalanced Loss Weights

**Before:**
```yaml
particle_features: 10.0
edge_features: 5.0
hyperedge_features: 3.0
```

**After:**
```yaml
particle_features: 1.0   # 10x smaller
edge_features: 0.5       # 10x smaller
hyperedge_features: 0.3  # 10x smaller
```

**Why**: Normalized features have smaller magnitudes, so weights should be smaller.

### ✅ 4. Increased Batch Size

**Before**: batch_size=4 (only 50 batches per epoch)  
**After**: batch_size=8 (25 batches per epoch)

**Why**: 
- You have 1.5GB/4GB VRAM → can fit larger batches
- Larger batches = more stable gradients
- Fewer steps = faster epochs

### ✅ 5. More Epochs

**Before**: 200 epochs  
**After**: 300 epochs

**Why**: Small datasets need more epochs to converge.

## Expected Behavior Now

### Good Training Signs ✅

```
Epoch 1: loss=1000.0, kl=5.0, kl_w=0.00001
Epoch 50: loss=500.0, kl=8.0, kl_w=0.00005
Epoch 100: loss=200.0, kl=10.0, kl_w=0.0001
Epoch 150: loss=150.0, kl=10.5, kl_w=0.0001  ← KL capped
Epoch 200: loss=120.0, kl=10.8, kl_w=0.0001
Epoch 300: loss=100.0, kl=11.0, kl_w=0.0001
```

**What to look for:**
- ✅ Loss decreasing (not oscillating wildly)
- ✅ KL increasing slowly then stabilizing ~10-15
- ✅ KL weight capped at 0.0001
- ✅ Validation loss tracking training loss

### Warning Signs ⚠️

1. **KL > 20**: Still too high, reduce `kl_max_weight` to 0.00005
2. **Loss increasing**: Learning rate too high, reduce to 0.00005
3. **Val loss >> Train loss**: Overfitting (expected with 250 samples)

## Monitoring

### TensorBoard
```bash
tensorboard --logdir runs
```

**Watch these metrics:**
- `Loss/train_total` - should decrease
- `Loss/train_kl` - should stabilize ~10
- `KL/weight` - should cap at 0.0001
- `Loss/val_total` - should track training

### Console Output
```
Epoch 50: 100%|██████████| 30/30 [00:08<00:00, loss=500, kl=8.0, kl_w=0.00005]
                                                  ↑       ↑       ↑
                                               Total    KL    KL weight
```

## If Still Not Working

### Option 1: Reduce Model Capacity
```yaml
model:
  particle_hidden: 32      # Down from 64
  edge_hidden: 24          # Down from 48
  hyperedge_hidden: 16     # Down from 32
  latent_dim: 64           # Down from 128
```

**Why**: Smaller model = less overfitting on small dataset

### Option 2: Add More Dropout
```yaml
encoder:
  dropout: 0.2  # Up from 0.1
decoder:
  dropout: 0.2  # Up from 0.1
```

### Option 3: Lower Learning Rate
```yaml
training:
  learning_rate: 0.00005  # Down from 0.0001
```

### Option 4: Stronger Regularization
```yaml
training:
  weight_decay: 0.0001  # Up from 0.00001
```

### Option 5: More Data (Best Solution!)

**Current**: 250 jets  
**Recommended**: 1000+ jets

If possible, collect more data or use data augmentation:
- Random rotations in φ
- Random permutations of particles
- Small Gaussian noise

## Quick Test

After making changes, train for 10 epochs and check:

```bash
python train.py --config config.yaml --data-path data.pt --save-test-indices
```

**After 10 epochs, you should see:**
- Loss decreasing (at least a little)
- KL < 15
- No OOM errors
- Training completes without crashes

## Summary of Changes

| Setting | Before | After | Why |
|---------|--------|-------|-----|
| Batch size | 4 | 8 | More stable gradients |
| KL base weight | 0.001 | 0.00001 | Prevent explosion |
| KL max weight | ∞ | 0.0001 | Cap growth |
| KL warmup | 50 | 100 | Slower annealing |
| Particle loss weight | 10.0 | 1.0 | Match normalized scale |
| Normalization | ❌ | ✅ | Stabilize training |
| Epochs | 200 | 300 | More time to converge |

## Next Steps

1. ✅ Use updated config.yaml
2. ✅ Retrain from scratch
3. ✅ Monitor KL weight (should cap at 0.0001)
4. ✅ Check loss decreasing
5. ⏭️ If still issues, reduce model capacity
6. 🎯 Long-term: Get more data (1000+ jets)

---

**Expected timeline with 250 jets:**
- First 50 epochs: Loss drops significantly
- Epochs 50-150: Slow improvement
- Epochs 150-300: Refinement, may plateau

**With these fixes, training should be stable!** 🎯
