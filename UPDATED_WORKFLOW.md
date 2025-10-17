# Complete Workflow with Your .pt Dataset

## ✅ Your Data Format is Perfect!

Your dataset format matches exactly what we need:
```python
Data(
    x=[30, 3],                    # ✓ Particles (pt, eta, phi)
    edge_index=[2, 870],          # ✓ Edge connections
    edge_attr=[870, 5],           # ✓ Edge features
    hyperedge_index=[2, 121800],  # ✓ Hyperedge connections
    hyperedge_attr=[31465, 2],    # ✓ Hyperedge features (EEC)
    y=[1]                         # ✓ Jet type
)
```

**Note:** If your data has 4 features per particle, the model will automatically use only the first 3 (pt, eta, phi).

## Step-by-Step Workflow

### Step 1: Validate Your Data

First, validate your .pt file to ensure it's in the correct format:

```bash
python validate_data.py /path/to/your/data.pt
```

This will show:
- ✓ Data structure verification
- ✓ Feature dimensions check
- ✓ Statistics (particle count, edges, etc.)
- ✓ Jet type distribution

**Expected output:**
```
Validating data file: your_data.pt
============================================================
1. Loading data...
   ✓ Loaded successfully
   ✓ Data is a list with 10000 jets

2. Checking data structure...
   Jet 0:
      ✓ x (particles): torch.Size([30, 4])
      ✓ edge_index: torch.Size([2, 870])
      ✓ edge_attr: torch.Size([870, 5])
      ✓ hyperedge_index: torch.Size([2, 121800])
      ✓ hyperedge_attr: torch.Size([31465, 2])
      ✓ y (jet type): tensor([1])

✓ Data validation passed!
```

### Step 2: Run Quick Test

Test the model with dummy data first:

```bash
python quickstart.py
```

This verifies:
- ✓ Model architecture works
- ✓ Forward/backward passes
- ✓ Loss computation
- ✓ Generation works

### Step 3: Train on Your Data

Now train with your actual data:

```bash
python train.py \
    --config config.yaml \
    --data-path /path/to/train_data.pt \
    --val-data-path /path/to/val_data.pt \
    --save-dir checkpoints \
    --log-dir runs
```

**What you'll see:**
```
Loading training data from: train_data.pt
Loaded 8000 jets
Sample jet structure:
  x (particles): torch.Size([30, 3])
  edge_index: torch.Size([2, 870])
  edge_attr: torch.Size([870, 5])
  hyperedge_index: torch.Size([2, 121800])
  hyperedge_attr: torch.Size([31465, 2])
  y (jet type): tensor([1])

Model parameters: 10.23M
============================================================
Epoch 1/200
============================================================
  Loss: 1.234, KL: 0.567
Train Loss: 1.234 | Val Loss: 1.456
✓ Saved best model
```

### Step 4: Monitor Training

In another terminal:

```bash
tensorboard --logdir runs
```

Open http://localhost:6006

**What to watch:**
- Total loss should decrease
- Particle loss most important (largest weight)
- KL loss starts small, gradually increases
- After ~50 epochs, losses should stabilize

### Step 5: Generate Jets

After training (or while training continues):

```bash
python generate.py \
    --checkpoint checkpoints/best_model.pt \
    --output generated_jets.pt \
    --num-samples 1000 \
    --batch-size 32 \
    --gpu
```

**Output:**
```
Using device: cuda
GPU: NVIDIA GeForce GTX 1650 Ti
Loaded model from epoch 100

Jet type distribution: Quark=0.33, Gluon=0.33, Top=0.34

Generating 1000 jets...
100%|██████████| 32/32 [00:03<00:00, 9.5it/s]

Generated Jet Statistics
============================================================
Jet Type Distribution:
  Quark: 333 (33.3%)
  Gluon: 333 (33.3%)
  Top:   334 (33.4%)

Particles per Jet:
  Mean: 28.5
  Std:  4.2
  Min:  15
  Max:  45

Saved 1000 generated jets in PyG format
```

### Step 6: Evaluate Quality

Compare generated jets to real ones:

```bash
python evaluate.py \
    --real-data /path/to/test_data.pt \
    --generated-data generated_jets.pt \
    --plot \
    --plot-dir plots
```

**Output:**
```
Loading jets from test_data.pt
Loading jets from generated_jets.pt
Loaded 1000 real jets
Loaded 1000 generated jets

Computing Wasserstein distances...

Evaluation Results
============================================================
Wasserstein Distances:
  pt              : 0.012345
  eta             : 0.008765
  phi             : 0.009876
  mass            : 0.015432
  n_particles     : 0.003210

Structural Metrics:
  Mean particles (real): 29.8
  Mean particles (gen):  28.5
  
  Jet Type Distribution:
    Quark  - Real: 0.330, Gen: 0.333
    Gluon  - Real: 0.340, Gen: 0.333
    Top    - Real: 0.330, Gen: 0.334

Saved plots to plots/
```

Check the plots in `plots/` directory:
- `particle_features.png` - Distribution comparison
- `n_particles.png` - Particle count comparison

## Memory Management for GTX 1650Ti

Your GTX 1650Ti has **4GB VRAM**. The default config is optimized for this:

```yaml
# config.yaml (already optimized)
training:
  batch_size: 4                     # Small batch
  gradient_accumulation_steps: 8   # Effective batch = 32
  mixed_precision: true            # FP16 saves memory
```

### If you still get OOM:

1. **Reduce batch size**:
```yaml
training:
  batch_size: 2
  gradient_accumulation_steps: 16
```

2. **Monitor VRAM**:
```bash
watch -n 0.5 nvidia-smi
```

3. **Close other GPU applications** (browser, etc.)

## Expected Performance

On your **i5 10th gen + GTX 1650Ti**:

| Dataset Size | Time/Epoch | Total Training |
|-------------|------------|----------------|
| 1,000 jets  | ~35 sec    | ~2 hours (200 epochs) |
| 5,000 jets  | ~3 min     | ~10 hours |
| 10,000 jets | ~6 min     | ~20 hours |

**Recommendations:**
- Start with 1,000-2,000 jets for testing
- Train for 100 epochs initially
- Check generation quality
- Then train full dataset for 200 epochs

## Common Issues & Solutions

### Issue 1: CUDA Out of Memory
```
RuntimeError: CUDA out of memory
```
**Solution:**
```yaml
# Reduce batch size in config.yaml
training:
  batch_size: 2  # or even 1
```

### Issue 2: Slow Training
```
30 particles × 870 edges is dense
```
**Solution:** This is normal! Your data is rich. If too slow:
```yaml
# Reduce model size
model:
  particle_hidden: 48  # from 64
  edge_hidden: 32      # from 48
```

### Issue 3: Generated Particle Count Wrong
```
Generated: 15 particles, Real: 30 particles
```
**Solution:**
```yaml
# Increase topology loss weight
training:
  loss_weights:
    topology: 5.0  # from 1.0
```

## Complete Example Session

```bash
# 1. Validate data
python validate_data.py train.pt
# ✓ Data validation passed!

# 2. Quick test
python quickstart.py
# ✓ All tests passed!

# 3. Start training
python train.py --data-path train.pt --val-data-path val.pt

# 4. (In new terminal) Monitor
tensorboard --logdir runs

# 5. (After 50+ epochs) Generate
python generate.py --checkpoint checkpoints/best_model.pt --output gen.pt

# 6. Evaluate
python evaluate.py --real-data test.pt --generated-data gen.pt --plot
```

## Loading Generated Jets

```python
import torch

# Load generated jets
data_list = torch.load('generated_jets.pt')

# Access first jet
jet = data_list[0]

# Particle features
particles = jet.x  # [N, 3]
pt = particles[:, 0]
eta = particles[:, 1]
phi = particles[:, 2]

# Edges
edge_connections = jet.edge_index  # [2, M]
edge_features = jet.edge_attr      # [M, 5]

# Hyperedges
hyperedge_connections = jet.hyperedge_index  # [2, K]
hyperedge_features = jet.hyperedge_attr      # [L, 2]

# Jet type
jet_type = jet.y.item()  # 0, 1, or 2
```

## Tips for Best Results

1. **Start Small**: Test with 1000 jets first
2. **Monitor Losses**: All losses should decrease
3. **KL Annealing**: First 50 epochs, KL weight increases slowly
4. **Patience**: Good results need 100-200 epochs
5. **Checkpoints**: Saved every 20 epochs

## Next Steps

✅ Validate your data  
✅ Run quick test  
✅ Start training  
✅ Monitor with TensorBoard  
✅ Generate after 50+ epochs  
✅ Evaluate quality  
✅ Tune if needed  

**You're all set!** 🚀

Your data format is perfect, the model is ready, and you have all the tools to train and generate jets!
