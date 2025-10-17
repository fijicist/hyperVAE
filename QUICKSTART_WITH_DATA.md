# Quick Start with Your .pt Dataset

Your dataset format is perfect! It's already in PyTorch Geometric format:
```python
Data(
    x=[30, 3],              # 30 particles with (pt, eta, phi)
    edge_index=[2, 870],    # 870 edges between particles
    edge_attr=[870, 5],     # 5D edge features
    y=[1],                  # Jet type label
    hyperedge_index=[2, 121800],   # Hyperedge connections
    hyperedge_attr=[31465, 2]      # 2D hyperedge features (3pt/4pt EEC)
)
```

**Note:** If your data has 4 features (pt, eta, phi, m), the model will automatically use only the first 3 (pt, eta, phi).

## Training with Your Data

### 1. Quick Test First

Test that the data loader works:
```bash
python quickstart.py
```

### 2. Train with Your Dataset

```bash
# Train on your data
python train.py \
    --config config.yaml \
    --data-path /path/to/your/train_data.pt \
    --val-data-path /path/to/your/val_data.pt \
    --save-dir checkpoints \
    --log-dir runs
```

### 3. Monitor Training

In another terminal:
```bash
tensorboard --logdir runs
```

Open http://localhost:6006 in your browser.

### 4. Generate New Jets

After training:
```bash
python generate.py \
    --checkpoint checkpoints/best_model.pt \
    --output generated_jets.pt \
    --num-samples 1000 \
    --batch-size 32 \
    --gpu
```

### 5. Evaluate Generated Jets

```bash
python evaluate.py \
    --real-data /path/to/your/test_data.pt \
    --generated-data generated_jets.pt \
    --plot \
    --plot-dir plots
```

## Loading Your Data in Python

```python
import torch
from torch_geometric.data import Data

# Load your dataset
data_list = torch.load('your_data.pt')

# Inspect first jet
jet = data_list[0]
print(f"Particles: {jet.x.shape}")           # [30, 4]
print(f"Edges: {jet.edge_index.shape}")      # [2, 870]
print(f"Edge features: {jet.edge_attr.shape}") # [870, 5]
print(f"Hyperedge index: {jet.hyperedge_index.shape}")  # [2, 121800]
print(f"Hyperedge features: {jet.hyperedge_attr.shape}") # [31465, 2]
print(f"Jet type: {jet.y}")                  # [1]

# Access particle features
pt = jet.x[:, 0]    # Transverse momentum
eta = jet.x[:, 1]   # Pseudorapidity
phi = jet.x[:, 2]   # Azimuthal angle
```

## Expected Training Time

On **GTX 1650Ti (4GB)**:
- With your dataset size, training will take:
  - ~30-40 sec/epoch for 1000 jets
  - ~5-10 min/epoch for 10,000 jets
  - Recommended: 100-200 epochs

## Memory Management

If you get OOM (Out of Memory) errors:

1. **Reduce batch size** in `config.yaml`:
```yaml
training:
  batch_size: 2  # Reduce from 4
  gradient_accumulation_steps: 16  # Increase to maintain effective batch
```

2. **Limit particles** if your jets are very large:
```yaml
model:
  max_particles: 100  # Reduce from 150
```

3. **Check VRAM usage**:
```bash
watch -n 0.5 nvidia-smi
```

## Sample Training Command

For a typical training run on your system:

```bash
python train.py \
    --config config.yaml \
    --data-path train_jets.pt \
    --val-data-path val_jets.pt \
    --save-dir checkpoints \
    --log-dir runs/experiment1
```

## What to Expect

### During Training
You should see:
```
Epoch 1/200
  Particle loss: 0.1234
  Edge loss: 0.0567
  Hyperedge loss: 0.0234
  Topology loss: 0.0123
  KL loss: 1.2345
  Total loss: 1.5678
```

### Loss should:
- Decrease steadily over first 20-30 epochs
- Stabilize after 50-100 epochs
- KL loss will start small (0.001 weight) and gradually increase

### After Training
Best model saved at: `checkpoints/best_model.pt`

## Generation

Generated jets will match your input format:
```python
# Load generated jets
generated = torch.load('generated_jets.pt')

# Each jet has the same structure as your input
gen_jet = generated[0]
print(gen_jet.x.shape)              # [N, 4] - variable N
print(gen_jet.edge_attr.shape)      # [M, 5]
print(gen_jet.hyperedge_attr.shape) # [K, 2]
```

## Tips for Your Dataset

1. **Particle Count**: Your jets have ~30 particles - this is perfect!
   - Model `max_particles: 150` can easily handle this
   - Consider reducing to 50 if memory is tight

2. **Edge Density**: 870 edges for 30 particles is very dense
   - This is fine for memory (edges are efficiently handled)
   - Model will learn this density pattern

3. **Hyperedges**: 31,465 hyperedge features is rich
   - Model has capacity for this
   - May need to increase `hyperedge_hidden` if quality is low

## Troubleshooting

### Problem: "CUDA out of memory"
**Solution**: Reduce `batch_size` to 2 or even 1

### Problem: "RuntimeError: tensors must be on same device"
**Solution**: Already handled in code, but if it occurs, check that CUDA is available

### Problem: Training loss not decreasing
**Solution**: 
1. Check learning rate (try 0.0001)
2. Increase KL warmup epochs
3. Verify data is normalized

### Problem: Generated jets have wrong particle count
**Solution**: Increase `topology` loss weight from 1.0 to 2.0 or 5.0

## Next Steps

1. ✅ Run `python quickstart.py` to verify setup
2. ✅ Train with your data: `python train.py --data-path your_data.pt`
3. ✅ Monitor with TensorBoard
4. ✅ Generate jets after ~50 epochs
5. ✅ Evaluate quality
6. ✅ Tune hyperparameters if needed

Good luck with your training! 🚀
