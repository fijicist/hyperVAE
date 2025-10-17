# Bipartite HyperVAE for Jet Generation

A memory-efficient **Variational Autoencoder** for generating jets (quark, gluon, and top) with hypergraph structure, optimized for **NVIDIA GTX 1650Ti (4GB VRAM)**.

## Features

- **Bipartite Graph Representation**: Converts hypergraphs to bipartite graphs with particles and hyperedges
- **Lorentz-Equivariant Attention (L-GATr)**: Physics-aware attention mechanism for particles
- **Edge-Aware Transformer**: Handles particle pairs with 5D edge features
- **Multi-Feature Generation**:
  - Node features: `pt`, `η`, `φ` (3D)
  - Edge features: `ln Δ`, `ln kT`, `ln z`, `ln m²`, `feat5` (5D)
  - Hyperedge features: 3-pt EEC, 4-pt EEC (2D)
- **Memory Optimized**: 
  - Gradient accumulation (batch size 4 × 8 steps)
  - Mixed precision (FP16) training
  - ~10M parameters

## Architecture

```
Encoder:
  Particles (pt,η,φ) → L-GATr (3 layers) → 64D
  Edges (5 features) → TransformerConv (2 layers) → 48D
  Hyperedges (2 EEC) → L-GATr (2 layers) → 32D
  → Bipartite Cross-Attention → Latent (128D)

Decoder:
  Latent + Jet Type → MLP Expander
  → Topology Decoder (Gumbel-Softmax)
  → Parallel Feature Decoders:
     - Particles: L-GATr → (pt,η,φ)
     - Edges: GATv2Conv → (5 features)
     - Hyperedges: L-GATr → (2 EEC)
```

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Or with conda
conda create -n hypervae python=3.10
conda activate hypervae
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install torch-geometric torch-scatter torch-sparse
pip install -r requirements.txt
```

## Data Format

Expected `.pt` format (PyTorch Geometric Data list):
```python
# List of PyG Data objects
[
    Data(
        x=[N_particles, 3],              # Node features: pt, eta, phi
        edge_index=[2, N_edges],         # Edge connectivity
        edge_attr=[N_edges, 5],          # Edge features: ln_delta, ln_kt, ln_z, ln_m2, feat5
        hyperedge_index=[2, N_hyperedge_connections],  # Hyperedge connectivity
        hyperedge_attr=[N_hyperedges, 2],  # Hyperedge features: 3pt_eec, 4pt_eec
        y=[1]                            # Jet type: 0=quark, 1=gluon, 2=top
    ),
    ...
]
```

## Usage

### 1. Training

**Option A: Single data file (auto-split 80/10/10)**
```bash
python train.py \
    --config config.yaml \
    --data-path data/all_jets.pt \
    --save-test-indices \
    --save-dir checkpoints \
    --log-dir runs
```

**Option B: Separate train/val files**
```bash
python train.py \
    --config config.yaml \
    --data-path data/train_jets.pt \
    --val-data-path data/val_jets.pt \
    --save-dir checkpoints \
    --log-dir runs
```

**Option C: Test with dummy data**
```bash
python train.py --config config.yaml
```

See [USAGE_SINGLE_FILE.md](USAGE_SINGLE_FILE.md) for detailed splitting options.

Monitor training with TensorBoard:
```bash
tensorboard --logdir runs
```

### 2. Generation

Generate jets from trained model:
```bash
python generate.py \
    --checkpoint checkpoints/best_model.pt \
    --output generated_jets.pt \
    --num-samples 10000 \
    --batch-size 32 \
    --quark-frac 0.33 \
    --gluon-frac 0.33 \
    --top-frac 0.34 \
    --gpu
```

### 3. Evaluation

Evaluate generated jets against real data:
```bash
python evaluate.py \
    --real-data data/test_jets.pt \
    --generated-data generated_jets.pt \
    --max-jets 10000 \
    --plot \
    --plot-dir plots
```

This computes:
- Wasserstein distances for all features
- Structural metrics (particle count, jet type distribution)
- Distribution plots

## Configuration

Edit `config.yaml` to customize:

```yaml
model:
  particle_hidden: 64      # Particle embedding dimension
  edge_hidden: 48          # Edge embedding dimension
  hyperedge_hidden: 32     # Hyperedge embedding dimension
  latent_dim: 128          # Latent space dimension
  max_particles: 150       # Maximum particles per jet

training:
  batch_size: 4            # Physical batch size (1650Ti: 4)
  gradient_accumulation_steps: 8  # Effective batch: 32
  learning_rate: 0.0001
  epochs: 200
  mixed_precision: true    # FP16 for memory efficiency
  
  loss_weights:
    particle_features: 10.0
    edge_features: 5.0
    hyperedge_features: 3.0
    topology: 1.0
    kl_divergence: 0.001
```

## Memory Optimization Tips

For **1650Ti (4GB VRAM)**:
- Keep `batch_size: 4`
- Use `gradient_accumulation_steps: 8` for effective batch size 32
- Enable `mixed_precision: true`
- Set `max_particles: 150` (adjust based on your data)

For **more VRAM** (6-8GB):
- Increase `batch_size: 8-16`
- Reduce `gradient_accumulation_steps: 2-4`
- Increase hidden dimensions if needed

## Model Testing

Test individual components:

```bash
# Test data loading
python data/bipartite_dataset.py

# Test L-GATr layers
python models/lgat_layers.py

# Test encoder
python models/encoder.py

# Test decoder
python models/decoder.py

# Test full model
python models/hypervae.py
```

## Project Structure

```
hyperVAE/
├── config.yaml                 # Configuration file
├── requirements.txt           # Dependencies
├── README.md                  # This file
│
├── data/
│   └── bipartite_dataset.py   # Dataset and data loading
│
├── models/
│   ├── lgat_layers.py         # Lorentz-equivariant layers
│   ├── encoder.py             # VAE encoder
│   ├── decoder.py             # VAE decoder
│   └── hypervae.py           # Complete VAE model
│
├── train.py                   # Training script
├── generate.py                # Inference/generation script
└── evaluate.py                # Evaluation metrics
```

## Loss Function

```python
Total Loss = 10.0 × MSE(particle features)
           + 5.0 × MSE(edge features)
           + 3.0 × MSE(hyperedge features)
           + 1.0 × BCE(topology)
           + 0.001 × KL divergence (with annealing)
```

## Physics Constraints

The model enforces:
- `pt > 0` (Softplus activation)
- `η ∈ [-2.5, 2.5]` (Tanh scaling)
- `φ ∈ [-π, π]` (Tanh scaling)

## Performance

On **GTX 1650Ti**:
- Training: ~30-40 seconds/epoch (1000 samples)
- Generation: ~300 jets/second
- Memory usage: ~3.5GB VRAM

## Citation

If you use this code, please cite:
```bibtex
@software{bipartite_hypervae,
  title={Bipartite HyperVAE for Jet Generation},
  year={2025},
  author={Your Name}
}
```

## License

MIT License

## References

- L-GATr: Lorentz Geometric Algebra Transformer
- Energy-Energy Correlators (EEC) for jet substructure
- Bipartite graph representation for hypergraphs
