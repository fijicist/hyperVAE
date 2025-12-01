# HyperVAE: Lorentz-Equivariant Hypergraph VAE for Jet Generation

**HyperVAE** is a physics-informed deep generative model for synthesizing high-energy physics jets using Variational Autoencoders with Lorentz-equivariant transformations. The model represents jets as bipartite hypergraphs with particle nodes, pairwise edges encoding 2-point Energy-Energy Correlators (EEC), and hyperedges capturing higher-order N-point correlations. By leveraging L-GATr (Lorentz Group Attention) layers, HyperVAE aims to generate jets that respect special relativity and fundamental spacetime symmetries.

Designed for consumer-grade GPUs, HyperVAE employs memory-optimized training strategies including gradient accumulation, mixed-precision computation. Deep generative models offer the potential for significantly faster jet generation compared to traditional physics simulators, which could be useful for data augmentation, detector studies, and physics analyses.

---

## Model Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            INPUT: Real Jet Data                             │
│                  PyG Graphs: Particles + Edges + Hyperedges                 │
└────────────────────────────────────┬────────────────────────────────────────┘
                                     │
                                     ▼
         ┌───────────────────────────────────────────────────┐
         │              ENCODER (q_φ)                        │
         │                                                   │
         │  ┌─────────────────────────────────────────┐     │
         │  │   Particle Encoding (L-GATr)            │     │
         │  │   • Lorentz-equivariant attention       │     │
         │  │   • Input: [E, px, py, pz]              │     │
         │  │   • Output: Scalar features             │     │
         │  └─────────────────┬───────────────────────┘     │
         │                    │                             │
         │  ┌─────────────────┴───────────────────────┐     │
         │  │   Edge/Hyperedge Encoding (MLPs)        │     │
         │  │   • 2-point EEC features                │     │
         │  │   • N-point EEC features                │     │
         │  └─────────────────┬───────────────────────┘     │
         │                    │                             │
         │  ┌─────────────────┴───────────────────────┐     │
         │  │   Cross-Attention Fusion                │     │
         │  │   • Particle ↔ Edge interactions        │     │
         │  └─────────────────┬───────────────────────┘     │
         │                    │                             │
         │  ┌─────────────────┴───────────────────────┐     │
         │  │   Global Pooling (mean/max)             │     │
         │  │   • Aggregates to jet-level embedding   │     │
         │  └─────────────────┬───────────────────────┘     │
         │                    │                             │
         │  ┌─────────────────┴───────────────────────┐     │
         │  │   Latent Projection                     │     │
         │  │   → μ(x), log(σ²(x))                    │     │
         │  └─────────────────────────────────────────┘     │
         └──────────────────────┬────────────────────────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │  LATENT SPACE (z)     │
                    │  z ~ N(μ, σ²)         │
                    │  [Reparameterization] │
                    └───────────┬───────────┘
                                │
                                ▼
         ┌───────────────────────────────────────────────────┐
         │              DECODER (p_θ)                        │
         │                                                   │
         │  ┌─────────────────────────────────────────┐     │
         │  │   Latent Expansion + Conditioning       │     │
         │  │   • Broadcast z to N particles          │     │
         │  │   • Concat jet type embedding           │     │
         │  └─────────────────┬───────────────────────┘     │
         │                    │                             │
         │  ┌─────────────────┴───────────────────────┐     │
         │  │   Topology Decoder (Gumbel-Softmax)     │     │
         │  │   • Predict particle multiplicity       │     │
         │  │   • Differentiable sampling             │     │
         │  └─────────────────┬───────────────────────┘     │
         │                    │                             │
         │  ┌─────────────────┼───────────────────────┐     │
         │  │   Parallel Feature Decoders:            │     │
         │  │                 │                       │     │
         │  │   ┌─────────────┴──────────┐            │     │
         │  │   │ Particle Decoder       │            │     │
         │  │   │ (L-GATr)               │            │     │
         │  │   │ → [E, px, py, pz]      │            │     │
         │  │   └────────────────────────┘            │     │
         │  │                                         │     │
         │  │   ┌────────────────────────┐            │     │
         │  │   │ Jet Feature Head       │            │     │
         │  │   │ → [pt, eta, mass]      │            │     │
         │  │   └────────────────────────┘            │     │
         │  └─────────────────────────────────────────┘     │
         └───────────────────────┬───────────────────────────┘
                                 │
                                 ▼
                ┌────────────────────────────────┐
                │   LOSS COMPUTATION             │
                │                                │
                │  • Chamfer Distance (particles)│
                │  • Distribution Loss (EEC)     │
                │  • Jet Feature Loss            │
                │  • Consistency Loss            │
                │  • KL Divergence (annealed)    │
                └────────────────────────────────┘
```

**Key Components:**
- **L-GATr Layers**: Ensure Lorentz equivariance (boosts, rotations)
- **Bipartite Structure**: Particles + pre-computed edge/hyperedge observables for encoding
- **Gumbel-Softmax**: Differentiable topology learning
- **Distribution Matching**: Generated particles compared with dataset EEC distributions

---

## Quick Start

### Prerequisites
- **Python**: 3.8+ (3.10 recommended)
- **GPU**: CUDA-capable (optional, but recommended)
- **Memory**: 4GB VRAM minimum (GTX 1650 Ti or better)

### 1. Installation

```bash
# Clone repository
git clone https://github.com/fijicist/hyperVAE.git
cd hyperVAE

# Automated setup (recommended)
bash setup.sh
```

The setup script will:
- Auto-detect your Python version and CUDA version
- Install PyTorch with correct CUDA support (or CPU-only)
- Install PyTorch Geometric with matching wheels
- Install L-GATr, FastJet, and physics libraries
- Verify all dependencies
- Run a quickstart test

**Manual installation** (if setup.sh fails):
```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

---

### 2. Generate Graph Dataset

Before training, preprocess raw jet data into PyG graph format:

```bash
# Generate graphs from JetNet dataset
python graph_constructor.py
```

**What this does:**
- Loads jets from JetNet (quark, gluon, top jets)
- Transforms to 4-momentum: `(pt, η, φ) → (E, px, py, pz)`
- Computes Energy-Energy Correlators (2-point, 3-point, ...)
- Builds fully-connected graphs with edge/hyperedge features
- Applies global normalization (z-score or min-max)
- Saves to `data/real/graphs_pyg_particle__fully_connected_*.pt`

**Configuration**: Edit `GRAPH_CONSTRUCTION_CONFIG` in the file to customize:
```python
{
    'N': 18000,                    # Number of jets
    'normalization_method': 'zscore',  # 'zscore' or 'minmax'
    'eec_prop': [[2, 3], 200, (1e-4, 2)],  # EEC orders and binning
    'output_dir': './data/real/',
}
```

**Expected output:**
```
data/real/
├── graphs_pyg_particle__fully_connected_part_1.pt  (if > 85k jets)
└── graphs_pyg_particle__fully_connected_final.pt
```

---

### 3. Validate Data Format

Ensure generated graphs have correct structure:

```bash
python validate_data.py data/real/graphs_pyg_particle__fully_connected_final.pt
```

**Expected format:**
```python
Data(
    x=[N, 4],              # Normalized particle 4-momenta [E, px, py, pz]
    edge_index=[2, M],     # Fully-connected topology
    edge_attr=[M, 5],      # [2pt_EEC, ln_delta, ln_kT, ln_z, ln_m²]
    hyperedge_index=[N, K], # N-point incidence matrix (optional)
    hyperedge_attr=[K, F], # N-point EEC features (optional)
    y=[4],                 # [jet_type, log(pt), eta, log(mass)]
    particle_norm_stats={}, # For denormalization
    jet_norm_stats={},
    edge_norm_stats={},
    hyperedge_norm_stats={},
)
```

---

### 4. Train Model

Start training with default configuration:

```bash
python train.py \
    --config config.yaml \
    --data-path data/real/real_jets.pt \
    --save-dir checkpoints \
    --log-dir runs
```

**Training configuration** (`config.yaml`):
```yaml
model:
  latent_dim: 256           # Latent space size
  particle_hidden_dim: 512  # L-GATr hidden dimensions
  
training:
  batch_size: 2             # Per-GPU batch size
  gradient_accumulation_steps: 128  # Effective batch = 2×128 = 256
  num_epochs: 300
  learning_rate: 0.0001
  mixed_precision: true     # Enable AMP for memory efficiency
  precision_type: "fp16"    # "fp16" (Volta/Turing) or "bf16" (Ampere+)
  
loss_weights:
  particle: 12000.0         # Primary loss
  edge_distribution: 1.0    # 2-pt EEC Wasserstein
  hyperedge_distribution: 1.0  # N-pt EEC Wasserstein
  jet_features: 3000.0      # Soft constraint
  consistency: 3000.0       # Local-global physics
  kl_weight: 0.3            # Annealed during training
```

**Precision type selection:**
- **BF16** (`precision_type: "bf16"`): Ampere+ GPUs (RTX 30xx/40xx, A100)
  - Better numerical stability, wider dynamic range
  - Recommended for newer GPUs
- **FP16** (`precision_type: "fp16"`): Volta/Turing GPUs (V100, T4, RTX 20xx)
  - 2× memory savings, good for older architectures

**Check your GPU compatibility:**
```bash
python test_bf16.py  # Shows GPU compute capability and BF16 support
```

**Monitor training:**
```bash
# In another terminal
tensorboard --logdir runs
```

Visit `http://localhost:6006` to see:
- Loss curves (particle, edge, KL, total)
- Learning rate schedule
- Gradient norms

**Expected training time:**
- **GTX 1650 Ti (4GB)**: ~12 hours for 100 epochs (20k jets)

**Note**: Training time varies based on hardware and dataset size. The model is designed to be trainable on consumer hardware, though performance may vary.

**Checkpoints saved:**
- `checkpoints/best_model.pt` - Lowest validation loss
- `checkpoints/checkpoint_epoch_*.pt` - Regular snapshots

---

### 5. Generate Jets

Sample new jets from trained model:

```bash
python generate.py \
    --checkpoint checkpoints//best_model.pt \
    --output data/generated/generated_jets.pt \
    --num-samples 10000 \
    --batch-size 32 \
    --quark-frac 0.3333 \
    --gpu
```

**Options:**
- `--num-samples`: Number of jets to generate
- `--jet-type`: `q` (quark), `g` (gluon), `t` (top), or `None` (sample from prior)
- `--temperature`: Sampling temperature (default: final_temperature of gumbel-softmax, lower = more conservative)
- `--batch-size`: Generation batch size (default: 256)

**Output format:**
```python
# List of PyG Data objects (list)
[
    Data(x=[N₁, 4], y=[4]),  # x: particles [E,px,py,pz], y: [type,pt,eta,mass]
    Data(x=[N₂, 4], y=[4]),
    ...
]
```

**Note**: Generated jets contain only particles (`x`) and jet labels (`y`). No edge features since the decoder generates particles only.

**Denormalization** happens automatically using stored `*_norm_stats` from training data.

---

### 6. Evaluate Generated Jets

Compare generated jets to real data:

```bash
python evaluate.py \
    --real-data data/real/real_jets.pt \
    --generated-data data/generated/generated_jets.pt \
    --max-jets 10000 \
    --plot \
    --plot-dir plots/
```

**Metrics computed:**

**Wasserstein Distances** (measures distribution similarity, lower = better):
- **Particle features**: E, px, py, pz distributions
- **Jet features**: pt, η, mass distributions
- **Structural**: Number of particles per jet

**Structural Metrics**:
- Mean/std particle multiplicity (real vs generated)
- Jet type distribution (quark/gluon/top fractions)

**Example output:**
```
Wasserstein Distances:
  Particle Features:
    particle_E          : 0.347934
    particle_px         : 0.320851
    particle_py         : 0.422025
    particle_pz         : 0.137377

  Jet Features:
    jet_pt              : 0.571115
    jet_eta             : 0.127950
    jet_mass            : 0.064019

  n_particles         : 4.485000

Structural Metrics:
  Mean particles (real): 29.15
  Mean particles (gen):  33.63
  
  Jet Type Distribution:
    Quark  - Real: 0.342, Gen: 0.333
    Gluon  - Real: 0.328, Gen: 0.332
    Top    - Real: 0.331, Gen: 0.335
```

**Plot outputs** (if `--plot` flag used):
```
plots/
├── particle_features.png      # E, px, py, pz histograms
├── jet_features.png           # pt, η, mass, n_particles histograms
```

**Note**: Generated jets typically have only particle and jet features (no edges/hyperedges since decoder generates particles and jet features only).

## Advanced Usage

### Resume Training from Checkpoint

```bash
python train.py \
    --config config.yaml \
    --data-path data/real/graphs_pyg_particle__fully_connected_final.pt \
    --resume checkpoints/checkpoint_epoch_100.pt \
    --save-dir checkpoints \
    --log-dir runs
```

### Multi-GPU Training

```bash
# Use DataParallel (simple)
python train.py --config config.yaml --data-path data/... --num-gpus 2

# Or use DistributedDataParallel (faster)
torchrun --nproc_per_node=2 train.py --config config.yaml --data-path data/...
```

### Custom Dataset

To use your own jet data:

1. **Format as PyG graphs**: See `graph_constructor.py` as template
2. **Required fields**: `x`, `edge_index`, `edge_attr`, `y`
3. **Validate**: `python validate_data.py your_data.pt`
4. **Train**: `python train.py --data-path your_data.pt`

---

## Documentation

- **[PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md)**: High-level overview, physics motivation, architecture
- **[MODEL_ARCHITECTURE.md](MODEL_ARCHITECTURE.md)**: Detailed technical documentation
- **[requirements.txt](requirements.txt)**: Python dependencies with installation guide
- **Module docstrings**: Every Python file has comprehensive header documentation

---

## Troubleshooting

### Out of Memory (OOM)

```bash
# Reduce batch size in config.yaml
training:
  batch_size: 1
  gradient_accumulation_steps: 256
```

### CUDA Version Mismatch

```bash
# Reinstall PyTorch with correct CUDA
pip uninstall torch torch-geometric
bash setup.sh  # Auto-detects CUDA version
```

### Slow Training

- Enable mixed precision: `mixed_precision: true` in config
- Use BF16 on Ampere+ GPUs: `precision_type: "bf16"` (RTX 30xx/40xx, A100)
- Use FP16 on Volta/Turing GPUs: `precision_type: "fp16"` (V100, T4, RTX 20xx)
- Increase batch size if you have more VRAM
- Check GPU utilization: `nvidia-smi -l 1`

**Mixed Precision Guide:**
- **BF16 (bfloat16)**: Better numerical stability, wider dynamic range. Recommended for Ampere+ (SM 8.0+)
- **FP16 (float16)**: Faster on older GPUs, requires careful gradient scaling. Good for Volta/Turing
- Test your GPU: `python test_bf16.py`

### Poor Generation Quality

- Train longer (model may need 300+ epochs to converge)
- Check loss curves for plateaus or divergence
- Increase KL annealing warmup epochs for smoother training
- Verify data normalization statistics are computed correctly
- Consider adjusting loss weights if one component dominates


## Acknowledgments

- **L-GATr**: [Brehmer et al., "Geometric Algebra Transformers"](https://arxiv.org/abs/2305.18415)
- **JetNet**: [Kansal et al., "JetNet Dataset"](https://arxiv.org/abs/2106.11535)
- **PyTorch Geometric**: [Fey & Lenssen](https://arxiv.org/abs/1903.02428)
- **EnergyFlow**: [Komiske et al.](https://energyflow.network/)

---
