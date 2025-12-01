# HyperVAE Model Architecture

**Comprehensive Technical Documentation for Bipartite Hypergraph Variational Autoencoder (HyperVAE) for Jet Generation**

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture Components](#architecture-components)
3. [Encoder Architecture](#encoder-architecture)
4. [Decoder Architecture](#decoder-architecture)
5. [Loss Functions](#loss-functions)
6. [Training Strategies](#training-strategies)
7. [Memory Optimization](#memory-optimization)
8. [Configuration Parameters](#configuration-parameters)
9. [Mathematical Formulations](#mathematical-formulations)
10. [Design Decisions and Innovations](#design-decisions-and-innovations)

---

## Overview

### High-Level Architecture

HyperVAE is a **Variational Autoencoder (VAE)** designed for generating particle physics jets as **bipartite hypergraphs**. The architecture combines:

1. **L-GATr (Lorentz Group Attention)**: Physics-preserving transformations for particle 4-vectors
2. **Bipartite Graph Structure**: Particle-level + edge/hyperedge features
3. **Squared Distance Chamfer Loss**: Stable gradient flow for set generation
4. **Multi-Task Learning**: Particle, edge, hyperedge, and jet-level losses
5. **KL Annealing**: Prevents posterior collapse in VAE training

### Pipeline Flow

```
Input Jets (PyG Data)
    ↓
┌─────────────────────────────────────────────┐
│           ENCODER                           │
│  Particles → L-GATr → Scalars               │
│  Edges → MLP                                │
│  Hyperedges → MLP                           │
│  Cross-Attention Fusion                     │
│  Global Pooling                             │
│  → μ (mean), σ² (variance)                  │
└─────────────────────────────────────────────┘
    ↓
  z ~ N(μ, σ²)  [Reparameterization Trick]
    ↓
┌─────────────────────────────────────────────┐
│           DECODER                           │
│  z → Latent Expansion                       │
│  → Topology (Gumbel-Softmax)                │
│  → Particle Features (L-GATr)               │
│  → Edge Features (MLP)                      │
│  → Hyperedge Features (MLP)                 │
│  → Jet Features (MLP)                       │
└─────────────────────────────────────────────┘
    ↓
Generated Jets
```

### Key Innovation: Physics-Preserving Generation

Unlike standard generative models, HyperVAE respects **Lorentz symmetry** (special relativity) through L-GATr:

- **Input**: Particle 4-vectors `[E, px, py, pz]` (energy + 3-momentum)
- **Transform**: Geometric Algebra (GA) multivectors preserving spacetime symmetries
- **Output**: Physically consistent particles satisfying `E² = p² + m²`

This ensures generated jets obey fundamental physics laws.

---

## Architecture Components

### 1. Data Representation

Jets are represented as **PyTorch Geometric (PyG) Data objects** with:

```python
Data(
    # Particle Information
    particle_x=[N_particles, 4],      # 4-vectors: [E, px, py, pz] (normalized)
    n_particles=int,                  # Number of particles in jet
    
    # Graph Structure
    edge_index=[2, N_edges],          # Pairwise particle connections
    edge_attr=[N_edges, 5],           # Edge features [ln_delta, ln_kt, ln_z, ln_m2, feat5]
    hyperedge_index=[2, N_connections],  # Particle-hyperedge incidence
    hyperedge_x=[N_hyperedges, 2],    # Hyperedge features [3pt_eec, 4pt_eec]
    n_hyperedges=int,                 # Number of hyperedges
    
    # Jet-Level Information
    y=[num_features],                 # [jet_type, jet_pt, jet_eta, jet_mass, ...]
    jet_type=int,                     # 0=quark, 1=gluon, 2=top
)
```

**Feature Normalization**:
- 4-momenta: Z-score normalized `(x - μ) / σ`
- Edge features: Log-transformed (e.g., `ln_kt`, `ln_delta`)
- Hyperedge features: Energy-Energy Correlators (EECs)

### 2. Model Dimensions

From `config.yaml`:

```yaml
model:
  particle_hidden: 128        # Particle MLP hidden dim
  edge_hidden: 96             # Edge MLP hidden dim
  hyperedge_hidden: 64        # Hyperedge MLP hidden dim
  jet_hidden: 128             # Jet-level MLP hidden dim
  latent_dim: 32              # Latent space dimension
  max_particles: 60           # Maximum particles per jet
  
  particle_features: 4        # [E, px, py, pz]
  edge_features: 5            # 5 pairwise features
  hyperedge_features: 1       # EEC values
  jet_features: 3             # [jet_pt, jet_eta, jet_mass]
  
  encoder:
    cross_attention_heads: 4  # Heads for particle↔edge/hyperedge cross-attention
    # ... (see config.yaml for full encoder config)
```

**Design Rationale**:
- `latent_dim=32`: Compressed representation (128D particles → 32D latent)
- `particle_hidden=128`: Largest hidden dim (particles are primary signal)
- `edge/hyperedge_hidden < particle_hidden`: Auxiliary features need less capacity
- `max_particles=60`: Accommodates 99% of jets (typical: 10-50 particles)

---

## Encoder Architecture

### Overall Flow

```
Particles [N, 4] ──→ L-GATr Blocks (3 layers) ──→ Scalars [N, 32]
                                                      ↓
Edges [M, 5] ──────→ MLP (5 layers) ──────────────→ Scalars [M, 96]
                                                      ↓
Hyperedges [K, 2] ─→ MLP (3 layers) ──────────────→ Scalars [K, 64]
                                                      ↓
                                            Cross-Attention Fusion
                                                      ↓
                                            Global Pooling (mean)
                                                      ↓
                                            MLP Projection
                                                      ↓
                                    μ [batch, 32], logvar [batch, 32]
```

### Component Details

#### 1. L-GATr Particle Encoder

**Configuration** (from `config.yaml`):
```yaml
encoder:
  lgatr:
    in_mv_channels: 1           # Input: 1 channel of 4-vectors
    out_mv_channels: 0          # Output: scalars only (no multivectors)
    hidden_mv_channels: 16      # Hidden GA multivector channels
    
    in_s_channels: 0            # No auxiliary scalar inputs
    out_s_channels: 32          # Output: 32 scalar features per particle
    hidden_s_channels: 32       # Hidden scalar channels
    
    num_blocks: 2               # 2 L-GATr transformer blocks
    
    attention:
      num_heads: 4              # 4-head attention
      multi_query: true         # Memory-efficient (critical for 4GB VRAM)
      dropout_prob: 0.25        # Attention dropout
      head_scale: true          # Stabilize training
```

**L-GATr Block Structure**:
```
4-vectors [N, 4] → Embed to GA multivectors [N, 16 channels]
    ↓
┌──────────────────────────────┐
│  L-GATr Block 1              │
│  • Multi-head Attention      │  ← Equivariant attention on multivectors
│  • Equivariant MLP (2 layers)│  ← Geometric algebra operations
│  • Residual + LayerNorm      │
└──────────────────────────────┘
    ↓
┌──────────────────────────────┐
│  L-GATr Block 2              │
│  • Multi-head Attention      │
│  • Equivariant MLP (2 layers)│
│  • Residual + LayerNorm      │
└──────────────────────────────┘
    ↓
Project to scalars [N, 32]
```

**Why L-GATr?**
- **Lorentz Equivariance**: Outputs transform correctly under boosts/rotations
- **Geometric Algebra**: Natural representation for spacetime vectors
- **Physics Prior**: Enforces `E² - p² = m²` (mass-shell condition)

#### 2. Edge Feature Encoder

```python
MLP(
    in_features=5,        # [ln_delta, ln_kt, ln_z, ln_m2, feat5]
    hidden=96,
    layers=5,
    activation=ReLU,
    dropout=0.25,
    residual=True         # Skip connections every layer
)
→ Output: [M, 96]
```

**Edge Features**:
- `ln_delta`: Log angular distance between particles
- `ln_kt`: Log transverse momentum of pair
- `ln_z`: Energy sharing fraction
- `ln_m2`: Log invariant mass squared

#### 3. Hyperedge Feature Encoder

```python
MLP(
    in_features=2,        # [3pt_eec, 4pt_eec] (if separate) or [eec] (if merged)
    hidden=64,
    layers=3,
    activation=ReLU,
    dropout=0.25,
    residual=True
)
→ Output: [K, 64]
```

**Hyperedge Features**:
- `3pt_eec`: 3-point Energy-Energy Correlator (triangle structures)
- `4pt_eec`: 4-point Energy-Energy Correlator (quad structures)

#### 4. Cross-Attention Fusion (Optional)

Fuses particle, edge, and hyperedge representations via cross-attention:

```python
# Particle attends to edges
particle_features = CrossAttention(
    query=particle_features,  # [N, 32]
    key=edge_features,        # [M, 96]
    value=edge_features,
    num_heads=4
)

# Particle attends to hyperedges
particle_features = CrossAttention(
    query=particle_features,  # [N, 32]
    key=hyperedge_features,   # [K, 64]
    value=hyperedge_features,
    num_heads=4
)
```

This enriches particle representations with graph structure information.

#### 5. Global Pooling

```python
# Pool each feature type across jet
particle_global = particle_features.mean(dim=0)      # [32]
edge_global = edge_features.mean(dim=0)              # [96]
hyperedge_global = hyperedge_features.mean(dim=0)    # [64]

# Concatenate
jet_embedding = concat([particle_global, edge_global, hyperedge_global])  # [192]
```

#### 6. Latent Projection

```python
# Project to latent distribution parameters
mu = Linear(192 → 32)(jet_embedding)        # [batch, 32]
logvar = Linear(192 → 32)(jet_embedding)    # [batch, 32]

# Clamp for numerical stability
mu = clamp(mu, -10, 10)
logvar = clamp(logvar, -10, 10)
```

**Why Clamp?**
- Prevents `exp(logvar)` overflow (exp(50) = ∞)
- Keeps gradients stable during training

---

## Decoder Architecture

### Overall Flow

```
z [batch, 32] + jet_type [batch] ──→ Latent Expansion
                                          ↓
                              ┌───────────┴───────────┐
                              ↓                       ↓
                    Topology Decoder          Feature Decoders
                    (Gumbel-Softmax)          (Parallel MLPs)
                              ↓                       ↓
                    Masks: particle_mask      L-GATr → Particles [B, 60, 4]
                                              MLP → Jet Features [B, 3]
```

Note: The decoder generates only particles and jet features. Edge and hyperedge features 
are pre-computed from the dataset and used during encoding and loss computation, but are 
not generated by the decoder.

### Component Details

#### 1. Latent Expansion

```python
# Expand latent code to per-particle features
z_expanded = z.unsqueeze(1).expand(-1, max_particles, -1)  # [B, 60, 32]

# Condition on jet type (one-hot encoding)
jet_type_onehot = one_hot(jet_type, num_classes=3)  # [B, 3] for quark/gluon/top
jet_type_broadcast = jet_type_onehot.unsqueeze(1).expand(-1, max_particles, -1)  # [B, 60, 3]

# Concatenate
decoder_input = concat([z_expanded, jet_type_broadcast], dim=-1)  # [B, 60, 35]
```

**Why Condition on Jet Type?**
- Quark jets: Fewer particles, narrow cone
- Gluon jets: More particles, wider cone
- Top jets: Complex substructure (W boson decay)

#### 2. Topology Decoder (Gumbel-Softmax)

Predicts which particles/hyperedges are "real" vs "padding":

```python
# Particle existence logits
particle_logits = MLP(decoder_input)  # [B, 60, 1] → logit per particle

# Gumbel-Softmax sampling (differentiable discrete sampling)
particle_mask = gumbel_softmax(
    particle_logits, 
    temperature=τ,     # Annealed from 5.0 → 0.5 over training
    hard=False         # Soft during training, hard at inference
)  # [B, 60] ∈ [0, 1]
```

**Gumbel-Softmax Mathematics**:

During training (soft):
```
g_i ~ Gumbel(0, 1)
mask_i = exp((logit_i + g_i) / τ) / Σ_j exp((logit_j + g_j) / τ)
```

At inference (hard):
```
mask_i = 1 if i == argmax(logits) else 0
```

**Temperature Annealing**:
- High τ (5.0): Soft, exploratory assignments (training start)
- Low τ (0.5): Sharp, confident assignments (training end)
- Schedule: `τ = max(0.5, 5.0 × 0.98^epoch)`

#### 3. L-GATr Particle Decoder

**Configuration**:
```yaml
decoder:
  lgatr:
    in_mv_channels: 1           # Seed 4-vectors from noise
    out_mv_channels: 1          # Output: 1 channel of 4-vectors
    hidden_mv_channels: 16      # Hidden GA channels
    
    in_s_channels: 32           # Scalar features from latent
    out_s_channels: 0           # No auxiliary scalar outputs
    hidden_s_channels: 32       # Hidden scalar channels
    
    num_blocks: 2               # 2 L-GATr blocks (symmetric with encoder)
```

**Decoding Process**:
```
Latent scalars [B, 60, 32] → Embed to GA multivectors [B, 60, 16 channels]
    ↓
┌──────────────────────────────┐
│  L-GATr Block 1              │
│  • Multi-head Attention      │
│  • Equivariant MLP           │
│  • Residual + LayerNorm      │
└──────────────────────────────┘
    ↓
┌──────────────────────────────┐
│  L-GATr Block 2              │
│  • Multi-head Attention      │
│  • Equivariant MLP           │
│  • Residual + LayerNorm      │
└──────────────────────────────┘
    ↓
Project to 4-vectors [B, 60, 4]
    ↓
Apply particle_mask → [B, 60, 4] (masked particles → padding)
```

#### 4. Jet Feature Decoder

Predicts jet-level observables `[jet_pt, jet_eta, jet_mass]`:

```python
MLP(
    in_features=32,       # From latent z
    hidden=128,
    layers=2,
    activation=ReLU,
    dropout=0.25
)
→ Output: [B, 3]
```

**Why Predict Jet Features?**
- Acts as **soft constraint**: Particles must sum to correct jet 4-momentum
- **Physics consistency**: Generated jets have realistic total pT, η, mass
- **Auxiliary signal**: Provides additional gradient for particle generation

---

## Loss Functions

### Total Loss Formulation

```
L_total = w_particle × L_particle 
        + w_edge_dist × L_edge_distribution
        + w_hyperedge_dist × L_hyperedge_distribution
        + w_jet × L_jet
        + w_consistency × L_consistency
        + β(epoch) × w_kl × L_KL
```

Where:
- `w_particle = 12000.0`: Particle loss weight (primary objective)
- `w_edge_dist = 1.0`, `w_hyperedge_dist = 1.0`: Distribution matching weights
- `w_jet = 3000.0`: Jet feature weight
- `w_consistency = 3000.0`: Local-global consistency weight
- `w_kl = 0.3`: KL divergence weight
- `β(epoch)`: KL annealing schedule

### 1. Particle Reconstruction Loss (Chamfer Distance)

**Primary loss** for particle generation. Chamfer distance is a **permutation-invariant set loss** that matches predicted particles to true particles.

#### Mathematical Formulation

For each jet `k`:

```
L_chamfer(k) = L_true→pred(k) + L_pred→true(k)
```

Where:
```
L_true→pred = Σ_{i ∈ true} w_i × min_{j ∈ pred} D(true_i, pred_j)
L_pred→true = Σ_{j ∈ pred} w_j × min_{i ∈ true} D(pred_j, true_i)
```

#### Distance Metric: Weighted 4D Euclidean

For 4-vectors `[E, px, py, pz]`:

```
D²(i,j) = w_energy × (E_i - E_j)² + (px_i - px_j)² + (py_i - py_j)² + (pz_i - pz_j)²
```

**Configuration**:
```yaml
loss_config:
  particle_distance_metric: "euclidean_4d"
  w_energy: 2.0              # Weight energy component 2× momentum
  use_squared_distance: true  # Use D² instead of D (stronger gradients)
```

**Why Squared Distance?**

| Metric     | Formula     | Gradient ∂L/∂x                    | Properties                          |
|------------|-------------|-----------------------------------|-------------------------------------|
| Squared    | D² = x²     | ∂D²/∂x = 2x (linear)              | Strong, stable gradients            |
| Euclidean  | D = √(x²)   | ∂D/∂x = x/(2√x²) = 1/(2√x)        | Vanishing gradients for large x     |

For normalized features `x ∈ [-3, 3]`:
- Squared: Gradient ≈ 6 for x=3 (strong pull)
- Euclidean: Gradient ≈ 0.17 for x=3 (weak pull) → **gradient vanishing**

**Implementation**:

```python
def _compute_particle_distance(particles1, particles2):
    """
    Compute pairwise distances [N1, N2].
    
    Args:
        particles1: [N1, 4] - [E, px, py, pz]
        particles2: [N2, 4] - [E, px, py, pz]
    """
    # Extract components with broadcasting
    E1 = particles1[:, 0].unsqueeze(1)    # [N1, 1]
    px1 = particles1[:, 1].unsqueeze(1)   # [N1, 1]
    py1 = particles1[:, 2].unsqueeze(1)   # [N1, 1]
    pz1 = particles1[:, 3].unsqueeze(1)   # [N1, 1]
    
    E2 = particles2[:, 0].unsqueeze(0)    # [1, N2]
    px2 = particles2[:, 1].unsqueeze(0)   # [1, N2]
    py2 = particles2[:, 2].unsqueeze(0)   # [1, N2]
    pz2 = particles2[:, 3].unsqueeze(0)   # [1, N2]
    
    # Compute squared differences [N1, N2]
    delta_E_sq = (E1 - E2) ** 2
    delta_px_sq = (px1 - px2) ** 2
    delta_py_sq = (py1 - py2) ** 2
    delta_pz_sq = (pz1 - pz2) ** 2
    
    # Weighted squared distance
    dist_sq = w_energy * delta_E_sq + delta_px_sq + delta_py_sq + delta_pz_sq
    
    # Squared distance (NO sqrt)
    dist = torch.clamp(dist_sq, min=1e-8)
    
    return dist  # [N1, N2]
```

#### pT Weighting (Optional)

Weight particles by transverse momentum `pT = √(px² + py²)`:

```
w_i = (pT_i)^α / Σ_k (pT_k)^α
```

- `α = 0`: Uniform weighting (all particles equal)
- `α = 1`: Linear pT weighting (high-pT particles important)
- `α = 2`: Quadratic weighting (very high-pT dominated)

**Configuration**:
```yaml
loss_config:
  use_pt_weighting: false    # Disabled by default
  pt_weight_alpha: 1.0       # Linear weighting if enabled
```

**Why pT Weighting?**
- High-pT particles form jet core (most important for physics)
- Low-pT particles are soft radiation (less critical)
- **Trade-off**: Can neglect low-pT structure → disabled for this work

#### Bidirectionality

Chamfer distance is **bidirectional**:

1. **True → Pred**: Ensures all true particles are covered
   - Prevents model from missing particles
   
2. **Pred → True**: Penalizes spurious/hallucinated particles
   - Prevents model from generating extra particles

Without bidirectionality, model could generate many fake particles (low pred→true loss) while ignoring some true particles.

### 2. Edge Feature Loss (Auxiliary)

**Distribution matching** for pairwise edge features:

```
L_edge = MSE(mean(pred_edges), mean(true_edges))
```

**Why Distribution Matching?**
- Edges and hyperedges are derived from particles (not independent)
- Match statistical distributions of observables, not individual features
- Uses pre-computed dataset features as ground truth
- Wasserstein distance for robust distribution comparison

**Implementation**:
```python
def _distribution_loss(generated_particles, dataset_features):
    """
    Compute distribution matching loss between generated and true features.
    Generated features are computed on-the-fly from particles.
    """
    # Compute features from generated particles
    gen_features = compute_eec_features(generated_particles)
    
    # Wasserstein distance between distributions
    loss = wasserstein_distance(gen_features, dataset_features)
    return loss
```

**Loss Weight**:
```
L_edge_dist_total = 1.0 × L_edge_distribution
L_hyperedge_dist_total = 1.0 × L_hyperedge_distribution
```

Much smaller weights than particle loss (12000.0), providing weak distributional guidance.

### 3. Consistency Loss

Enforces local-global consistency between particle kinematics and jet-level properties:

```
L_consistency = MSE(sum(particles), jet_features)
```

This ensures generated particles sum to physically consistent jet properties.

**Loss Weight**:
```
L_consistency_total = 3000.0 × L_consistency
```

### 4. Jet Feature Loss

Enforces **jet-level constraints** via weighted MSE:

```
L_jet = w_pt × MSE(pred_pt, true_pt)
      + w_eta × MSE(pred_eta, true_eta)
      + w_mass × MSE(pred_mass, true_mass)
```

**Configuration**:
```yaml
loss_config:
  jet_pt_weight: 2.5        # Highest weight (most important observable)
  jet_eta_weight: 0.75      # Medium weight
  jet_mass_weight: 2.0      # High weight (critical for substructure)
  jet_n_constituents_weight: 0.0  # DISABLED (wrong scale)
```

**Why These Weights?**
- **jet_pt**: Primary energy scale → highest weight
- **jet_mass**: Discriminates jet types (W/Z/Higgs tagging) → high weight
- **jet_eta**: Well-measured but less critical → medium weight
- **N_constituents**: Integer counts have wrong scale (values ~20-40) → disabled

**Implementation**:
```python
def _jet_feature_loss(batch, pred_jet_features, topology):
    # Extract from y tensor [jet_type, jet_pt, jet_eta, jet_mass, ...]
    true_pt = batch.y[:, 1]     # [B]
    true_eta = batch.y[:, 2]    # [B]
    true_mass = batch.y[:, 3]   # [B]
    
    # Extract predictions
    pred_pt = pred_jet_features[:, 0]    # [B]
    pred_eta = pred_jet_features[:, 1]   # [B]
    pred_mass = pred_jet_features[:, 2]  # [B]
    
    # Weighted MSE
    loss = (2.5 * F.mse_loss(pred_pt, true_pt) +
            0.75 * F.mse_loss(pred_eta, true_eta) +
            2.0 * F.mse_loss(pred_mass, true_mass))
    
    return loss
```

**Total Jet Loss Weight**:
```
L_jet_total = 3000.0 × L_jet
```

### 5. KL Divergence Loss

**VAE Regularization**: Prevents posterior collapse, enables generation.

#### Mathematical Formulation

KL divergence between approximate posterior `q(z|x) = N(μ, σ²)` and prior `p(z) = N(0, I)`:

```
D_KL(q||p) = -½ Σ_d [1 + log(σ_d²) - μ_d² - σ_d²]
           = ½ Σ_d [μ_d² + σ_d² - log(σ_d²) - 1]
```

Where:
- `μ = encoder.mean`: [batch, 32]
- `σ² = exp(logvar)`: [batch, 32]
- `d`: Latent dimension index

#### Free Bits Regularization

Only penalize KL if it exceeds threshold:

```
KL_d^{free} = max(0, KL_d - λ_free)
```

**Configuration**:
```yaml
training:
  kl_free_bits: 3.0    # Bits per dimension
```

**Why Free Bits?**
- Early training: KL can collapse to 0 (deterministic encoding)
- Free bits allow each dimension to carry ≥ 3 bits of information
- Prevents aggressive compression that loses information

#### KL Annealing

Gradually increase KL weight over training:

```
β(epoch) = min(1.0, epoch / T_warmup) × β_max
```

**Configuration**:
```yaml
training:
  kl_warmup_epochs: 15     # Warmup over 15 epochs
  kl_max_weight: 1.0       # Maximum multiplier
  loss_weights:
    kl_divergence: 0.3     # Base weight
```

**Annealing Schedule**:
```
Epoch 0-15:  β = epoch / 15  → linearly increase from 0 to 1
Epoch 15+:   β = 1.0          → full KL penalty
```

**Total KL Weight**:
```
L_KL_total = β(epoch) × 0.3 × L_KL
```

**Why Anneal?**
1. **Early epochs (β ≈ 0)**: Focus on reconstruction, learn meaningful latent representations
2. **Late epochs (β = 1)**: Enforce prior matching, enable random sampling at generation time

Without annealing, KL dominates early training → posterior collapse → poor reconstruction.

#### Implementation

```python
def _kl_divergence(mu, logvar):
    # Clamp for numerical stability
    mu = torch.clamp(mu, min=-10.0, max=10.0)
    logvar = torch.clamp(logvar, min=-10.0, max=10.0)
    
    # KL per dimension
    kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    
    # Clamp to prevent explosion
    kl_per_dim = torch.clamp(kl_per_dim, min=0.0, max=100.0)
    
    # Free bits
    if free_bits > 0:
        kl_per_dim = torch.clamp(kl_per_dim - free_bits, min=0.0)
    
    # Sum over dimensions, mean over batch
    kl_loss = torch.sum(kl_per_dim, dim=-1).mean()
    
    return kl_loss
```

---

## Training Strategies

### 1. Mixed Precision Training

**Disabled** due to L-GATr library compatibility issues:

```yaml
training:
  mixed_precision: false   # torch.autocast not compatible with lgatr
```

**Why Disable?**
- L-GATr library uses `torch.is_autocast_enabled()` checks
- Mixed precision (FP16) causes numerical instability in GA operations
- Memory savings not critical for small model (fits in 4GB VRAM)

**Alternative**: Use gradient accumulation for effective larger batches.

### 2. Gradient Accumulation

Simulate larger batch size without memory overhead:

```yaml
training:
  batch_size: 64                  # Physical batch
  gradient_accumulation_steps: 4  # Accumulate over 4 batches
  # Effective batch = 64 × 4 = 256
```

**Implementation**:
```python
for i, batch in enumerate(train_loader):
    output = model(batch)
    loss = model.compute_loss(batch, output, epoch)['total']
    
    # Scale loss by accumulation steps
    loss = loss / gradient_accumulation_steps
    loss.backward()
    
    # Update every N steps
    if (i + 1) % gradient_accumulation_steps == 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        optimizer.step()
        optimizer.zero_grad()
```

**Benefits**:
- Larger effective batch → more stable gradients
- Fits in 4GB VRAM (GTX 1650 Ti compatible)
- Equivalent to batch_size=256 on larger GPU

### 3. Gradient Clipping

Prevent gradient explosion:

```yaml
training:
  gradient_clip: 3.0    # Clip gradient norm to 3.0
```

**Why Increased to 3.0?**
- Squared distance metric has **linear gradients** (∂d²/∂x = 2x)
- More stable than Euclidean (∂√d²/∂x = 1/(2√x) can spike)
- Can allow stronger gradients without instability
- Previous value (0.75) was too conservative

**Implementation**:
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
```

### 4. Learning Rate Schedule

**Cosine annealing with warmup**:

```yaml
training:
  learning_rate: 0.0001     # Initial LR
  warmup_epochs: 10         # Linear warmup
  min_lr: 0.00001          # Minimum LR after cosine decay
```

**Schedule**:
```
Epoch 0-10:  LR = 0.0001 × (epoch / 10)      [Linear warmup]
Epoch 10+:   LR = 0.00001 + 0.5 × (0.0001 - 0.00001) × (1 + cos(π × t / T))
             [Cosine decay from 0.0001 to 0.00001]
```

**Why Cosine?**
- Smooth decay (no sudden drops)
- Exploration early (high LR)
- Fine-tuning late (low LR)
- Better than step decay for VAEs

### 5. Temperature Annealing (Gumbel-Softmax)

Control sharpness of topology predictions:

```yaml
training:
  initial_temperature: 5.0   # Soft (exploration)
  final_temperature: 0.5     # Sharp (exploitation)
  temperature_decay: 0.98    # Exponential decay
```

**Schedule**:
```
τ(epoch) = max(0.5, 5.0 × 0.98^epoch)
```

**Temperature Effect**:
- **τ = 5.0** (start): Soft masks, many particles "partially exist"
- **τ = 1.0** (mid): Moderate sharpness
- **τ = 0.5** (end): Sharp masks, binary particle existence

**Why Anneal?**
- Early: Explore different topologies (soft assignments)
- Late: Commit to specific topology (hard assignments)
- Smooth transition prevents training instability

### 6. Regularization

#### Dropout
```yaml
encoder:
  dropout: 0.25              # Standard dropout
decoder:
  dropout: 0.3               # Higher dropout (combat overfitting)
  feature_proj_dropout: 0.3  # Dropout in feature projections
```

**Higher Dropout in Decoder**: Decoder has more capacity (generates features) → more prone to overfitting.

#### Weight Decay (L2 Regularization)
```yaml
training:
  weight_decay: 0.001   # L2 penalty on weights
```

Prevents weights from growing too large.

#### Latent Noise (Variational Dropout)
```yaml
training:
  latent_noise: 0.01    # Gaussian noise std in latent space
```

**Implementation**:
```python
z = mu + std * eps           # Standard reparameterization
z = z + torch.randn_like(z) * 0.01  # Additional noise
```

**Benefits**:
- Smooths latent space
- Prevents overfitting to specific latent codes
- Improves interpolation quality

### 7. Early Stopping

```yaml
training:
  patience: 50         # Stop if no improvement for 50 epochs
  min_delta: 0.001     # Minimum improvement threshold
```

Monitors validation loss and stops if plateau detected.

---

## Memory Optimization

### Strategies for 4GB VRAM (GTX 1650 Ti)

1. **Multi-Query Attention** (L-GATr):
   ```yaml
   attention:
     multi_query: true   # Share key/value across heads
   ```
   Reduces memory: `O(n_heads × d_k)` → `O(d_k)`

2. **Gradient Accumulation**:
   - Physical batch: 64 (fits in 4GB)
   - Effective batch: 256 (simulated)

3. **Gradient Checkpointing** (optional):
   ```python
   from torch.utils.checkpoint import checkpoint
   output = checkpoint(layer, input)  # Recompute activations instead of storing
   ```

4. **Auxiliary Loss Gating**:
   ```python
   if self.training and self.use_auxiliary_losses:
       edge_loss = ...  # Only compute during training
   ```
   Saves memory during validation/inference.

5. **Efficient Collation**:
   ```python
   # PyG batch format: variable-size graphs → single large graph
   # Memory: O(Σ_i n_i) instead of O(batch_size × max_n)
   ```

### Memory Profile

Approximate memory usage (batch_size=64):

| Component                  | Memory    |
|---------------------------|-----------|
| Model parameters          | ~50 MB    |
| Input batch (PyG)         | ~100 MB   |
| Activations (forward)     | ~500 MB   |
| Gradients (backward)      | ~500 MB   |
| Optimizer state (Adam)    | ~100 MB   |
| **Total**                 | **~1.25 GB** |

Leaves ~2.75 GB for OS/CUDA overhead → comfortable fit in 4GB.

---

## Configuration Parameters

### Model Configuration

```yaml
model:
  # Dimensions
  particle_hidden: 128        # Particle MLP hidden dim
  edge_hidden: 96             # Edge MLP hidden dim
  hyperedge_hidden: 64        # Hyperedge MLP hidden dim
  jet_hidden: 128             # Jet MLP hidden dim
  latent_dim: 32              # Latent space dimension
  max_particles: 60           # Maximum particles per jet
  
  # Feature dimensions
  particle_features: 4        # [E, px, py, pz]
  edge_features: 5            # [2pt_EEC, ln_delta, ln_kT, ln_z, ln_m2] (from dataset)
  hyperedge_features: 1       # EEC values (from dataset)
  jet_features: 3             # [jet_pt, jet_eta, jet_mass]
```

### Training Configuration

```yaml
training:
  # Optimization
  batch_size: 64
  gradient_accumulation_steps: 4
  epochs: 100
  learning_rate: 0.0001
  weight_decay: 0.001
  
  # Regularization
  latent_noise: 0.01
  gradient_clip: 3.0
  
  # Scheduler
  scheduler: "cosine"
  warmup_epochs: 10
  min_lr: 0.00001
  
  # Loss weights
  loss_weights:
    particle_features: 12000.0
    edge_distribution: 1.0
    hyperedge_distribution: 1.0
    jet_features: 3000.0
    consistency: 3000.0
    kl_divergence: 0.3
  
  # KL annealing
  kl_warmup_epochs: 15
  kl_free_bits: 3.0
  kl_max_weight: 1.0
  
  # Temperature annealing
  initial_temperature: 5.0
  final_temperature: 0.5
  temperature_decay: 0.98
```

### Loss Configuration

```yaml
training:
  loss_config:
    # Particle loss
    particle_loss_type: "chamfer"
    particle_distance_metric: "euclidean_4d"
    use_squared_distance: true
    use_pt_weighting: false
    pt_weight_alpha: 1.0
    
    # Distance weights
    w_energy: 2.0
    
    # Evaluation
    evaluation_metric: "chamfer"
    
    # Jet feature weights
    jet_pt_weight: 2.5
    jet_eta_weight: 0.75
    jet_mass_weight: 2.0
    jet_n_constituents_weight: 0.0
```

---

## Mathematical Formulations

### 1. VAE Objective (ELBO)

```
log p(x) ≥ E_q(z|x)[log p(x|z)] - D_KL(q(z|x) || p(z))
         = Reconstruction Loss - KL Divergence
```

**Maximize** this lower bound (ELBO) = **minimize** negative ELBO:

```
L_VAE = -E_q(z|x)[log p(x|z)] + β × D_KL(q(z|x) || p(z))
      = L_reconstruction + β × L_KL
```

### 2. Reparameterization Trick

Sample `z ~ q(z|x) = N(μ(x), σ²(x))` differentiably:

```
ε ~ N(0, I)
z = μ(x) + σ(x) ⊙ ε
```

Where `σ(x) = exp(0.5 × logvar(x))`.

Gradient flows through `μ` and `σ`, not through random `ε`.

### 3. Chamfer Distance (Set Matching)

For sets `X = {x_1, ..., x_n}` and `Y = {y_1, ..., y_m}`:

```
Chamfer(X, Y) = 1/n Σ_{x ∈ X} min_{y ∈ Y} ||x - y||²
              + 1/m Σ_{y ∈ Y} min_{x ∈ X} ||y - x||²
```

Bidirectional: measures coverage (X→Y) and precision (Y→X).

### 4. Gumbel-Softmax

Sample categorical distribution `z ~ Cat(π)` differentiably:

```
g_i ~ Gumbel(0, 1) = -log(-log(U_i)), U_i ~ Uniform(0, 1)
z_i = exp((log π_i + g_i) / τ) / Σ_j exp((log π_j + g_j) / τ)
```

As `τ → 0`, converges to `argmax(π)` (hard categorical).

### 5. KL Divergence (Gaussian)

For `q = N(μ_q, σ_q²)` and `p = N(μ_p, σ_p²)`:

```
D_KL(q || p) = log(σ_p / σ_q) + (σ_q² + (μ_q - μ_p)²) / (2σ_p²) - 1/2
```

For `p = N(0, I)` (standard normal):

```
D_KL(q || p) = -1/2 Σ_d [1 + log σ_d² - μ_d² - σ_d²]
```

### 6. Lorentz Transformation

4-vector `p = (E, px, py, pz)` transforms under boost `v` as:

```
E'  = γ(E - v·p)
p'∥ = γ(p∥ - vE)
p'⊥ = p⊥
```

Where `γ = 1/√(1 - v²)` and `p∥`, `p⊥` are parallel/perpendicular to boost.

**L-GATr ensures**: If `p → decoder → p'`, then `Λp → decoder → Λp'` for any Lorentz transformation `Λ`.

---

## Design Decisions and Innovations

### 1. Why L-GATr?

**Problem**: Standard neural networks don't respect physics symmetries.

**Example**: If you boost the jet (change reference frame), naive NN predictions won't transform correctly.

**Solution**: L-GATr enforces **Lorentz equivariance** via Geometric Algebra:
- Inputs/outputs are 4-vectors in spacetime
- Operations preserve symmetries (boosts, rotations)
- Guarantees physical consistency

**Result**: Generated jets obey special relativity automatically.

### 2. Why Squared Distance?

**Problem**: Euclidean Chamfer loss has **vanishing gradients** for far predictions.

**Math**:
```
Euclidean:  ∂√(x²)/∂x = x / (2√x²) ≈ 1/(2√x)   → 0 as x → ∞
Squared:    ∂(x²)/∂x = 2x                       → ∞ as x → ∞
```

**Effect**:
- Euclidean: Particle at x=10 has gradient ≈ 0.05 (barely moves)
- Squared: Particle at x=10 has gradient = 20 (strong pull)

**Result**: Faster convergence, no training plateau.

**Trade-off**: Different scale than literature (must document clearly).

### 3. Why Distribution Losses?

**Problem**: Edges/hyperedges are derived from particles (not independent).

**Approach**:
1. **Primary**: Particle Chamfer loss (12000.0 weight)
2. **Distribution matching**: Edge/hyperedge Wasserstein distance (1.0 / 1.0 weights)

**Rationale**:
- Particles determine everything else (4-momentum conservation)
- Distribution losses provide weak statistical guidance
- Matches distributions of observables, not individual features

**Result**: Better particle arrangements with physically meaningful distributions.

### 4. Why KL Annealing?

**Problem**: High KL weight early → posterior collapse → bad reconstruction.

**Solution**: Start with β=0 (no KL), gradually increase to β=1.

**Effect**:
- Early: Model learns meaningful latent representations (reconstruction focus)
- Late: Latent space matches prior N(0, I) (generation capability)

**Result**: Best of both worlds (good reconstruction + good generation).

### 5. Why Free Bits?

**Problem**: KL can collapse to 0 (encoder outputs δ(z - z₀) instead of distribution).

**Solution**: Only penalize KL if it exceeds threshold:
```
L_KL = max(0, KL - λ_free)
```

**Effect**: Allows latent dimensions to carry information before being penalized.

**Result**: Prevents deterministic encoding (preserves VAE stochasticity).

### 6. Why Bipartite Hypergraph?

**Standard Graph**: Particles (nodes) + pairwise edges

**Bipartite Hypergraph**: Particles (nodes) + edges + **hyperedges** (3+particle clusters)

**Advantage**:
- Captures higher-order correlations (3-point, 4-point structures)
- Jets have complex substructure (e.g., W→qq' decay → 2-prong)
- Hyperedges encode EECs (Energy-Energy Correlators) → physics-motivated

**Result**: Richer representation, better generation quality.

### 7. Why Condition on Jet Type?

**Observation**: Different jet types have distinct structures:
- **Quark jets**: Narrow, few particles
- **Gluon jets**: Wide, many particles
- **Top jets**: Substructure (W boson decay)

**Implementation**: One-hot encode jet type, concatenate with latent code.

**Result**: Single model generates all jet types (multi-task learning).

---

## Summary

HyperVAE combines:
1. **L-GATr**: Physics-preserving transformations
2. **VAE**: Probabilistic latent space for generation
3. **Chamfer Loss**: Set-based matching for particles
4. **Distribution Matching**: Wasserstein distance for observables
5. **Annealing**: KL + temperature schedules for stable training

**Key Innovation**: Squared distance Chamfer loss prevents gradient vanishing, leading to faster and more stable training.

**Result**: Generates high-quality jets respecting both data distribution and physical laws.

---

## References

- **L-GATr Paper**: [Lorentz Group Equivariant Autoencoders](https://arxiv.org/abs/2306.03266)
- **L-GATr Library**: https://heidelberg-hepml.github.io/lgatr/
- **VAE Tutorial**: [Auto-Encoding Variational Bayes (Kingma & Welling, 2013)](https://arxiv.org/abs/1312.6114)
- **Chamfer Distance**: [Learning Representations and Generative Models for 3D Point Clouds](https://arxiv.org/abs/1707.02392)
- **Gumbel-Softmax**: [Categorical Reparameterization with Gumbel-Softmax](https://arxiv.org/abs/1611.01144)

---

**Document Version**: 1.0  
**Last Updated**: November 4, 2025  
**Corresponding Config**: `config.yaml`
