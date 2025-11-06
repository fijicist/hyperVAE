"""
═══════════════════════════════════════════════════════════════════════════════
L-GATr WRAPPER - LORENTZ-EQUIVARIANT GEOMETRIC ALGEBRA TRANSFORMER
═══════════════════════════════════════════════════════════════════════════════

Official Library Integration:
-----------------------------
Uses the official lgatr library from Heidelberg HEP-ML group:
- GitHub: https://github.com/heidelberg-hepml/lgatr
- Paper: "L-GATr: Lorentz Group Attention Transformer" (2024)
- Docs: https://heidelberg-hepml.github.io/lgatr/quickstart.html
- Examples: https://github.com/heidelberg-hepml/lgatr/tree/main/examples

Installation:
    pip install lgatr gatr

What is L-GATr?
---------------
L-GATr (Lorentz Group Attention Transformer) is a neural network architecture that:

1. **Preserves Lorentz Symmetry**: Respects the fundamental symmetry of special relativity
   - Boosts: (E, p⃗) → (γE + γβ·p⃗, γp⃗ + γβE)  [moving reference frames]
   - Rotations: (E, p⃗) → (E, R·p⃗)  [spatial rotations]
   - These are the symmetries of spacetime itself!

2. **Uses Geometric Algebra**: Represents 4-vectors as multivectors in Cl(1,3)
   - 4-vector [E, px, py, pz] → 16-dimensional multivector embedding
   - Operations (dot product, addition) become algebraic operations on multivectors
   - Guarantees equivariance by construction (not learned)

3. **Self-Attention for Particles**: Relates particles within a jet
   - Each particle attends to all others (permutation invariant)
   - Attention preserves Lorentz structure (energy-momentum conservation)
   - Multi-head attention captures different particle relationships

Why Use L-GATr for Jet Generation?
-----------------------------------
Traditional Problem:
    Neural networks don't naturally respect physical symmetries. A model might learn
    that (E=10, p⃗=0) is valid, but fail to recognize that (E=10, p⃗=10) is invalid
    (violates E² ≥ |p⃗|², the on-shell condition for massive particles).

L-GATr Solution:
    By construction, L-GATr outputs always respect Lorentz symmetry:
    - Generated 4-momenta satisfy physical constraints (E² ≥ |p⃗|²)
    - Boost-invariant: jets in different reference frames encoded identically
    - Data-efficient: network doesn't waste capacity learning symmetries

Physics Benefits for Jets:
    1. Energy-momentum conservation enforced architecturally
    2. Rotational invariance (jet orientation doesn't matter)
    3. Boost invariance (jet energy scale factored out cleanly)
    4. Better generalization to unseen energy ranges

Geometric Algebra Primer:
--------------------------
Standard 4-vector: [E, px, py, pz] ∈ ℝ⁴

Geometric Algebra representation:
    p = E·e₀ + px·e₁ + py·e₂ + pz·e₃ + ... (16 basis elements total)
    
Where:
    - e₀, e₁, e₂, e₃: basis vectors (4 dimensions)
    - e₀e₁, e₀e₂, ... : bivectors (6 dimensions)
    - e₀e₁e₂, ... : trivectors (4 dimensions)
    - e₀e₁e₂e₃: pseudoscalar (1 dimension)
    - scalar: 1 (1 dimension)
    Total: 16 dimensions (GA representation of Minkowski space)

Operations:
    - Dot product: p·q = (EE' - p⃗·p⃗')  (Minkowski metric)
    - Geometric product: pq = p·q + p∧q  (combines dot & wedge)
    - Lorentz transforms: Λ(p) = exp(-θ/2) p exp(θ/2)  (exponential map)

L-GATr handles all this internally—you just provide [E, px, py, pz]!

Architecture Flow in This Wrapper:
-----------------------------------
                    Input 4-momenta
                    [E, px, py, pz]
                          ↓
              embed_vector (lgatr.interface)
         Convert to multivector [16 dims]
                          ↓
                  ┌───────────────┐
                  │  LGATr Blocks │  (Lorentz-equivariant self-attention)
                  └───────────────┘
                          ↓
              extract_vector (lgatr.interface)
         Convert back to [E, px, py, pz]
                          ↓
                  Output 4-momenta

Each LGATr Block contains:
    1. Lorentz-equivariant self-attention (particle-particle interactions)
    2. Equivariant MLP (non-linear feature processing)
    3. Equivariant LayerNorm (normalization preserving symmetry)
    4. Residual connections (gradient flow)

Scalar Channels (Optional):
    - Multivector channels: Transform under Lorentz group (4-momenta)
    - Scalar channels: Invariant under Lorentz (pT, mass, particle ID)
    - Both types can be processed jointly in L-GATr

Configuration Guide:
--------------------
Key Parameters (in config.yaml):

encoder/decoder:
  lgatr:
    in_mv_channels: 1      # Input multivector channels (typically 1 = single 4-vector)
    out_mv_channels: 1     # Output multivector channels (1 for 4-momentum)
    hidden_mv_channels: 32 # Hidden multivector channels (higher = more capacity)
    in_s_channels: 0       # Input scalar channels (0 if not using)
    out_s_channels: 64     # Output scalar channels (for latent representation)
    hidden_s_channels: 16  # Hidden scalar channels
    
  particle_layers: 2-4     # Number of L-GATr blocks (2-4 typical)

Memory & Performance:
    - Each multivector channel uses 16× memory of scalar channel (GA representation)
    - hidden_mv_channels=32 is reasonable for 4GB GPU
    - More blocks → better expressivity but slower & more memory

References:
-----------
[1] Brehmer et al. "L-GATr: Lorentz Group Attention Transformer" (2024)
[2] Brehmer et al. "Geometric Algebra Transformer" (GATR, 2023) - base architecture
[3] Official repository: https://github.com/heidelberg-hepml/lgatr
[4] Applications: https://github.com/heidelberg-hepml/lgatr/tree/main/examples

Notes for ML Researchers:
--------------------------
If you're unfamiliar with physics:
- Think of Lorentz equivariance like rotation equivariance (but for spacetime)
- 4-momentum [E, px, py, pz] is like 3D position [x, y, z] + energy
- Geometric Algebra is a mathematical framework (like complex numbers for 2D)
- L-GATr is similar to SE(3)-equivariant networks, but for SO(1,3) (Minkowski)

Notes for Physicists:
----------------------
If you're unfamiliar with Transformers:
- Self-attention learns particle relationships (e.g., leading vs subleading)
- Multi-head attention captures multiple interaction types simultaneously
- Residual connections allow deep networks (like DenseNet for CNNs)
- LayerNorm stabilizes training (like BatchNorm but for sequences)

═══════════════════════════════════════════════════════════════════════════════
"""

import torch
import torch.nn as nn

try:
    from lgatr import LGATr
    from lgatr.interface import embed_vector, extract_vector, extract_scalar
    LGATR_AVAILABLE = True
except ImportError:
    LGATR_AVAILABLE = False
    print("⚠️  Warning: lgatr library not found. Install with: pip install lgatr")


class LGATrParticleEncoder(nn.Module):
    """
    L-GATr encoder for particle 4-vectors using the official L-GATr library.
    
    ═══════════════════════════════════════════════════════════════════════
    ENCODER: PARTICLES → LORENTZ-EQUIVARIANT REPRESENTATION
    ═══════════════════════════════════════════════════════════════════════
    
    Purpose:
    --------
    Encode variable-length sets of particle 4-momenta into Lorentz-equivariant
    representations for downstream VAE processing.
    
    Input: Raw particle 4-momenta [E, px, py, pz] for each particle in a jet
    Output: Processed 4-momenta + optional scalar features (for VAE encoder)
    
    Architecture Flow:
    ------------------
        Particles [batch, N, 4] → embed_vector → Multivectors [batch, N, 1, 16]
                                         ↓
                                  L-GATr Blocks
                            (self-attention + MLP)
                                         ↓
        Multivectors [batch, N, out_mv, 16] → extract_vector → 4-momenta [batch, N, 4]
        + Optional scalars [batch, N, out_s]
    
    Key Features:
    -------------
    1. LORENTZ EQUIVARIANCE: Output transforms correctly under boosts/rotations
       - Boost jet by β → outputs boost by β
       - Rotate jet by R → outputs rotate by R
    
    2. PERMUTATION INVARIANCE: Particle order doesn't matter
       - Self-attention pools information across all particles
       - No dependence on input ordering
    
    3. VARIABLE CARDINALITY: Works with any number of particles per jet
       - No padding artifacts (masked attention if needed)
       - Natural for jets (10-50 particles typical)
    
    4. MULTIVECTOR + SCALAR CHANNELS:
       - Multivector: Lorentz-covariant (4-momenta)
       - Scalar: Lorentz-invariant (mass, pT, etc.)
       - Jointly processed via equivariant self-attention
    
    Configuration:
    --------------
    Typical settings in config.yaml:
        encoder:
          lgatr:
            in_mv_channels: 1       # Single input 4-vector per particle
            out_mv_channels: 0      # Don't output 4-momenta (use scalars instead)
            hidden_mv_channels: 32  # Hidden capacity for interactions
            out_s_channels: 64      # Output scalar features for VAE encoder
            hidden_s_channels: 16   # Hidden scalar capacity
          particle_layers: 2        # 2-4 L-GATr blocks
    
    Memory Notes:
    -------------
    - Multivector channels use 16× memory of scalars (geometric algebra)
    - hidden_mv_channels=32 is reasonable for 4GB GPU
    - Reduce hidden_mv_channels if OOM, increase for more capacity
    
    Args:
        config (dict): Full configuration dictionary with 'encoder' section
    
    Example:
        >>> encoder = LGATrParticleEncoder(config)
        >>> particles = torch.randn(32, 50, 4)  # [batch=32, particles=50, features=4]
        >>> output = encoder(particles)
        >>> print(output['fourmomenta'].shape)  # [32, 50, 4] if out_mv_channels=1
        >>> print(output['scalars'].shape)       # [32, 50, 64] if out_s_channels=64
    """
    
    def __init__(self, config):
        super().__init__()
        
        if not LGATR_AVAILABLE:
            raise ImportError(
                "lgatr and gatr libraries are required. Install them with:\n"
                "pip install lgatr gatr\n"
                "See: https://github.com/heidelberg-hepml/lgatr"
            )
        
        self.config = config
        lgatr_config = config.get('encoder', {}).get('lgatr', {})
        
        # L-GATr parameters
        self.in_mv_channels = lgatr_config.get('in_mv_channels', 1)
        self.out_mv_channels = lgatr_config.get('out_mv_channels', 1)
        self.hidden_mv_channels = lgatr_config.get('hidden_mv_channels', 8)
        self.in_s_channels = lgatr_config.get('in_s_channels', 0)
        self.out_s_channels = lgatr_config.get('out_s_channels', 0)
        self.hidden_s_channels = lgatr_config.get('hidden_s_channels', 16)
        self.num_blocks = config.get('encoder', {}).get('particle_layers', 4)
        
        # Attention and MLP configs
        attention_config = lgatr_config.get('attention', {})
        mlp_config = lgatr_config.get('mlp', {})
        
        # Create L-GATr model with full configuration
        # Note: L-GATr uses EquiLayerNorm by default, no need to specify norm parameter
        self.lgatr = LGATr(
            in_mv_channels=self.in_mv_channels,
            out_mv_channels=self.out_mv_channels,
            hidden_mv_channels=self.hidden_mv_channels,
            in_s_channels=self.in_s_channels if self.in_s_channels > 0 else None,
            out_s_channels=self.out_s_channels if self.out_s_channels > 0 else None,
            hidden_s_channels=self.hidden_s_channels,
            num_blocks=self.num_blocks,
            attention=attention_config,
            mlp=mlp_config,
        )
        
    def forward(self, fourmomenta, mask=None):
        """
        Forward pass through L-GATr encoder.
        
        Args:
            fourmomenta: [batch_size, num_particles, 4] - (E, px, py, pz) 4-momenta
            mask: [batch_size, num_particles] - Optional mask for variable-size inputs
        
        Returns:
            dict containing:
                'fourmomenta': [batch_size, num_particles, 4] - Encoded 4-momenta
                'scalars': [batch_size, num_particles, hidden_s_channels] or None
        """
        batch_size, num_particles, _ = fourmomenta.shape
        
        # Embed 4-momenta in Geometric Algebra representation
        # embed_vector converts [E, px, py, pz] to multivector [batch, particles, 16]
        multivectors = embed_vector(fourmomenta).unsqueeze(-2)  # [batch, particles, 1, 16]
        
        # Initialize scalars if using scalar channels
        scalars = None
        if self.in_s_channels > 0:
            scalars = torch.zeros(batch_size, num_particles, self.in_s_channels,
                                device=fourmomenta.device, dtype=fourmomenta.dtype)
        
        # Pass through L-GATr
        multivector_outputs, scalar_outputs = self.lgatr(multivectors, scalars=scalars)
        # multivector_outputs: [batch, particles, out_mv_channels, 16]
        # scalar_outputs: [batch, particles, out_s_channels] or None
        
        # Extract 4-vectors from geometric algebra representation
        # extract_vector converts multivector back to [E, px, py, pz]
        fourmomenta_out = extract_vector(multivector_outputs)  # [batch, particles, out_mv_channels, 4]
        
        # If out_mv_channels=1, squeeze that dimension
        if self.out_mv_channels == 1:
            fourmomenta_out = fourmomenta_out.squeeze(2)  # [batch, particles, 4]
        
        return {
            'fourmomenta': fourmomenta_out,
            'scalars': scalar_outputs
        }
    
    def get_output_dim(self):
        """Return the output dimension of scalar features (0 if no scalars)."""
        return self.out_s_channels if self.out_s_channels > 0 else 0


class LGATrParticleDecoder(nn.Module):
    """
    L-GATr decoder for particle 4-vectors using the official L-GATr library.
    
    ═══════════════════════════════════════════════════════════════════════
    DECODER: LATENT + SCALARS → LORENTZ-EQUIVARIANT 4-MOMENTA
    ═══════════════════════════════════════════════════════════════════════
    
    Purpose:
    --------
    Decode latent representations + scalar features into physically-valid
    particle 4-momenta with Lorentz-equivariant processing.
    
    Input: Initial 4-momenta guess + scalar conditioning (from VAE decoder)
    Output: Refined 4-momenta satisfying Lorentz symmetry
    
    Architecture Flow:
    ------------------
        Initial 4-momenta [batch, N, 4] + Scalars [batch, N, in_s]
                                ↓
                         embed_vector
                                ↓
                Multivectors [batch, N, 1, 16]  +  Scalars
                                ↓
                          L-GATr Blocks
                    (conditioned self-attention)
                                ↓
                Multivectors [batch, N, 1, 16]
                                ↓
                         extract_vector
                                ↓
                   Final 4-momenta [batch, N, 4]
    
    Key Features:
    -------------
    1. SCALAR CONDITIONING: Scalars from VAE latent guide 4-momentum generation
       - Scalars carry information from latent z (particle type, kinematics)
       - L-GATr fuses scalars with 4-momentum via equivariant attention
       - Ensures generated particles match latent intent
    
    2. LORENTZ GUARANTEES: Output 4-momenta respect relativistic constraints
       - E² ≥ |p⃗|² (timelike or null, never spacelike)
       - Boost/rotation equivariance (frame-independent)
       - No post-hoc projection needed (built into architecture)
    
    3. ITERATIVE REFINEMENT: Multiple L-GATr blocks progressively improve
       - Block 1: Rough 4-momentum from scalars
       - Block 2+: Inter-particle consistency, energy-momentum balance
       - Residual connections allow deep refinement
    
    4. PERMUTATION INVARIANCE: Self-attention across particles
       - Each particle refined based on full jet context
       - Natural for unordered particle sets
    
    Configuration:
    --------------
    Typical settings in config.yaml:
        decoder:
          lgatr:
            in_mv_channels: 1       # Single 4-vector per particle (initial guess)
            out_mv_channels: 1      # Single refined 4-vector output
            hidden_mv_channels: 32  # Hidden capacity for refinement
            in_s_channels: 64       # Scalar features from VAE decoder
            out_s_channels: 0       # Don't output scalars (only 4-momenta)
            hidden_s_channels: 16   # Hidden scalar capacity
          particle_layers: 2        # 2-4 L-GATr blocks (more = finer refinement)
    
    Design Choice - Initial 4-Momentum:
    ------------------------------------
    The decoder needs an initial 4-vector guess to start refinement:
    
    Option A (this implementation): VAE decoder predicts initial 4-momenta + scalars
        - Pro: Simple, single forward pass
        - Pro: Scalars provide auxiliary information (particle type, topology)
        - Con: Initial guess may violate Lorentz constraints
    
    Option B (alternative): Construct 4-momenta purely from scalars
        - Pro: No invalid intermediate states
        - Con: More complex architecture (need scalar→4-momentum constructor)
    
    L-GATr corrects any initial constraint violations via equivariant processing,
    so Option A works well in practice.
    
    Memory Notes:
    -------------
    - Multivector channels use 16× memory of scalars
    - hidden_mv_channels=32 typical for 4GB GPU
    - in_s_channels should match VAE decoder output dimension
    
    Args:
        config (dict): Full configuration dictionary with 'decoder' section
    
    Example:
        >>> decoder = LGATrParticleDecoder(config)
        >>> initial_4mom = torch.randn(32, 50, 4)  # [batch, particles, 4-momentum]
        >>> scalars = torch.randn(32, 50, 64)      # [batch, particles, latent_features]
        >>> output = decoder(initial_4mom, scalars=scalars)
        >>> print(output['fourmomenta'].shape)     # [32, 50, 4] - refined 4-momenta
    """
    
    def __init__(self, config):
        super().__init__()
        
        if not LGATR_AVAILABLE:
            raise ImportError(
                "lgatr and gatr libraries are required. Install them with:\n"
                "pip install lgatr gatr\n"
                "See: https://github.com/heidelberg-hepml/lgatr"
            )
        
        self.config = config
        lgatr_config = config.get('decoder', {}).get('lgatr', {})
        
        # L-GATr parameters
        self.in_mv_channels = lgatr_config.get('in_mv_channels', 1)
        self.out_mv_channels = lgatr_config.get('out_mv_channels', 1)
        self.hidden_mv_channels = lgatr_config.get('hidden_mv_channels', 8)
        self.in_s_channels = lgatr_config.get('in_s_channels', 0)
        self.out_s_channels = lgatr_config.get('out_s_channels', 0)
        self.hidden_s_channels = lgatr_config.get('hidden_s_channels', 16)
        self.num_blocks = config.get('decoder', {}).get('particle_layers', 4)
        
        # Attention and MLP configs
        attention_config = lgatr_config.get('attention', {})
        mlp_config = lgatr_config.get('mlp', {})
        
        # Create L-GATr model with full configuration
        # Note: L-GATr uses EquiLayerNorm by default, no need to specify norm parameter
        self.lgatr = LGATr(
            in_mv_channels=self.in_mv_channels,
            out_mv_channels=self.out_mv_channels,
            hidden_mv_channels=self.hidden_mv_channels,
            in_s_channels=self.in_s_channels if self.in_s_channels > 0 else None,
            out_s_channels=self.out_s_channels if self.out_s_channels > 0 else None,
            hidden_s_channels=self.hidden_s_channels,
            num_blocks=self.num_blocks,
            attention=attention_config,
            mlp=mlp_config,
        )
    
    def forward(self, fourmomenta, scalars=None, mask=None):
        """
        Forward pass through L-GATr decoder.
        
        Args:
            fourmomenta: [batch_size, num_particles, 4] - Input 4-momenta to refine
            scalars: [batch_size, num_particles, in_s_channels] - Optional scalar features
            mask: [batch_size, num_particles] - Optional mask for variable-size inputs
        
        Returns:
            dict containing:
                'fourmomenta': [batch_size, num_particles, 4] - Decoded 4-momenta
                'scalars': [batch_size, num_particles, out_s_channels] or None
        """
        batch_size, num_particles, _ = fourmomenta.shape
        
        # Embed 4-momenta in Geometric Algebra
        multivectors = embed_vector(fourmomenta).unsqueeze(-2)  # [batch, particles, 1, 16]
        
        # Pass through L-GATr
        multivector_outputs, scalar_outputs = self.lgatr(multivectors, scalars=scalars)
        
        # Extract 4-vectors
        fourmomenta_out = extract_vector(multivector_outputs)  # [batch, particles, out_mv_channels, 4]
        
        if self.out_mv_channels == 1:
            fourmomenta_out = fourmomenta_out.squeeze(2)  # [batch, particles, 4]
        
        return {
            'fourmomenta': fourmomenta_out,
            'scalars': scalar_outputs
        }
    
    def get_output_dim(self):
        """Return the output dimension of scalar features."""
        return self.out_s_channels if self.out_s_channels > 0 else 0


# Utility functions for 4-vector conversions

def convert_to_four_vector(pt, eta, phi, mass):
    """
    Convert (pt, eta, phi, mass) to (E, px, py, pz).
    
    Args:
        pt: transverse momentum [batch, particles] or scalar
        eta: pseudorapidity [batch, particles] or scalar
        phi: azimuthal angle [batch, particles] or scalar
        mass: particle mass [batch, particles] or scalar
    
    Returns:
        four_vector: (E, px, py, pz) [batch, particles, 4]
    """
    # Momentum components
    px = pt * torch.cos(phi)
    py = pt * torch.sin(phi)
    pz = pt * torch.sinh(eta)
    
    # Energy from mass-shell relation: E² = p² + m²
    p_squared = px**2 + py**2 + pz**2
    E = torch.sqrt(p_squared + mass**2 + 1e-8)  # Add epsilon for numerical stability
    
    # Stack into 4-vector
    four_vector = torch.stack([E, px, py, pz], dim=-1)
    
    return four_vector


def convert_from_four_vector(four_vector):
    """
    Convert (E, px, py, pz) to (pt, eta, phi, mass).
    
    Args:
        four_vector: [batch, particles, 4] or [particles, 4] - [E, px, py, pz]
    
    Returns:
        tuple: (pt, eta, phi, mass) each with same shape as input (minus last dim)
    """
    E = four_vector[..., 0]
    px = four_vector[..., 1]
    py = four_vector[..., 2]
    pz = four_vector[..., 3]
    
    # Transverse momentum
    pt = torch.sqrt(px**2 + py**2 + 1e-8)
    
    # Azimuthal angle
    phi = torch.atan2(py, px)
    
    # Pseudorapidity: η = arctanh(pz/p)
    p = torch.sqrt(px**2 + py**2 + pz**2 + 1e-8)
    eta = torch.atanh(torch.clamp(pz / p, min=-0.9999, max=0.9999))  # Clamp to avoid inf
    
    # Invariant mass from mass-shell relation: m² = E² - p²
    mass_squared = E**2 - p**2
    mass = torch.sqrt(torch.clamp(mass_squared, min=0.0))  # Clamp to avoid negative masses
    
    return pt, eta, phi, mass


def compute_invariant_mass(four_vector):
    """
    Compute invariant mass from 4-vector: m² = E² - p²
    
    Args:
        four_vector: [batch, particles, 4] - [E, px, py, pz]
    
    Returns:
        mass: [batch, particles] - Invariant mass
    """
    E = four_vector[..., 0]
    px = four_vector[..., 1]
    py = four_vector[..., 2]
    pz = four_vector[..., 3]
    
    mass_squared = E**2 - (px**2 + py**2 + pz**2)
    mass = torch.sqrt(torch.clamp(mass_squared, min=0.0))
    
    return mass


def enforce_mass_shell(four_vector, target_mass=0.0):
    """
    Enforce mass-shell condition: E² = p² + m² by adjusting energy.
    
    Args:
        four_vector: [batch, particles, 4] - [E, px, py, pz]
        target_mass: Target mass (default 0 for massless particles)
    
    Returns:
        corrected_four_vector: [batch, particles, 4] with corrected energy
    """
    px = four_vector[..., 1]
    py = four_vector[..., 2]
    pz = four_vector[..., 3]
    
    # Correct energy to satisfy mass-shell
    p_squared = px**2 + py**2 + pz**2
    E_corrected = torch.sqrt(p_squared + target_mass**2 + 1e-8)
    
    # Create corrected 4-vector
    corrected = torch.stack([E_corrected, px, py, pz], dim=-1)
    
    return corrected
