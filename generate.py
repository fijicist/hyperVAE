"""
═══════════════════════════════════════════════════════════════════════════════
GENERATION SCRIPT - SAMPLE JETS FROM TRAINED HYPERVAE
═══════════════════════════════════════════════════════════════════════════════

Purpose:
--------
Generate synthetic jets by sampling from the trained VAE's latent space.
Outputs jets in PyTorch Geometric Data format for downstream analysis.

Usage:
------
Generate jets from best checkpoint:
    python generate.py --checkpoint checkpoints/best_model.pt \
                       --output generated_jets.pt \
                       --num-samples 10000 \
                       --gpu

Key Features:
-------------
1. LATENT SAMPLING: Sample z ~ N(0, I) from VAE prior
2. CONDITIONAL GENERATION: Control jet type distribution (quark/gluon/top)
3. BATCH GENERATION: Memory-efficient batched sampling
4. PyG FORMAT: Output as list of PyTorch Geometric Data objects
5. PARTICLE-ONLY: Generates particle 4-momenta (no edges/hyperedges by default)

Generation Pipeline:
--------------------
For each batch:
    1. Sample latent vectors: z ~ N(0, I)
    2. Sample jet types from distribution
    3. Decode: z, jet_type → particles (4-momenta)
    4. Threshold particle mask (keep only valid particles)
    5. Convert to PyG Data format
    6. Save as list of Data objects

Output Format:
--------------
Saves a list of PyG Data objects to --output path:

    data_list = [
        Data(
            particle_x=[n_particles_1, 4],  # 4-momenta [E, px, py, pz]
            n_particles=n_particles_1,
            jet_type=0,                      # 0=quark, 1=gluon, 2=top
            y=[1, num_jet_features]          # [jet_type, jet_pt, jet_eta, jet_mass, ...]
        ),
        Data(...),  # jet 2
        ...         # 10000 jets total
    ]

Load with:
    import torch
    data_list = torch.load('generated_jets.pt')
    print(f"Generated {len(data_list)} jets")
    print(f"First jet: {data_list[0].n_particles} particles")

Arguments:
----------
--checkpoint: Path to trained model checkpoint (.pt file)
--output: Path to save generated jets (.pt file)
--num-samples: Number of jets to generate (default: 1000)
--batch-size: Generation batch size (default: 128)
--jet-type-dist: Jet type distribution as "q,g,t" (default: "0.33,0.33,0.34")
                 Example: "0.5,0.3,0.2" = 50% quark, 30% gluon, 20% top
--gpu: Use GPU if available (default: True)
--temperature: Gumbel-Softmax temperature for mask sampling (default: 0.5)
               Lower = more deterministic, Higher = more stochastic

Jet Type Distribution:
----------------------
Controls the proportion of each jet flavor in the generated sample:
- Quark jets (type 0): Light quark fragmentation, fewer particles
- Gluon jets (type 1): More radiation, higher multiplicity
- Top jets (type 2): Heavy top decay, boosted substructure

Default "0.33,0.33,0.34" generates balanced dataset.
For realistic physics, use data-driven distribution from training set.

Typical Usage Patterns:
-----------------------
1. Generate test set for evaluation:
    python generate.py --checkpoint best_model.pt --output test_gen.pt --num-samples 10000

2. Generate large sample for analysis:
    python generate.py --checkpoint best_model.pt --output large_sample.pt --num-samples 100000 --batch-size 256

3. Generate specific jet types (e.g., only gluon jets):
    python generate.py --checkpoint best_model.pt --output gluon_jets.pt --num-samples 5000 --jet-type-dist "0,1,0"

4. GPU acceleration:
    python generate.py --checkpoint best_model.pt --output jets.pt --num-samples 50000 --gpu

Output Validation:
------------------
After generation, validate outputs with:
    python evaluate.py --real-data data/real/jets.pt --generated-data generated_jets.pt

This computes Wasserstein distances and generates comparison plots.

Troubleshooting:
----------------
Issue: Generated particles have invalid 4-momenta (E < |p⃗|)
- L-GATr should prevent this, but check decoder temperature
- Lower temperature (0.3-0.5) = more conservative/physical outputs

Issue: Too many/few particles per jet
- Check model.decoder.max_particles setting
- Check particle_mask thresholding (threshold > 0.5 for validity)

Issue: OOM during generation
- Reduce --batch-size (128 → 64 or 32)
- Generate in multiple runs and concatenate

Issue: Slow generation
- Use --gpu flag (10-50× speedup)
- Increase --batch-size if memory allows
- Generation is much faster than training (~1000 jets/sec on GPU)

═══════════════════════════════════════════════════════════════════════════════
"""
import torch
import yaml
import argparse
import numpy as np
from tqdm import tqdm
from torch_geometric.data import Data

from models.hypervae import BipartiteHyperVAE


def generate_jets(model, num_samples, batch_size, jet_type_dist, device, temperature=0.5):
    """
    Generate jets using trained VAE model by sampling from latent prior.
    
    Samples latent vectors z ~ N(0, I) and decodes them to particle 4-momenta
    conditioned on jet type. Generates particle features only (no edges/hyperedges).
    
    Args:
        model: Trained BipartiteHyperVAE model in eval mode
        num_samples: Total number of jets to generate
        batch_size: Batch size for generation (memory vs speed tradeoff)
        jet_type_dist: [3] array with jet type probabilities [p_quark, p_gluon, p_top]
                       Must sum to 1.0
        device: torch.device for generation ('cuda' or 'cpu')
        temperature: Gumbel-Softmax temperature for topology sampling (default: 0.5)
                     Lower = more deterministic, Higher = more stochastic
    
    Returns:
        dict with keys:
            'particle_features': List of [n_particles, 4] arrays (4-momenta per jet)
            'jet_features': List of [num_jet_features] arrays (jet-level observables)
            'jet_types': List of jet type integers (0=quark, 1=gluon, 2=top)
            'n_particles': List of particle counts per jet
    """
    model.eval()
    
    # Sample jet types according to distribution
    jet_types = np.random.choice(
        [0, 1, 2], 
        size=num_samples, 
        p=jet_type_dist
    )
    
    all_particle_features = []
    all_jet_types = []
    all_jet_features = []
    all_n_particles = []
    
    with torch.no_grad():
        for i in tqdm(range(0, num_samples, batch_size), desc="Generating jets"):
            batch_jet_types = jet_types[i:i+batch_size]
            current_batch_size = len(batch_jet_types)
            
            # Sample from prior N(0, I)
            z = torch.randn(current_batch_size, model.config['model']['latent_dim'], device=device)
            jet_type_tensor = torch.tensor(batch_jet_types, dtype=torch.long, device=device)
            
            # Generate (particles only - no edges/hyperedges)
            output = model.decoder(
                z, 
                jet_type_tensor, 
                temperature=temperature,
            )
            
            # Extract particle features and jet features
            particle_features = output['particle_features'].cpu().numpy()
            jet_features = output['jet_features'].cpu().numpy()  # [batch_size, num_jet_features]
            n_particles = output['topology']['n_particles'].cpu().numpy()
            
            # Store
            for j in range(current_batch_size):
                n_p = int(n_particles[j])
                
                # Only store non-zero particles
                if n_p > 0:
                    all_particle_features.append(particle_features[j, :n_p])
                    all_n_particles.append(n_p)
                else:
                    # Use actual feature dimension from model config
                    num_features = particle_features.shape[-1]
                    all_particle_features.append(np.zeros((1, num_features)))
                    all_n_particles.append(1)
                
                # Store jet features and jet type
                all_jet_features.append(jet_features[j])
                all_jet_types.append(batch_jet_types[j])
    
    return {
        'particle_features': all_particle_features,
        'jet_features': all_jet_features,
        'jet_types': np.array(all_jet_types),
        'n_particles': np.array(all_n_particles)
    }


def save_generated_jets(generated_data, output_path):
    """Save generated jets to .pt file (PyG Data format) - particles and jet features"""
    print(f"\nSaving generated jets to {output_path}...")
    
    data_list = []
    
    for i in range(len(generated_data['jet_types'])):
        # Get particle features
        particle_feat = torch.tensor(generated_data['particle_features'][i], dtype=torch.float32)
        
        # Combine jet_type (at index 0) with jet features (jet_pt, jet_eta, jet_mass)
        jet_type = generated_data['jet_types'][i]
        jet_features = generated_data['jet_features'][i]
        
        # Create y tensor: [jet_type, jet_pt, jet_eta, jet_mass, ...]
        y = torch.cat([
            torch.tensor([jet_type], dtype=torch.float32),
            torch.tensor(jet_features, dtype=torch.float32)
        ])
        
        # Create PyG Data object
        data = Data(
            x=particle_feat,  # [n_particles, num_features] - particle features
            y=y,              # [1 + num_jet_features] - jet_type + jet_features
            num_nodes=particle_feat.size(0)
        )
        
        data_list.append(data)
    
    # Save to file
    torch.save(data_list, output_path)
    print(f"Saved {len(data_list)} jets to {output_path}")


def print_statistics(generated_data):
    """Print statistics about generated jets (particles only)"""
    print("\n" + "="*60)
    print("Generated Jet Statistics")
    print("="*60)
    
    # Jet types
    jet_types = generated_data['jet_types']
    n_quark = (jet_types == 0).sum()
    n_gluon = (jet_types == 1).sum()
    n_top = (jet_types == 2).sum()
    total = len(jet_types)
    
    print(f"\nJet Type Distribution:")
    print(f"  Quark: {n_quark} ({100*n_quark/total:.1f}%)")
    print(f"  Gluon: {n_gluon} ({100*n_gluon/total:.1f}%)")
    print(f"  Top:   {n_top} ({100*n_top/total:.1f}%)")
    
    # Particles
    n_particles = generated_data['n_particles']
    print(f"\nParticles per Jet:")
    print(f"  Mean: {n_particles.mean():.2f}")
    print(f"  Std:  {n_particles.std():.2f}")
    print(f"  Min:  {n_particles.min()}")
    print(f"  Max:  {n_particles.max()}")
    
    # Particle features
    all_particles = np.concatenate(generated_data['particle_features'], axis=0)
    num_features = all_particles.shape[1]
    print(f"\nParticle Feature Statistics:")
    print(f"  Total particles: {all_particles.shape[0]}")
    print(f"  Feature dimensions: {num_features}")
    
    # Print feature statistics based on number of features
    if num_features == 3:
        print(f"  Features: (pt, eta, phi)")
        print(f"  pt   - Mean: {all_particles[:, 0].mean():.2f}, Std: {all_particles[:, 0].std():.2f}")
        print(f"  eta  - Mean: {all_particles[:, 1].mean():.2f}, Std: {all_particles[:, 1].std():.2f}")
        print(f"  phi  - Mean: {all_particles[:, 2].mean():.2f}, Std: {all_particles[:, 2].std():.2f}")
    elif num_features == 4:
        print(f"  Features: (E, px, py, pz)")
        print(f"  E    - Mean: {all_particles[:, 0].mean():.2f}, Std: {all_particles[:, 0].std():.2f}")
        print(f"  px   - Mean: {all_particles[:, 1].mean():.2f}, Std: {all_particles[:, 1].std():.2f}")
        print(f"  py   - Mean: {all_particles[:, 2].mean():.2f}, Std: {all_particles[:, 2].std():.2f}")
        print(f"  pz   - Mean: {all_particles[:, 3].mean():.2f}, Std: {all_particles[:, 3].std():.2f}")
    else:
        print(f"  Feature statistics for {num_features} features:")
        for i in range(num_features):
            print(f"  feat{i} - Mean: {all_particles[:, i].mean():.2f}, Std: {all_particles[:, i].std():.2f}")
    
    # Jet features
    jet_features = np.array(generated_data['jet_features'])
    num_jet_features = jet_features.shape[1]
    print(f"\nJet Feature Statistics:")
    print(f"  Number of jet features: {num_jet_features}")
    if num_jet_features >= 3:
        print(f"  Features: (jet_pt, jet_eta, jet_mass)")
        print(f"  jet_pt   - Mean: {jet_features[:, 0].mean():.2f}, Std: {jet_features[:, 0].std():.2f}")
        print(f"  jet_eta  - Mean: {jet_features[:, 1].mean():.2f}, Std: {jet_features[:, 1].std():.2f}")
        print(f"  jet_mass - Mean: {jet_features[:, 2].mean():.2f}, Std: {jet_features[:, 2].std():.2f}")
    else:
        for i in range(num_jet_features):
            print(f"  jet_feat{i} - Mean: {jet_features[:, i].mean():.2f}, Std: {jet_features[:, i].std():.2f}")



def main(args):
    # Load config
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    config = checkpoint['config']
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() and args.gpu else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    model = BipartiteHyperVAE(config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
    
    # Jet type distribution
    if args.jet_type is not None:
        # Single jet type mode
        jet_type_map = {'quark': 0, 'q': 0, 'gluon': 1, 'g': 1, 'top': 2, 't': 2}
        jet_type_name_map = {0: 'Quark', 1: 'Gluon', 2: 'Top'}
        
        jet_type_lower = args.jet_type.lower()
        if jet_type_lower not in jet_type_map:
            raise ValueError(f"Invalid jet type: {args.jet_type}. Use 'quark', 'gluon', or 'top'")
        
        jet_type_idx = jet_type_map[jet_type_lower]
        jet_type_dist = [0.0, 0.0, 0.0]
        jet_type_dist[jet_type_idx] = 1.0
        print(f"\nGenerating only {jet_type_name_map[jet_type_idx]} jets")
    else:
        # Mixed jet type mode (legacy)
        jet_type_dist = [args.quark_frac, args.gluon_frac, args.top_frac]
        jet_type_dist = np.array(jet_type_dist) / sum(jet_type_dist)
        print(f"\nJet type distribution: Quark={jet_type_dist[0]:.2f}, "
              f"Gluon={jet_type_dist[1]:.2f}, Top={jet_type_dist[2]:.2f}")
    
    # Determine generation temperature
    if args.temperature is not None:
        temperature = args.temperature
        print(f"\nUsing temperature: {temperature:.2f} (from command line)")
    else:
        temperature = config['training'].get('final_temperature', 0.5)
        print(f"\nUsing temperature: {temperature:.2f} (from config final_temperature)")
    
    # Generate jets
    print(f"\nGenerating {args.num_samples} jets...")
    generated_data = generate_jets(
        model, 
        args.num_samples,
        args.batch_size,  
        jet_type_dist,    
        device,
        temperature=temperature
    )
    
    # Print statistics
    print_statistics(generated_data)
    
    # Save
    if args.output:
        save_generated_jets(generated_data, args.output)
    
    print("\nGeneration completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate jets using trained HyperVAE')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--output', type=str, default='generated_jets.pt', help='Output .pt file')
    parser.add_argument('--num-samples', type=int, default=10000, help='Number of jets to generate')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for generation')
    parser.add_argument('--jet-type', type=str, default=None, 
                        help='Generate single jet type: "quark"/"q", "gluon"/"g", or "top"/"t" (overrides frac args)')
    parser.add_argument('--quark-frac', type=float, default=0.33, help='Fraction of quark jets (used if --jet-type not set)')
    parser.add_argument('--gluon-frac', type=float, default=0.33, help='Fraction of gluon jets (used if --jet-type not set)')
    parser.add_argument('--top-frac', type=float, default=0.34, help='Fraction of top jets (used if --jet-type not set)')
    parser.add_argument('--gpu', action='store_true', help='Use GPU if available')
    parser.add_argument('--temperature', type=float, default=None, 
                        help='Gumbel-Softmax temperature (default: uses final_temperature from config)')
    
    args = parser.parse_args()
    main(args)
