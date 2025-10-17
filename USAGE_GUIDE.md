# Complete Usage Guide: Bipartite HyperVAE

## Quick Start (5 minutes)

Test the model with dummy data:

```bash
python quickstart.py
```

This will verify all components work on your system.

## Preparing Your Data

### Expected Data Format

Your jet data should be in **HDF5 format** with the following structure:

```python
import h5py
import numpy as np

# Example: Create HDF5 file
with h5py.File('my_jets.h5', 'w') as f:
    # For each jet, store:
    
    # 1. Particle features [N_jets, variable_particles, 4]
    #    Features: (pt, eta, phi, mass)
    particle_features = [...]  # List of arrays with shape (n_particles_i, 4)
    
    # 2. Edge features [N_jets, variable_edges, 5]
    #    Features: (ln_delta, ln_kt, ln_z, ln_m2, feat5)
    edge_features = [...]  # List of arrays with shape (n_edges_i, 5)
    
    # 3. Hyperedge features [N_jets, variable_hyperedges, 2]
    #    Features: (3pt_eec, 4pt_eec)
    hyperedge_features = [...]  # List of arrays with shape (n_hyperedges_i, 2)
    
    # 4. Bipartite edge connections [N_jets, 2, variable]
    #    Connections from particles to hyperedges
    edge_index_bipartite = [...]  # List of arrays with shape (2, n_connections_i)
    
    # 5. Jet types [N_jets]
    #    0 = quark, 1 = gluon, 2 = top
    jet_types = np.array([...])  # Shape (N_jets,)
    
    # Save with variable-length arrays
    dt = h5py.vlen_dtype(np.dtype('float32'))
    f.create_dataset('particle_features', data=particle_features, dtype=dt)
    f.create_dataset('edge_features', data=edge_features, dtype=dt)
    f.create_dataset('hyperedge_features', data=hyperedge_features, dtype=dt)
    f.create_dataset('edge_index_bipartite', data=edge_index_bipartite, dtype=dt)
    f.create_dataset('jet_types', data=jet_types)
```

### Converting Your Data

If you have jet data in a different format, here's a template:

```python
import numpy as np
import h5py

def convert_to_hypervae_format(your_jets, output_path):
    """
    Convert your jet data to HyperVAE format.
    
    Args:
        your_jets: Your jet data structure
        output_path: Output HDF5 file path
    """
    particle_features_list = []
    edge_features_list = []
    hyperedge_features_list = []
    edge_index_list = []
    jet_types_list = []
    
    for jet in your_jets:
        # 1. Extract particle 4-vectors
        particles = jet['particles']  # Your format here
        pt = particles['pt']
        eta = particles['eta']
        phi = particles['phi']
        mass = particles['mass']
        particle_feat = np.stack([pt, eta, phi, mass], axis=-1)
        
        # 2. Compute edge features (2-point correlators)
        # This is problem-specific - compute your 5 edge features
        edge_feat = compute_edge_features(particles)
        
        # 3. Compute hyperedge features (3,4-point EEC)
        eec_3pt = compute_3point_eec(particles)
        eec_4pt = compute_4point_eec(particles)
        hyperedge_feat = np.stack([eec_3pt, eec_4pt], axis=-1)
        
        # 4. Build bipartite connections
        # Connect particles to hyperedges based on your structure
        edge_index = build_bipartite_graph(particles, hyperedges)
        
        # 5. Jet type
        jet_type = jet['label']  # 0, 1, or 2
        
        particle_features_list.append(particle_feat.astype(np.float32))
        edge_features_list.append(edge_feat.astype(np.float32))
        hyperedge_features_list.append(hyperedge_feat.astype(np.float32))
        edge_index_list.append(edge_index.astype(np.int64))
        jet_types_list.append(jet_type)
    
    # Save to HDF5
    with h5py.File(output_path, 'w') as f:
        dt_float = h5py.vlen_dtype(np.dtype('float32'))
        dt_int = h5py.vlen_dtype(np.dtype('int64'))
        
        f.create_dataset('particle_features', (len(particle_features_list),), dtype=dt_float)
        for i, pf in enumerate(particle_features_list):
            f['particle_features'][i] = pf.flatten()
        
        # Similar for other fields...
        f.create_dataset('jet_types', data=np.array(jet_types_list))

# Use it
convert_to_hypervae_format(your_jets, 'train_jets.h5')
```

## Training

### Basic Training

```bash
python train.py \
    --config config.yaml \
    --data-path train_jets.h5 \
    --val-data-path val_jets.h5 \
    --save-dir checkpoints \
    --log-dir runs
```

### Monitor Training

In another terminal:
```bash
tensorboard --logdir runs
```

Then open http://localhost:6006 in your browser.

### Memory Issues?

If you run out of VRAM, edit `config.yaml`:

```yaml
training:
  batch_size: 2  # Reduce from 4
  gradient_accumulation_steps: 16  # Increase to maintain effective batch size
```

### Training Tips

1. **Start with KL warmup**: The model uses KL annealing to stabilize training
2. **Monitor all losses**: Check that particle, edge, and hyperedge losses all decrease
3. **Save checkpoints**: Models are saved every 20 epochs
4. **Best model**: Automatically saved when validation loss improves

## Generation

### Generate Jets

After training, generate new jets:

```bash
python generate.py \
    --checkpoint checkpoints/best_model.pt \
    --output generated_jets.h5 \
    --num-samples 10000 \
    --batch-size 32 \
    --quark-frac 0.4 \
    --gluon-frac 0.4 \
    --top-frac 0.2 \
    --gpu
```

### Control Jet Types

Adjust the fractions to control jet type distribution:

```bash
# Generate only quark jets
python generate.py ... --quark-frac 1.0 --gluon-frac 0.0 --top-frac 0.0

# Generate 50% gluon, 50% top
python generate.py ... --quark-frac 0.0 --gluon-frac 0.5 --top-frac 0.5
```

### Generated Data Format

The output HDF5 file contains:
- `particle_features`: (pt, eta, phi, mass) for each jet
- `edge_features`: 5D edge features
- `hyperedge_features`: 3-pt and 4-pt EEC
- `jet_types`: Type labels (0, 1, 2)
- `n_particles`: Number of particles per jet
- `n_hyperedges`: Number of hyperedges per jet

## Evaluation

### Compare Distributions

```bash
python evaluate.py \
    --real-data test_jets.h5 \
    --generated-data generated_jets.h5 \
    --max-jets 10000 \
    --plot \
    --plot-dir plots
```

This computes:
- **Wasserstein distances** for each feature
- **Structural metrics** (particle counts, jet types)
- **Distribution plots** saved to `plots/`

### Interpreting Results

**Wasserstein Distance**: Lower is better (0 = perfect match)
- `< 0.01`: Excellent
- `0.01 - 0.05`: Good
- `0.05 - 0.1`: Acceptable
- `> 0.1`: Poor

**Check the plots**:
- Particle feature distributions should overlap
- Number of particles histogram should match
- Jet type fractions should be similar

## Advanced Usage

### Custom Loss Weights

Edit `config.yaml` to prioritize certain features:

```yaml
training:
  loss_weights:
    particle_features: 20.0  # Increase if particles are poor quality
    edge_features: 10.0      # Increase if edges are important
    hyperedge_features: 3.0
    topology: 1.0
    kl_divergence: 0.001
```

### Adjust Model Size

For more VRAM or better quality:

```yaml
model:
  particle_hidden: 128  # Increase from 64
  edge_hidden: 96       # Increase from 48
  hyperedge_hidden: 64  # Increase from 32
  latent_dim: 256       # Increase from 128
```

### Resume Training

```bash
python train.py \
    --config config.yaml \
    --resume checkpoints/checkpoint_epoch100.pt \
    --data-path train_jets.h5
```

### Using Generated Jets

Load generated jets in Python:

```python
import h5py
import numpy as np

with h5py.File('generated_jets.h5', 'r') as f:
    for i in range(len(f['jet_types'])):
        # Get particle features
        particles = np.array(f['particle_features'][i]).reshape(-1, 4)
        pt = particles[:, 0]
        eta = particles[:, 1]
        phi = particles[:, 2]
        mass = particles[:, 3]
        
        # Get jet type
        jet_type = f['jet_types'][i]
        
        # Use particles...
```

## Troubleshooting

### Out of Memory

1. Reduce `batch_size` in config.yaml
2. Increase `gradient_accumulation_steps`
3. Reduce `max_particles` if your jets are smaller
4. Enable `mixed_precision: true`

### Poor Generation Quality

1. Train for more epochs (200+)
2. Increase model capacity (hidden dimensions)
3. Adjust loss weights to focus on problematic features
4. Check input data quality and normalization
5. Increase KL warmup period

### Training Unstable

1. Reduce learning rate
2. Enable gradient clipping (default: 1.0)
3. Increase KL warmup epochs
4. Check for NaN in input data

### Generated Jets Have Wrong Particle Count

The model learns particle counts from data. If consistently wrong:
1. Increase `topology` loss weight
2. Check `n_particles` in training data
3. Adjust `max_particles` in config

## Performance Benchmarks

**GTX 1650Ti (4GB)**:
- Training: ~35 sec/epoch (1000 jets, batch=4)
- Generation: ~300 jets/sec
- Memory: ~3.5GB VRAM

**RTX 3060 (12GB)**:
- Training: ~15 sec/epoch (1000 jets, batch=16)
- Generation: ~800 jets/sec
- Memory: ~6GB VRAM

## Citation & References

Key papers:
- **L-GATr**: Brehmer et al., "Geometric Algebra Transformers"
- **Energy Correlators**: Komiske et al., "Energy Flow Networks"
- **Graph VAEs**: Simonovsky & Komodakis, "GraphVAE"

## Support

For issues or questions:
1. Check the README.md
2. Run `python quickstart.py` to verify installation
3. Check TensorBoard logs for training issues
4. Ensure data format matches specification

## Next Steps

1. ✓ Run quickstart test
2. ✓ Prepare your jet data
3. ✓ Train on small subset (1000 jets, 10 epochs)
4. ✓ Evaluate generated jets
5. ✓ Fine-tune hyperparameters
6. ✓ Full training on complete dataset
7. ✓ Generate final jet samples
8. ✓ Physics validation (invariant masses, etc.)
