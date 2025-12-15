# Edge Encoder Update: GATv2Conv on Precomputed Line Graphs

## Summary
Updated the edge encoder in `models/encoder.py` to use PyTorch Geometric's **GATv2Conv** layers operating on precomputed **line graphs** (edges-as-nodes representation), replacing the previous MLP-based edge encoder.

## Motivation
- **Graph structure learning**: Line graphs explicitly represent relationships between edges (two edges connected if they share a particle)
- **Attention-based aggregation**: GATv2Conv learns importance weights for edge-edge interactions
- **Efficient preprocessing**: Line graphs are precomputed offline and saved with the data, avoiding runtime computation

## Changes Made

### 1. `models/encoder.py`

#### Imports
- Added: `from torch_geometric.nn import GATv2Conv`

#### `BipartiteEncoder.__init__`
**Removed:**
- `self.edge_mlp = nn.ModuleList([...])` - Old residual MLP stack

**Added:**
- `self.edge_gat_layers = nn.ModuleList([GATv2Conv(...)])` - GAT stack for line graph processing
- New config parameters:
  - `edge_gat_layers` (default: 3) - Number of GAT layers
  - `edge_gat_heads` (default: 8) - Number of attention heads per layer
- Assert: `edge_hidden % edge_gat_heads == 0` to ensure proper head splitting

**Configuration:**
```python
GATv2Conv(
    in_channels=edge_hidden,
    out_channels=edge_hidden // num_edge_gat_heads,
    heads=num_edge_gat_heads,
    concat=True,              # Output = heads * out_channels = edge_hidden
    dropout=dropout,
    add_self_loops=True,
)
```

#### `BipartiteEncoder.forward`
**Signature change:**
```python
# OLD
def forward(self, batch):

# NEW
def forward(self, batch_particles, batch_edges):
```

**Parameters:**
- `batch_particles`: Original PyG Batch with particles, hyperedges, jet features
- `batch_edges`: PyG Batch of line graphs with:
  - `x`: [N_line_nodes, edge_features] - Line graph node features (original edge_attr)
  - `edge_index`: [2, N_line_edges] - Line graph connectivity
  - `batch`: [N_line_nodes] - Batch assignment

**Edge encoding logic:**
```python
# OLD: MLP with manual per-graph pooling
edge_x = self.edge_embed(batch.edge_attr)
for layer in self.edge_mlp:
    edge_x = edge_x + layer(edge_x)
# Manual loop over graphs to pool edges

# NEW: GATv2Conv with built-in global pooling
edge_x = self.edge_embed(batch_edges.x)
for gat_layer in self.edge_gat_layers:
    edge_x = gat_layer(edge_x, batch_edges.edge_index)
edge_pooled = self._global_mean_pool(edge_x, batch_edges.batch)
```

**Validation:**
- Assert `edge_pooled.shape[0] == batch_particles.num_graphs`
- Assert `edge_pooled.shape[1] == self.config['edge_hidden']`

#### Test block update
```python
# OLD
from data.bipartite_dataset import BipartiteJetDataset, collate_bipartite_batch
loader = DataLoader(dataset, collate_fn=collate_bipartite_batch)
batch = next(iter(loader))
mu, logvar = encoder(batch)

# NEW
from data.bipartite_dataset import BipartiteJetDataset, collate_bipartite_batch_with_line_graphs
loader = DataLoader(dataset, collate_fn=collate_bipartite_batch_with_line_graphs)
batch_particles, batch_edges = next(iter(loader))
mu, logvar = encoder(batch_particles, batch_edges)
```

### 2. `data/bipartite_dataset.py`

#### Dataset class
**Added:**
- `self.line_graphs = None` - Storage for precomputed line graphs

#### `_load_from_pt`
**Updated to load line graphs:**
```python
if isinstance(loaded_data, dict):
    data_list = loaded_data['graphs']
    self.line_graphs = loaded_data.get('line_graphs', None)  # NEW
    metadata = loaded_data.get('metadata', {})
```

**Validation:**
- Assert `len(self.line_graphs) == len(data_list)` if line graphs present

#### `__getitem__`
**Changed return type:**
```python
# OLD
return data

# NEW
line_graph = self.line_graphs[idx] if self.line_graphs is not None else None
return data, line_graph
```

#### New collate function
**Added `collate_bipartite_batch_with_line_graphs`:**
```python
def collate_bipartite_batch_with_line_graphs(data_list):
    """
    Returns: (batch_particles, batch_edges)
        - batch_particles: Batch of particle graphs
        - batch_edges: Batch of line graphs (edges-as-nodes)
    """
    particle_graphs = [item[0] for item in data_list]
    line_graphs = [item[1] for item in data_list]
    
    batch_particles = Batch.from_data_list(particle_graphs)
    batch_edges = Batch.from_data_list(line_graphs) if line_graphs[0] is not None else None
    
    return batch_particles, batch_edges
```

**Updated `collate_bipartite_batch` for backward compatibility:**
- Detects tuple format `(data, line_graph)` vs old list format
- Returns tuple if line graphs present, single batch otherwise

### 3. `models/hypervae.py`

#### `BipartiteHyperVAE.forward`
**Signature change:**
```python
# OLD
def forward(self, batch, temperature=1.0, generate_all_features=None):

# NEW
def forward(self, batch_particles, batch_edges=None, temperature=1.0, generate_all_features=None):
```

**Updated calls:**
```python
# Encoder
mu, logvar = self.encoder(batch_particles, batch_edges)

# Decoder
output = self.decoder(z, batch_particles.jet_type, temperature)
```

#### `BipartiteHyperVAE.compute_loss`
**Signature change:**
```python
# OLD
def compute_loss(self, batch, output, epoch=0):

# NEW
def compute_loss(self, batch_particles, output, epoch=0, batch_edges=None):
```

**Updated all internal references:**
- All `batch.x`, `batch.y`, `batch.num_graphs` → `batch_particles.*`

### 4. `train.py`

#### Imports
**Added:**
```python
from data.bipartite_dataset import collate_bipartite_batch_with_line_graphs
```

#### `collate_with_stats`
**Updated to handle both batches:**
```python
# OLD
batch = collate_bipartite_batch(data_list)
# ... attach stats to batch ...
return batch

# NEW
batch_particles, batch_edges = collate_bipartite_batch_with_line_graphs(data_list)
# ... attach stats to batch_particles ...
return batch_particles, batch_edges
```

#### Training loop
**Updated to unpack and pass both batches:**
```python
# OLD
for i, batch in enumerate(pbar):
    batch = batch.to(device)
    output = model(batch, temperature=temperature)
    losses = model.compute_loss(batch, output, epoch=epoch)

# NEW
for i, batch_data in enumerate(pbar):
    batch_particles, batch_edges = batch_data
    batch_particles = batch_particles.to(device)
    batch_edges = batch_edges.to(device) if batch_edges is not None else None
    output = model(batch_particles, batch_edges, temperature=temperature)
    losses = model.compute_loss(batch_particles, output, epoch=epoch, batch_edges=batch_edges)
```

#### Validation loop
**Same pattern as training loop.**

## Configuration Updates

### Required Config Keys
Add to `config.yaml` under `encoder`:
```yaml
encoder:
  edge_gat_layers: 3      # Number of GATv2Conv layers for line graph
  edge_gat_heads: 8       # Number of attention heads per layer
  dropout: 0.1            # Used in GATv2Conv
```

### Constraints
- `edge_hidden` must be divisible by `edge_gat_heads`
- Example: If `edge_hidden=48` and `edge_gat_heads=8`, each head outputs 6 channels

## Data Requirements

### Line Graph Format
Each line graph Data object must contain:
- `x`: [N_edges, edge_features] - Node features (original graph's `edge_attr`)
- `edge_index`: [2, N_line_edges] - Connectivity (two edges share a particle)

### Saved Shard Format
Each `.pt` file should have:
```python
{
    'graphs': [Data, Data, ...],           # Original particle graphs
    'line_graphs': [Data, Data, ...],      # Line graphs (1:1 with graphs)
    'metadata': {
        'particle_norm_stats': {...},
        'edge_norm_stats': {...},
        ...
    }
}
```

## Backward Compatibility

### Old Data Files
If `line_graphs` key is missing:
- Dataset returns `(data, None)` tuples
- `batch_edges` will be `None`
- Encoder must handle `batch_edges=None` case (creates zero tensor)

### Old Collate Function
`collate_bipartite_batch` now detects format and handles both:
- Old: List of Data → returns single batch
- New: List of (Data, line_graph) → returns tuple of batches

## Testing

### Encoder Test
```bash
python models/encoder.py
```
Expected output:
```
Batch size: 4
Mu shape: torch.Size([4, 128])
Logvar shape: torch.Size([4, 128])
```

### Full Training Test
```bash
python train.py --config config.yaml --data-path data/jets_with_line_graphs.pt
```

## Benefits

1. **Structural reasoning**: GAT learns edge-edge relationships via shared particles
2. **Attention mechanism**: Learns importance weights for different edge interactions
3. **Scalability**: Batched GAT operations more efficient than per-graph loops
4. **Flexibility**: Number of layers and heads easily configurable
5. **Expressiveness**: Multi-head attention captures diverse edge interaction patterns

## Performance Considerations

- **Memory**: Line graphs stored on disk (~same size as edges), loaded into memory during training
- **Compute**: GATv2Conv operations are batched and GPU-accelerated
- **Preprocessing**: Line graphs computed once offline (see `graph_constructor.py`)

## Next Steps

1. **Tune hyperparameters**: Try different `edge_gat_layers` (2-4) and `edge_gat_heads` (4-16)
2. **Experiment with edge features**: Could include additional edge observables in line graph node features
3. **Monitor attention weights**: Visualize learned attention patterns for interpretability
4. **Compare performance**: Benchmark against old MLP-based edge encoder

## References

- **GATv2**: Brody et al., "How Attentive are Graph Attention Networks?" (2021)
- **Line Graphs**: PyTorch Geometric `LineGraph` transform documentation
- **PyG Batching**: https://pytorch-geometric.readthedocs.io/en/latest/notes/batching.html
