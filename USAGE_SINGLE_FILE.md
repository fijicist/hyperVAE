# Using a Single .pt Data File

This guide shows how to train the model with a single `.pt` file that gets automatically split into train/validation/test sets.

## Quick Start

### Option 1: Auto-Split (Recommended)

Train with automatic 80/10/10 split:

```bash
python train.py \
    --config config.yaml \
    --data-path /path/to/your/all_jets.pt \
    --save-test-indices
```

This will:
- Load your single `.pt` file
- Automatically split into:
  - **80%** training
  - **10%** validation  
  - **10%** test
- Save test indices to `checkpoints/test_indices.pt` for reproducible evaluation

### Option 2: Custom Split Ratios

Use custom split ratios:

```bash
python train.py \
    --config config.yaml \
    --data-path /path/to/your/all_jets.pt \
    --train-frac 0.7 \
    --val-frac 0.15 \
    --test-frac 0.15 \
    --split-seed 42 \
    --save-test-indices
```

## Data Splitting Options

### 1. **Single File (Auto-Split)**
```bash
python train.py --data-path data.pt --save-test-indices
```
→ Auto-splits into 80/10/10 (train/val/test)

### 2. **Separate Val File**
```bash
python train.py --data-path train.pt --val-data-path val.pt
```
→ Uses separate validation file, auto-splits train into train/test

### 3. **All Separate Files**
```bash
python train.py \
    --data-path train.pt \
    --val-data-path val.pt \
    --test-data-path test.pt
```
→ Uses all separate files (no auto-splitting)

## Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--data-path` | None | Path to data .pt file |
| `--val-data-path` | None | Optional separate validation file |
| `--test-data-path` | None | Optional separate test file |
| `--train-frac` | 0.8 | Training fraction (80%) |
| `--val-frac` | 0.1 | Validation fraction (10%) |
| `--test-frac` | 0.1 | Test fraction (10%) |
| `--split-seed` | 42 | Random seed for reproducible splits |
| `--save-test-indices` | False | Save test indices for evaluation |

## Complete Workflow

### Step 1: Train with Auto-Split

```bash
python train.py \
    --config config.yaml \
    --data-path my_jets.pt \
    --train-frac 0.8 \
    --val-frac 0.1 \
    --test-frac 0.1 \
    --split-seed 42 \
    --save-test-indices \
    --save-dir checkpoints \
    --log-dir runs
```

**Output:**
```
Loading data from: my_jets.pt
Loaded 10000 jets
Sample jet structure:
  x (particles): torch.Size([30, 3])
  ...

Dataset split:
  Total: 10000 jets
  Train: 8000 jets (80.0%)
  Val:   1000 jets (10.0%)
  Test:  1000 jets (10.0%)

Saved test indices to: checkpoints/test_indices.pt

Model parameters: 0.51M
...
```

### Step 2: Generate Jets

```bash
python generate.py \
    --checkpoint checkpoints/best_model.pt \
    --output generated_jets.pt \
    --num-samples 1000 \
    --gpu
```

### Step 3: Evaluate Against Test Set

**Option A: Using saved test indices**
```bash
python evaluate_with_split.py \
    --data-path my_jets.pt \
    --test-indices checkpoints/test_indices.pt \
    --generated-data generated_jets.pt \
    --plot \
    --plot-dir plots
```

**Option B: Using separate test file (if you have one)**
```bash
python evaluate.py \
    --real-data test_jets.pt \
    --generated-data generated_jets.pt \
    --plot
```

## Reproducibility

To ensure reproducible splits across multiple runs:

1. **Use same seed**:
```bash
--split-seed 42
```

2. **Save test indices**:
```bash
--save-test-indices
```

3. **Reuse test indices** for evaluation:
```bash
python evaluate_with_split.py \
    --data-path data.pt \
    --test-indices checkpoints/test_indices.pt \
    --generated-data generated.pt
```

## Example: Full Training Pipeline

```bash
# 1. Train with auto-split
python train.py \
    --config config.yaml \
    --data-path /data/all_jets.pt \
    --train-frac 0.8 \
    --val-frac 0.1 \
    --test-frac 0.1 \
    --split-seed 42 \
    --save-test-indices \
    --save-dir checkpoints/run1 \
    --log-dir runs/run1

# 2. Monitor training
tensorboard --logdir runs/run1

# 3. Generate jets after training
python generate.py \
    --checkpoint checkpoints/run1/best_model.pt \
    --output generated_jets.pt \
    --num-samples 10000 \
    --batch-size 32 \
    --gpu

# 4. Evaluate against test split
python evaluate_with_split.py \
    --data-path /data/all_jets.pt \
    --test-indices checkpoints/run1/test_indices.pt \
    --generated-data generated_jets.pt \
    --plot \
    --plot-dir plots/run1
```

## What Gets Saved

After training with `--save-test-indices`:

```
checkpoints/
├── best_model.pt              # Best model checkpoint
├── checkpoint_epoch20.pt      # Periodic checkpoints
├── checkpoint_epoch40.pt
├── ...
└── test_indices.pt           # Test set indices (for reproducibility)

runs/
└── experiment/
    └── events.out.tfevents.*  # TensorBoard logs
```

## Checking Your Split

After loading data, the script will print:

```
Dataset split:
  Total: 10000 jets
  Train: 8000 jets (80.0%)
  Val:   1000 jets (10.0%)
  Test:  1000 jets (10.0%)
```

Verify this matches your expectations!

## Tips

1. **Start with default split (80/10/10)**:
   ```bash
   python train.py --data-path data.pt
   ```

2. **Always save test indices** if you want reproducible evaluation:
   ```bash
   --save-test-indices
   ```

3. **Use same seed** across experiments for fair comparison:
   ```bash
   --split-seed 42
   ```

4. **Validation size**: Should be large enough for reliable metrics (typically 10% or 1000+ jets)

5. **Test size**: Keep separate from validation to avoid overfitting to val set

## Common Scenarios

### Scenario 1: I have one big .pt file with 50,000 jets

```bash
# Default 80/10/10 split
python train.py --data-path big_data.pt --save-test-indices
# Results: 40k train, 5k val, 5k test
```

### Scenario 2: I want more validation data

```bash
# 70/20/10 split
python train.py \
    --data-path data.pt \
    --train-frac 0.7 \
    --val-frac 0.2 \
    --test-frac 0.1 \
    --save-test-indices
```

### Scenario 3: I already have train/val but want to split test from train

```bash
python train.py \
    --data-path train_and_test.pt \  # Will be split into train/test
    --val-data-path separate_val.pt \
    --save-test-indices
```

## Troubleshooting

### Issue: "Fractions must sum to 1.0"
**Solution:** Check that `train_frac + val_frac + test_frac = 1.0`

### Issue: "Dataset too small to split"
**Solution:** Need at least 10 jets total. For small datasets, use 70/15/15 or similar.

### Issue: "Can't find test_indices.pt"
**Solution:** Rerun training with `--save-test-indices` flag

---

**You're ready to train with a single data file!** 🚀

The auto-splitting makes it easy to use your complete dataset without manually splitting files.
