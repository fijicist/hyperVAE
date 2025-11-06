#!/bin/bash

# ═══════════════════════════════════════════════════════════════════════════
# HYPERVAE SETUP SCRIPT - AUTOMATED ENVIRONMENT CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════
#
# This script automates the installation of HyperVAE and its dependencies.
# It detects your system configuration (Python, CUDA, GPU) and installs
# the appropriate versions of PyTorch, PyTorch Geometric, and other packages.
#
# Usage:
#   bash setup.sh
#
# Requirements:
#   - Python 3.10 or higher
#   - CUDA 11.8+ (optional, for GPU support)
#   - 4GB+ GPU VRAM recommended (tested on GTX 1650 Ti)
#
# ═══════════════════════════════════════════════════════════════════════════

set -e  # Exit on error

echo "════════════════════════════════════════════════════════════════"
echo "  HyperVAE Setup - Memory-Optimized Jet Generation with L-GATr  "
echo "════════════════════════════════════════════════════════════════"
echo ""

# ┌─────────────────────────────────────────────────────────────────────────┐
# │ STEP 1: System Checks                                                    │
# └─────────────────────────────────────────────────────────────────────────┘
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║ Step 1: System Configuration Check                             ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# Check Python version
echo "→ Checking Python version..."
if ! command -v python3 &> /dev/null; then
    echo "  ✗ Python 3 not found. Please install Python 3.10 or higher."
    exit 1
fi

python_version=$(python3 --version 2>&1 | awk '{print $2}')
python_major=$(echo $python_version | cut -d'.' -f1)
python_minor=$(echo $python_version | cut -d'.' -f2)

echo "  ✓ Python version: $python_version"

if [ "$python_major" -lt 3 ] || ([ "$python_major" -eq 3 ] && [ "$python_minor" -lt 10 ]); then
    echo "  ⚠ Warning: Python 3.10+ recommended, found $python_version"
    read -p "  Continue anyway? (y/N): " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check pip
echo ""
echo "→ Checking pip..."
if ! command -v pip3 &> /dev/null && ! command -v pip &> /dev/null; then
    echo "  ✗ pip not found. Please install pip."
    exit 1
fi
echo "  ✓ pip is available"

# Check CUDA availability
echo ""
echo "→ Checking CUDA..."
CUDA_AVAILABLE=false
CUDA_VERSION=""

if command -v nvcc &> /dev/null; then
    nvcc_version=$(nvcc --version | grep "release" | awk '{print $5}' | cut -d',' -f1)
    CUDA_VERSION=$nvcc_version
    CUDA_AVAILABLE=true
    echo "  ✓ CUDA detected: $CUDA_VERSION"
else
    echo "  ⚠ CUDA not found - will install CPU-only PyTorch"
fi

# Check GPU
echo ""
echo "→ Checking GPU..."
GPU_AVAILABLE=false

if command -v nvidia-smi &> /dev/null; then
    gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1 2>/dev/null || echo "Unknown")
    gpu_memory=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -n 1 2>/dev/null || echo "Unknown")
    GPU_AVAILABLE=true
    echo "  ✓ GPU detected: $gpu_name"
    echo "    VRAM: $gpu_memory"
    
    # Check if it's the tested GPU
    if [[ "$gpu_name" == *"1650"* ]]; then
        echo "    ✓ GTX 1650 Ti detected (tested configuration)"
    fi
else
    echo "  ⚠ No GPU detected - will use CPU (slower)"
fi

# Determine PyTorch installation command
echo ""
echo "→ Determining installation strategy..."
if [ "$CUDA_AVAILABLE" = true ]; then
    # Extract major.minor version (e.g., 12.1 from 12.1.0)
    cuda_major=$(echo $CUDA_VERSION | cut -d'.' -f1)
    cuda_minor=$(echo $CUDA_VERSION | cut -d'.' -f2)
    
    if [ "$cuda_major" -ge 12 ]; then
        TORCH_INDEX="https://download.pytorch.org/whl/cu121"
        TORCH_CUDA="cu121"
        echo "  → Will install PyTorch with CUDA 12.1 support"
    elif [ "$cuda_major" -eq 11 ] && [ "$cuda_minor" -ge 8 ]; then
        TORCH_INDEX="https://download.pytorch.org/whl/cu118"
        TORCH_CUDA="cu118"
        echo "  → Will install PyTorch with CUDA 11.8 support"
    else
        echo "  ⚠ CUDA $CUDA_VERSION is older than 11.8"
        echo "    Falling back to CPU-only installation"
        TORCH_INDEX=""
        TORCH_CUDA="cpu"
    fi
else
    TORCH_INDEX=""
    TORCH_CUDA="cpu"
    echo "  → Will install CPU-only PyTorch"
fi

echo ""
echo "════════════════════════════════════════════════════════════════"
echo ""

# ┌─────────────────────────────────────────────────────────────────────────┐
# │ STEP 2: Create Project Directories                                       │
# └─────────────────────────────────────────────────────────────────────────┘
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║ Step 2: Creating Project Directories                           ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

directories=("checkpoints" "runs" "plots" "data/real" "data/generated")

for dir in "${directories[@]}"; do
    if [ ! -d "$dir" ]; then
        mkdir -p "$dir"
        echo "  ✓ Created: $dir"
    else
        echo "  ✓ Exists: $dir"
    fi
done

echo ""
echo "════════════════════════════════════════════════════════════════"
echo ""

# ┌─────────────────────────────────────────────────────────────────────────┐
# │ STEP 3: Install PyTorch                                                  │
# └─────────────────────────────────────────────────────────────────────────┘
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║ Step 3: Installing PyTorch (this may take a few minutes)       ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

if [ -n "$TORCH_INDEX" ]; then
    echo "→ Installing PyTorch 2.2.0 with $TORCH_CUDA support..."
    pip install torch==2.2.0 --index-url "$TORCH_INDEX"
else
    echo "→ Installing PyTorch 2.2.0 (CPU-only)..."
    pip install torch==2.2.0
fi

if [ $? -eq 0 ]; then
    echo ""
    echo "  ✓ PyTorch installed successfully"
else
    echo ""
    echo "  ✗ PyTorch installation failed"
    exit 1
fi

# Verify PyTorch installation
echo ""
echo "→ Verifying PyTorch installation..."
python3 -c "import torch; print(f'  ✓ PyTorch {torch.__version__} loaded'); print(f'  ✓ CUDA available: {torch.cuda.is_available()}')" 2>/dev/null || {
    echo "  ✗ PyTorch verification failed"
    exit 1
}

echo ""
echo "════════════════════════════════════════════════════════════════"
echo ""

# ┌─────────────────────────────────────────────────────────────────────────┐
# │ STEP 4: Install PyTorch Geometric                                        │
# └─────────────────────────────────────────────────────────────────────────┘
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║ Step 4: Installing PyTorch Geometric                           ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

echo "→ Installing torch-geometric..."
pip install torch-geometric

if [ $? -eq 0 ]; then
    echo "  ✓ torch-geometric installed"
else
    echo "  ✗ torch-geometric installation failed"
    exit 1
fi

echo ""
echo "→ Installing torch-geometric..."
pip install torch-geometric

if [ $? -eq 0 ]; then
    echo "  ✓ torch-geometric installed"
else
    echo "  ✗ torch-geometric installation failed"
    exit 1
fi

echo ""
echo "════════════════════════════════════════════════════════════════"
echo ""

# ┌─────────────────────────────────────────────────────────────────────────┐
# │ STEP 5: Install Remaining Dependencies                                   │
# └─────────────────────────────────────────────────────────────────────────┘
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║ Step 5: Installing Remaining Dependencies                      ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

echo "→ Installing core dependencies (numpy, scipy, pyyaml, tqdm)..."
pip install numpy>=1.26.0 scipy>=1.14.0 pyyaml>=6.0 tqdm>=4.66.0

echo ""
echo "→ Installing L-GATr (Lorentz-equivariant transformer)..."
pip install lgatr>=1.3.0

if [ $? -eq 0 ]; then
    echo "  ✓ L-GATr installed successfully"
else
    echo "  ✗ L-GATr installation failed"
    echo "    This is a critical dependency - cannot continue"
    exit 1
fi

echo ""
echo "→ Installing particle physics libraries..."
echo "  (fastjet, awkward, energyflow, EEC, jetnet)"
pip install fastjet>=3.4.0 awkward>=2.6.0 energyflow>=1.3.0 energyenergycorrelators>=2.0.0b1 jetnet>=0.2.5

echo ""
echo "→ Installing ML utilities..."
pip install scikit-learn>=1.5.0 numba>=0.58.0 joblib>=1.4.0

echo ""
echo "→ Installing visualization & logging..."
pip install matplotlib>=3.9.0 tensorboard>=2.18.0

echo ""
echo "→ Installing graph utilities..."
pip install networkx>=3.3

echo ""
echo "  ✓ All dependencies installed"

echo ""
echo "════════════════════════════════════════════════════════════════"
echo ""

# ┌─────────────────────────────────────────────────────────────────────────┐
# │ STEP 6: Verify Installation                                              │
# └─────────────────────────────────────────────────────────────────────────┘
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║ Step 6: Verifying Installation                                 ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

echo "→ Checking critical imports..."

# Check imports
python3 << EOF
import sys
success = True

try:
    import torch
    print(f"  ✓ torch {torch.__version__}")
except ImportError as e:
    print(f"  ✗ torch: {e}")
    success = False

try:
    import torch_geometric
    print(f"  ✓ torch_geometric")
except ImportError as e:
    print(f"  ✗ torch_geometric: {e}")
    success = False

try:
    from lgatr import LGATr
    print(f"  ✓ lgatr")
except ImportError as e:
    print(f"  ✗ lgatr: {e}")
    success = False

try:
    import numpy
    print(f"  ✓ numpy {numpy.__version__}")
except ImportError as e:
    print(f"  ✗ numpy: {e}")
    success = False

try:
    import scipy
    print(f"  ✓ scipy")
except ImportError as e:
    print(f"  ✗ scipy: {e}")
    success = False

try:
    import fastjet
    print(f"  ✓ fastjet")
except ImportError as e:
    print(f"  ⚠ fastjet: {e} (optional for graph construction)")

try:
    import awkward
    print(f"  ✓ awkward")
except ImportError as e:
    print(f"  ⚠ awkward: {e} (optional for graph construction)")

sys.exit(0 if success else 1)
EOF

if [ $? -eq 0 ]; then
    echo ""
    echo "  ✓ All critical imports successful"
else
    echo ""
    echo "  ✗ Some critical imports failed"
    echo "    Please check error messages above"
    exit 1
fi

echo ""
echo "════════════════════════════════════════════════════════════════"
echo ""

# ┌─────────────────────────────────────────────────────────────────────────┐
# │ STEP 7: Quick Functionality Test                                         │
# └─────────────────────────────────────────────────────────────────────────┘
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║ Step 7: Running Quick Functionality Test                       ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

if [ -f "quickstart.py" ]; then
    echo "→ Running quickstart.py (dummy data test)..."
    echo ""
    python3 quickstart.py
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "  ✓ Quickstart test passed"
    else
        echo ""
        echo "  ⚠ Quickstart test failed (setup may still be OK)"
    fi
else
    echo "  ⚠ quickstart.py not found, skipping test"
fi

echo ""
echo "════════════════════════════════════════════════════════════════"
echo ""

# ┌─────────────────────────────────────────────────────────────────────────┐
# │ Setup Complete                                                            │
# └─────────────────────────────────────────────────────────────────────────┘
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║                   ✓ SETUP COMPLETE!                            ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "HyperVAE is now ready to use!"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Next Steps:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "1. PREPARE DATA:"
echo "   • Place your jet dataset in data/real/"
echo "   • Or generate graphs: python graph_constructor.py"
echo ""
echo "2. TRAIN MODEL:"
echo "   python train.py --config config.yaml \\"
echo "                   --data-path data/real/jets.pt \\"
echo "                   --save-dir checkpoints/my_model \\"
echo "                   --log-dir runs/my_model"
echo ""
echo "3. MONITOR TRAINING:"
echo "   tensorboard --logdir runs"
echo ""
echo "4. GENERATE JETS:"
echo "   python generate.py --checkpoint checkpoints/my_model/best_model.pt \\"
echo "                      --output data/generated/jets.pt \\"
echo "                      --num-samples 10000"
echo ""
echo "5. EVALUATE:"
echo "   python evaluate.py --real-data data/real/jets.pt \\"
echo "                      --generated-data data/generated/jets.pt"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Documentation:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "• README.md - Project overview"
echo "• config.yaml - Configuration guide"
echo "• models/ - Comprehensive code documentation"
echo "• READMEs/ - Detailed guides and fixes"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
