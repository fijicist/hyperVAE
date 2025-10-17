#!/bin/bash

echo "=================================================="
echo "Bipartite HyperVAE Setup Script"
echo "=================================================="

# Check Python version
echo -e "\n1. Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "   Python version: $python_version"

# Check CUDA availability
echo -e "\n2. Checking CUDA..."
if command -v nvcc &> /dev/null; then
    nvcc_version=$(nvcc --version | grep "release" | awk '{print $5}' | cut -d',' -f1)
    echo "   CUDA version: $nvcc_version"
else
    echo "   CUDA not found (CPU-only mode)"
fi

# Check GPU
echo -e "\n3. Checking GPU..."
if command -v nvidia-smi &> /dev/null; then
    gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
    gpu_memory=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -n 1)
    echo "   GPU: $gpu_name"
    echo "   VRAM: $gpu_memory"
else
    echo "   No GPU detected (will use CPU)"
fi

# Create directories
echo -e "\n4. Creating directories..."
mkdir -p checkpoints
mkdir -p runs
mkdir -p plots
mkdir -p data
echo "   ✓ Directories created"

# Install dependencies
echo -e "\n5. Installing dependencies..."
echo "   This may take a few minutes..."
pip install -q -r requirements.txt

if [ $? -eq 0 ]; then
    echo "   ✓ Dependencies installed"
else
    echo "   ✗ Error installing dependencies"
    exit 1
fi

# Run quick test
echo -e "\n6. Running quick test..."
python3 quickstart.py

if [ $? -eq 0 ]; then
    echo -e "\n=================================================="
    echo "✓ Setup complete! HyperVAE is ready to use."
    echo "=================================================="
    echo -e "\nNext steps:"
    echo "  1. Prepare your data (see USAGE_GUIDE.md)"
    echo "  2. Train: python train.py --data-path your_data.h5"
    echo "  3. Generate: python generate.py --checkpoint checkpoints/best_model.pt"
else
    echo -e "\n✗ Setup failed. Please check error messages above."
    exit 1
fi
