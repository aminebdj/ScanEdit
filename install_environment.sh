#!/bin/bash

# SCANEDIT Environment Installation Script
# This script sets up the complete conda environment with all dependencies

set -e  # Exit on error

echo "=================================="
echo "SCANEDIT Environment Setup"
echo "=================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
ENV_NAME="scanedit"
PYTHON_VERSION="3.10.9"
CUDA_VERSION="11.8"  # Change this if you have a different CUDA version

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    print_error "Conda is not installed. Please install Anaconda or Miniconda first."
    exit 1
fi

print_status "Conda found: $(conda --version)"

# Check CUDA availability
if command -v nvidia-smi &> /dev/null; then
    print_status "NVIDIA GPU detected:"
    nvidia-smi --query-gpu=name --format=csv,noheader
    CUDA_AVAILABLE=true
else
    print_warning "No NVIDIA GPU detected. Installing CPU-only versions."
    CUDA_AVAILABLE=false
fi

# Create conda environment
print_status "Creating conda environment: $ENV_NAME with Python $PYTHON_VERSION"
conda create -n $ENV_NAME python=$PYTHON_VERSION -y

# Activate environment
print_status "Activating environment: $ENV_NAME"
eval "$(conda shell.bash hook)"
conda activate $ENV_NAME

# Verify activation
if [[ "$CONDA_DEFAULT_ENV" != "$ENV_NAME" ]]; then
    print_error "Failed to activate environment"
    exit 1
fi

print_status "Environment activated successfully"

# Install PyTorch and related packages
print_status "Installing PyTorch ecosystem..."
if [ "$CUDA_AVAILABLE" = true ]; then
    print_status "Installing PyTorch with CUDA $CUDA_VERSION support"
    pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu118
else
    print_status "Installing PyTorch CPU-only version"
    pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cpu
fi

# Install PyTorch3D (requires special handling)
print_status "Installing PyTorch3D..."
if [ "$CUDA_AVAILABLE" = true ]; then
    pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"
else
    print_warning "PyTorch3D installation may fail without CUDA. Attempting CPU-only build..."
    pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable" || print_warning "PyTorch3D installation failed (may need CUDA)"
fi

# Install CUDA-specific packages
if [ "$CUDA_AVAILABLE" = true ]; then
    print_status "Installing CUDA-specific packages..."
    pip install ninja
    
    # Install MinkowskiEngine (requires compilation)
    print_status "Installing MinkowskiEngine..."
    pip install -U git+https://github.com/NVIDIA/MinkowskiEngine -v --no-deps \
        --install-option="--blas_include_dirs=${CONDA_PREFIX}/include" \
        --install-option="--blas=openblas" || print_warning "MinkowskiEngine installation may have issues"
    
    # Install torch-scatter
    print_status "Installing torch-scatter..."
    pip install torch-scatter -f https://data.pyg.org/whl/torch-2.4.0+cu118.html
fi

# Install core scientific packages
print_status "Installing core scientific packages..."
conda install -c conda-forge -y \
    numpy=1.26.4 \
    scipy \
    scikit-learn \
    scikit-image \
    pandas \
    matplotlib \
    seaborn \
    opencv \
    pillow

# Install 3D processing libraries
print_status "Installing 3D processing libraries..."
pip install \
    open3d==0.18.0 \
    trimesh==3.14.0 \
    plyfile==0.7.4 \
    pyvista==0.44.2 \
    pymeshlab \
    point-cloud-utils==0.30.0 \
    meshio==5.3.5

# Install deep learning frameworks and tools
print_status "Installing deep learning frameworks..."
pip install \
    transformers==4.37.2 \
    tokenizers==0.15.1 \
    accelerate==0.21.0 \
    peft==0.13.2 \
    bitsandbytes==0.44.1 \
    safetensors==0.4.5 \
    einops==0.6.1 \
    timm==0.6.13

# Install detectron2
print_status "Installing Detectron2..."
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'

# Install Segment Anything
print_status "Installing Segment Anything..."
pip install git+https://github.com/facebookresearch/segment-anything.git

# Install vision and annotation tools
print_status "Installing vision and annotation tools..."
pip install \
    supervision==0.19.0 \
    albumentations==1.4.18 \
    pycocotools==2.0.8

# Install LLM and API clients
print_status "Installing LLM and API clients..."
pip install \
    openai==1.51.2 \
    ollama==0.4.6 \
    huggingface-hub==0.25.2

# Install Gradio and web frameworks
print_status "Installing Gradio and web frameworks..."
pip install \
    gradio==4.16.0 \
    fastapi==0.115.2 \
    uvicorn==0.21.1

# Install configuration and utilities
print_status "Installing configuration and utilities..."
pip install \
    hydra-core==1.0.5 \
    omegaconf==2.0.6 \
    pyyaml \
    easydict==1.13 \
    yacs==0.1.8 \
    addict==2.4.0

# Install logging and monitoring
print_status "Installing logging and monitoring..."
pip install \
    wandb==0.15.0 \
    tensorboard==2.12.2 \
    tqdm \
    loguru==0.6.0 \
    rich==13.4.2

# Install additional utilities
print_status "Installing additional utilities..."
pip install \
    fire==0.4.0 \
    click==8.1.3 \
    typer==0.12.5 \
    python-dotenv==0.20.0 \
    pathlib \
    joblib \
    psutil

# Install Jupyter and notebook tools
print_status "Installing Jupyter and notebook tools..."
pip install \
    jupyter \
    jupyterlab \
    ipykernel \
    ipywidgets \
    notebook

# Install test and development tools
print_status "Installing development tools..."
pip install \
    pytest \
    black \
    flake8 \
    mypy \
    ruff

# Install BlenderProc (optional, may take time)
print_status "Installing BlenderProc (optional)..."
pip install blenderproc || print_warning "BlenderProc installation failed (optional dependency)"

# Install remaining packages from requirements
print_status "Installing remaining packages..."
pip install \
    prettytable \
    termcolor \
    tabulate \
    natsort \
    fpsample \
    gdown \
    requests \
    aiohttp \
    httpx

# Set up IPython kernel for Jupyter
print_status "Setting up IPython kernel..."
python -m ipykernel install --user --name $ENV_NAME --display-name "Python ($ENV_NAME)"

# Verify installation
print_status "Verifying installation..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torchvision; print(f'TorchVision version: {torchvision.__version__}')"

if [ "$CUDA_AVAILABLE" = true ]; then
    python -c "import pytorch3d; print(f'PyTorch3D imported successfully')" || print_warning "PyTorch3D import failed"
fi

python -c "import open3d; print(f'Open3D version: {open3d.__version__}')"
python -c "import trimesh; print(f'Trimesh imported successfully')"
python -c "import transformers; print(f'Transformers version: {transformers.__version__}')"

echo ""
echo "=================================="
print_status "Installation completed successfully!"
echo "=================================="
echo ""
echo "To activate the environment, run:"
echo "  conda activate $ENV_NAME"
echo ""
echo "To deactivate, run:"
echo "  conda deactivate"
echo ""

# Create activation helper script
cat > activate_scanedit.sh << 'EOF'
#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate scanedit
echo "SCANEDIT environment activated!"
echo "Python: $(which python)"
echo "PyTorch CUDA: $(python -c 'import torch; print(torch.cuda.is_available())')"
EOF

chmod +x activate_scanedit.sh
print_status "Created activation helper: ./activate_scanedit.sh"

print_status "Setup complete! Happy editing! 🎨"