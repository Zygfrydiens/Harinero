# Harinero Setup Guide

## Prerequisites

### Required Software
- Python 3.8 or higher
- Anaconda or Miniconda ([Download](https://docs.conda.io/en/latest/miniconda.html))
- Git (for version control)

### System Requirements
- Operating System: Windows 10/11, macOS, or Linux
- RAM: Minimum 8GB (16GB recommended)
- Storage: At least 5GB free space

## Installation Steps

### 1. Create and Activate Conda Environment

```bash
# Create new environment
conda create -n harinero_dev python=3.8

# Activate environment
conda activate harinero_dev
```

### 2. Clone and Install Harinero

```bash
# Clone repository (if not already done)
git clone https://github.com/yourusername/harinero.git
cd harinero

# Install package in editable mode
pip install --use-pep517 -e .
```

### 3. Set Up Jupyter Integration

```bash
# Install kernel for Jupyter
python -m ipykernel install --user --name harinero_dev --display-name "Python (Harinero)"
```

## Verification Steps

### 1. Verify Installation
Open a Python console and try importing key components:

```python
# Test imports
from harinero import SongStruct, TandaStruct, MilongaStruct
print("Core structures imported successfully!")
```

### 2. Verify Jupyter Integration
1. Launch Jupyter Notebook: `jupyter notebook`
2. Create new notebook
3. Select "Python (Harinero)" kernel
4. Try running the test imports above

## Troubleshooting

### Common Issues and Solutions

#### 1. Package Not Found
If you get `ModuleNotFoundError: No module named 'harinero'`:
```bash
# Verify installation
pip list | grep harinero

# If not listed, reinstall
pip install --use-pep517 -e .
```

#### 2. Import Errors
If you experience import errors, try installing core dependencies via conda:
```bash
# Install core scientific packages
conda install numpy pandas scipy matplotlib jupyter ipykernel

# Install ML frameworks
conda install tensorflow pytorch torchvision torchaudio -c pytorch
```

#### 3. Jupyter Kernel Issues
If the Harinero kernel doesn't appear in Jupyter:
```bash
# Remove and reinstall kernel
jupyter kernelspec uninstall harinero_dev
python -m ipykernel install --user --name harinero_dev --display-name "Python (Harinero)"
```

## Additional Resources

- [Conda Documentation](https://docs.conda.io/)
- [Project Documentation](./docs/)

## Version Information

This guide is for Harinero v0.1.0. For other versions, check the corresponding tags in the repository.
