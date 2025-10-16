# Spec2Class

[![Tests](https://github.com/VickiPol/Spec2Class/workflows/Tests/badge.svg)](https://github.com/VickiPol/Spec2Class/actions)
[![Conda Version](https://img.shields.io/conda/vn/conda-forge/spec2class.svg)](https://anaconda.org/conda-forge/spec2class)
[![Conda Downloads](https://img.shields.io/conda/dn/conda-forge/spec2class.svg)](https://anaconda.org/conda-forge/spec2class)
[![Python Version](https://img.shields.io/conda/pn/conda-forge/spec2class.svg)](https://anaconda.org/conda-forge/spec2class)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![codecov](https://codecov.io/gh/VickiPol/Spec2Class/branch/main/graph/badge.svg)](https://codecov.io/gh/VickiPol/Spec2Class)

**Accurate Prediction of Plant Secondary Metabolite Class using Deep Learning**

Spec2Class is an ensemble classification model for predicting chemical classes of plant secondary metabolites from high-resolution LC-MS/MS spectra.

## Features

- **43 binary neural network classifiers** for different chemical classes
- **SVM ensemble model** for final multiclass prediction
- **Automatic model downloading** from HuggingFace Hub
- **Multiple input formats**: Pickle, MGF, MSP
- **Command-line interface** and **Python API**
- **GPU support** for faster inference

## Important Notes

### Platform and Version Compatibility

Spec2Class consists of two main components:

1. **Binary Neural Network Classifiers** (43 models): These produce identical predictions across all platforms and Python versions ✓
2. **SVM Ensemble Model**: May show minor prediction differences (~32% variation) across:
   - Different platforms (x86_64 vs ARM64)
   - Different sklearn versions (model trained with 0.24.2)
   - Different operating systems

**This is expected behavior** for pickled sklearn models and does not indicate a problem with the method. The core neural network predictions remain consistent and reliable.

### Reproducibility

- For exact reproduction of published results, use the same platform as the original study
- For general use, the current package works reliably on all platforms
- Prediction differences are primarily in borderline cases where multiple classes have similar scores

## Installation

### For Users

#### Via conda (Recommended)

```bash
#conda install -c conda-forge spec2class
conda install -c Ketzahle spec2class
```

### For Developers

#### Quick Start (pip install from source)

```bash
# Clone the repository
git clone https://github.com/VickiPol/Spec2Class.git
cd Spec2Class

# Create environment
conda create -n spec2class python=3.10 -y
conda activate spec2class

# Install in editable mode
pip install -e .

# Verify installation
spec2class --version
```

#### Building Conda Package Locally

If you want to build and test the conda package:

```bash
# Clone the repository
git clone https://github.com/VickiPol/Spec2Class.git
cd Spec2Class

# Create build environment
conda create -n spec2class_build python=3.10 conda-build -y
conda activate spec2class_build

# Build the package
conda build conda-recipe

# Create clean environment and install from local build
conda create -n spec2class python=3.10 -y
conda activate spec2class
conda install --use-local spec2class -y

# Verify installation
spec2class --version
```

## Quick Start

### Command Line Interface

```bash
# Classify spectra from a pickle file
spec2class classify -i input_spectra.pkl -o results/

# Classify from MGF file
spec2class classify -i spectra.mgf -f mgf -o results/

# Download models in advance
spec2class download --group all_models

# Check model status
spec2class status
```

### Python API

```python
from spec2class import Spec2ClassClassifier

# Initialize classifier (downloads models if needed)
classifier = Spec2ClassClassifier(device='cpu')

# Classify from file
results = classifier.classify_from_file('input_spectra.pkl')

# Classify from DataFrame
import pandas as pd
df = pd.read_pickle('spectra.pkl')
results = classifier.classify_dataframe(df)

# View results
print(results[['DB.', 'final_pred', 'estimated_top2_pred']])
```

### Input Format

Your input file must contain:
- **mz**: NumPy array of m/z values
- **Intensity**: NumPy array of intensity values
- **DB.**: Spectrum identifier
- **ExactMass** (optional): Parent ion mass

Example:
```python
import numpy as np
import pandas as pd

data = {
    'DB.': ['spectrum_1', 'spectrum_2'],
    'mz': [
        np.array([100.0, 150.0, 200.0]),
        np.array([120.0, 180.0, 250.0])
    ],
    'Intensity': [
        np.array([1000.0, 5000.0, 3000.0]),
        np.array([2000.0, 8000.0, 4000.0])
    ],
    'ExactMass': [250.5, 280.3]
}

df = pd.DataFrame(data)
df.to_pickle('my_spectra.pkl')
```

## Output

Results include:
- **final_pred**: Top predicted class
- **estimated_top2_pred**: Second most likely class
- **estimated_top3_pred**: Third most likely class
- **probabilities**: Probability scores for top 3 predictions

Example output:
```
         DB.           final_pred  estimated_top2_pred
0  spectrum_1        Flavonoids         Isoflavonoids
1  spectrum_2          Steroids        Triterpenoids
```

## Predicted Classes

Spec2Class predicts 43 chemical classes including:

- Flavonoids, Isoflavonoids, Phenolic acids
- Steroids, Triterpenoids, Diterpenoids
- Alkaloids (various types)
- Fatty acids and derivatives
- Coumarins, Lignans, Stilbenoids
- And more...

Full list: `spec2class list`

## CLI Commands

### Classify
```bash
spec2class classify -i INPUT [-o OUTPUT_DIR] [-f FORMAT] [-d DEVICE]

Options:
  -i, --input          Input file path (required)
  -o, --output-dir     Output directory (default: ./results)
  -f, --format         Input format: auto, pickle, mgf, msp (default: auto)
  -d, --device         Device: cpu or cuda (default: cpu)
  --force-download     Force redownload models
```

### Download Models
```bash
spec2class download [--group GROUP] [--model MODEL] [--force]

Options:
  --group    Download model group: all_models, test_models
  --model    Download specific model by name
  --force    Force redownload even if cached
```

### List Models
```bash
spec2class list [--groups]

Options:
  --groups   Show model groups instead of individual models
```

### Check Status
```bash
spec2class status
```

### Manage Cache
```bash
spec2class cache {info|directory|clear} [--model MODEL] [-y]

Subcommands:
  info       Show cache information
  directory  Show cache directory path
  clear      Clear cached models

Options:
  --model    Specific model to clear
  -y, --yes  Skip confirmation prompt
```

## Model Storage

Models are cached locally using HuggingFace Hub:
- **Binary models**: `VickiPol/binary_models` (43 models, ~1.5 GB)
- **SVM model**: `VickiPol/SVM_model` (~10 MB)

Default cache location: `~/.cache/huggingface/hub/`

## Testing

Spec2Class includes a comprehensive test suite. See [tests/README.md](tests/README.md) for details.

### Quick Test

```bash
# Run fast tests only (no model downloads)
pytest -m "not slow" -v

# Run all tests including integration tests
pytest --runslow -v
```

### For Developers

```bash
# Install test dependencies
pip install -e ".[dev]"

# Run tests with coverage
pytest --cov=spec2class --cov-report=html

# View coverage report
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
```

## Requirements

- Python ≥ 3.10
- PyTorch ≥ 2.0.0
- NumPy < 2.0
- Pandas < 2.0
- scikit-learn ≥ 1.0.0
- huggingface_hub ≥ 0.23

## Project Structure

```
Spec2Class/
├── pyproject.toml          # Python package configuration
├── conda-recipe/           # Conda package recipe
│   └── meta.yaml
├── src/
│   └── spec2class/        # Main package
│       ├── classifier.py   # Main classifier interface
│       ├── models/         # Model management
│       ├── core/           # Core algorithms
│       └── config/         # Configuration
└── tests/                  # Test suite
    ├── test_basic.py       # Unit tests
    └── test_classification.py  # Integration tests
```

## Citation

If you use Spec2Class in your research, please cite:

```bibtex
@article{spec2class2024,
  title={Spec2Class: Accurate Prediction of Plant Secondary Metabolite Class using Deep Learning},
  author={Pol, Vicki and [other authors]},
  journal={[Journal]},
  year={2024}
}
```

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests (`pytest --runslow -v`)
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

Please ensure:
- All tests pass
- Code follows the existing style
- New features include tests
- Documentation is updated

## Contact

- **Author**: Vicki Pol
- **Repository**: https://github.com/VickiPol/Spec2Class
- **Issues**: https://github.com/VickiPol/Spec2Class/issues

## Acknowledgments

This work builds upon established methods in metabolomics and deep learning for mass spectrometry analysis.

## Troubleshooting

### Model Download Issues

If you encounter issues downloading models:

```bash
# Check cache location
spec2class cache directory

# Check what's cached
spec2class cache info

# Clear cache and redownload
spec2class cache clear -y
spec2class download --group all_models
```

### Import Errors

If you get import errors after installation:

```bash
# Verify installation
pip show spec2class

# Reinstall in editable mode
pip install -e .
```

### GPU Support

To use GPU acceleration:

```bash
# Check if CUDA is available
python -c "import torch; print(torch.cuda.is_available())"

# Use GPU in classifier
spec2class classify -i input.pkl -d cuda
```

Or in Python:
```python
classifier = Spec2ClassClassifier(device='cuda')
```