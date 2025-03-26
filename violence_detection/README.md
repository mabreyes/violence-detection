# Violence Detection System for Philippine Context

A complete system for detecting violence in videos using 4D CNN + LSTM + Transformer architecture, specifically optimized for limited computational resources in the Philippine context.

## Features

- **Efficient Architecture**: 4D CNN + LSTM + Transformer designed for limited compute resources
- **Philippine Context Optimization**: Model parameters and data processing tuned for local context
- **Complete Pipeline**: Data preprocessing, training, inference and evaluation tools included
- **Mobile Deployment Support**: Optimized model variant for mobile applications
- **Visualization Tools**: Advanced visualization of violence detection results

## Requirements

- Python 3.7+
- PyTorch 1.8+
- OpenCV
- scikit-learn
- pandas
- numpy
- matplotlib
- tqdm

## Installation

### Option 1: Using UV (Recommended)

[UV](https://github.com/astral-sh/uv) is a much faster alternative to pip, built in Rust by Astral.

1. Install UV:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Clone the repository:

```bash
git clone https://github.com/yourusername/philippine-violence-detection.git
cd philippine-violence-detection
```

3. Create a virtual environment and install dependencies with UV:

```bash
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e ".[dev]"  # Install with development dependencies
```

### Option 2: Using pip

1. Clone the repository:

```bash
git clone https://github.com/yourusername/philippine-violence-detection.git
cd philippine-violence-detection
```

2. Create a virtual environment (recommended):

```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Dataset Preparation

The system works with video data organized as follows:

# Violence Detection Package Design

This package is organized following the single responsibility principle (SRP) and Pythonic design practices.

## Package Structure

```
violence_detection/
├── __init__.py             # Package entry point
├── __version__.py          # Version information
├── data/                   # Data loading and preprocessing
│   ├── __init__.py
│   ├── dataset.py          # Dataset classes
│   └── dataloader.py       # DataLoader utilities
├── models/                 # Model architecture
│   ├── __init__.py
│   ├── layers.py           # Basic building blocks
│   ├── transformer.py      # Transformer components
│   ├── violence_detection.py  # Main model implementation
│   └── factory.py          # Factory functions
├── utils/                  # Utilities
│   ├── __init__.py
│   ├── training.py         # Training utilities
│   └── train.py            # Main training loop
├── inference/              # Inference functionality
│   ├── __init__.py
│   └── detector.py         # Violence detector
└── cli/                    # Command-line interface
    ├── __init__.py
    ├── main.py             # Entry point for CLI
    ├── commands.py         # Command handlers
    └── preprocessing.py    # Data preprocessing
```

## Module Responsibilities

- **data**: Handles data loading and preprocessing
- **models**: Contains neural network models and components
- **utils**: Provides training and evaluation utilities
- **inference**: Contains inference utilities for running the model
- **cli**: Implements the command-line interface

## Design Principles

This package follows these design principles:

1. **Single Responsibility Principle**: Each module has a single responsibility.
2. **Separation of Concerns**: Data handling, model definition, training, inference, and CLI are separate.
3. **Abstraction**: High-level interfaces hide implementation details.
4. **Composability**: Components can be combined in multiple ways.
5. **Pythonic**: Follows Python's conventions and idioms.

## Extension Points

The system can be extended in the following ways:

1. Add new models in the `models` directory
2. Add new data loaders in the `data` directory
3. Add new training utilities in the `utils` directory
4. Add new inference methods in the `inference` directory
5. Add new commands in the `cli` directory
