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

## Datasets

### Philippine Violence Detection Dataset

The system is optimized for Philippine contexts and can be trained on your local dataset. The dataset structure should be organized with videos in separate folders for violent and non-violent content.

### Harvard VID Dataset

The system can also use the Violence in Videos (VID) dataset from Harvard Dataverse, which contains 3,000 video clips with equal representation of violent and non-violent scenarios:

```bash
# Download the VID dataset
violence-detection download-vid --output-dir ./data/vid

# Download and process in one step
violence-detection download-vid --output-dir ./data/vid --process
```

For more details, see the [Data Module README](violence_detection/data/README.md).
