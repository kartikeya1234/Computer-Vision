# Computer Vision Deep Learning Framework

A comprehensive PyTorch-based framework for training and evaluating deep learning models on standard computer vision datasets. This project provides a flexible command-line interface for experimenting with various neural network architectures and datasets.

## Features

- **9 Model Architectures**: Pre-trained and custom CNN/MLP implementations
- **5 Standard Datasets**: MNIST, CIFAR-10, CIFAR-100, NIPS2017, ImageNet
- **4 Optimizers**: SGD, Adam, Adagrad, and custom Nesterov Momentum SGD
- **GPU Acceleration**: Automatic CUDA detection and support
- **Data Augmentation**: Model-specific preprocessing and transformations
- **Progress Tracking**: Real-time training metrics with progress bars

## Supported Models

| Model | Paper | Supported Datasets |
|-------|-------|-------------------|
| ResNet18 | [Deep Residual Learning](https://arxiv.org/abs/1512.03385) | CIFAR-10, CIFAR-100 |
| ResNet50 | [Deep Residual Learning](https://arxiv.org/abs/1512.03385) | ImageNet, NIPS2017 |
| ResNet20 | [Deep Residual Learning](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf) | CIFAR-10, CIFAR-100 |
| VGG19 | [Very Deep CNNs](https://arxiv.org/abs/1409.1556) | ImageNet, NIPS2017 |
| WideResNet | [Wide Residual Networks](https://arxiv.org/abs/1605.07146) | CIFAR-10, CIFAR-100 |
| LeNet | [Understanding Deep Learning](https://arxiv.org/abs/1706.06083) | MNIST |
| BasicCNN | [Systematic Evaluation](https://arxiv.org/abs/1608.04644) | CIFAR-10, CIFAR-100 |
| SmallCNN | [Systematic Evaluation](https://arxiv.org/abs/1608.04644) | MNIST |
| BasicMLP | - | MNIST |

## Supported Datasets

- **[MNIST](http://yann.lecun.com/exdb/mnist/)**: Handwritten digits (28x28 grayscale)
- **[CIFAR-10](https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf)**: 10 object classes (32x32 color)
- **[CIFAR-100](https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf)**: 100 object classes (32x32 color)
- **[NIPS2017](https://www.kaggle.com/competitions/nips-2017-defense-against-adversarial-attack/data)**: Adversarial defense competition dataset
- **[ImageNet](https://www.image-net.org/)**: Large-scale image database

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (optional, for acceleration)

### Dependencies

Install required packages:

```bash
pip install -r requirements.txt
```

Dependencies include:
- torch==2.2.1
- torchvision==0.17.1
- pandas==2.2.1
- pillow==10.2.0
- tqdm==4.66.2
- natsort==8.4.0
- argparse==1.1

## Setup

1. Create necessary directories:
```bash
mkdir data
mkdir Trained_Models
```

2. For NIPS2017 dataset, place images in `./data/NIPS2017/images/` with corresponding CSV labels

3. For ImageNet, ensure data is available at `/software/ais2t/pytorch_datasets/imagenet/` (or modify path in code)

## Usage

### Basic Training Command

```bash
python main.py \
  --model ResNet20 \
  --dataSet CIFAR10 \
  --optim Adam \
  --maxIterations 10 \
  --batchSize 64 \
  --numWorkers 2 \
  --saveModel 1 \
  --ver 1
```

### Command-Line Arguments

| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `--model` | str | Yes | Model architecture (e.g., ResNet20, VGG19) |
| `--dataSet` | str | Yes | Dataset name (e.g., CIFAR10, MNIST) |
| `--optim` | str | Yes | Optimizer (SGD, Adam, Adagrad, MyNesterovMomentumSGD) |
| `--maxIterations` | int | Yes | Number of training epochs |
| `--batchSize` | int | Yes | Batch size for training |
| `--numWorkers` | int | Yes | Number of data loading workers |
| `--saveModel` | int | Yes | Save trained model (1=yes, 0=no) |
| `--ver` | int | Yes | Verbose output (1=yes, 0=no) |

### Examples

Train ResNet50 on ImageNet:
```bash
python main.py --model ResNet50 --dataSet ImageNet --optim SGD --maxIterations 50 --batchSize 128 --numWorkers 4 --saveModel 1 --ver 1
```

Train LeNet on MNIST:
```bash
python main.py --model BasicLeNet --dataSet MNIST --optim Adam --maxIterations 20 --batchSize 32 --numWorkers 2 --saveModel 1 --ver 1
```

Train WideResNet on CIFAR-100:
```bash
python main.py --model WideResNet --dataSet CIFAR100 --optim SGD --maxIterations 100 --batchSize 64 --numWorkers 4 --saveModel 1 --ver 1
```

## Project Structure

```
Computer-Vision/
├── main.py                 # Main training script with CLI interface
├── models.py              # Model architecture definitions
├── train_model.py         # Training and evaluation logic
├── utils.py               # Utility functions and custom dataset class
├── requirements.txt       # Project dependencies
├── trainingParams.json    # Configuration file for model parameters
├── data/                  # Dataset storage (create manually)
├── Trained_Models/        # Saved model weights (create manually)
└── README.md             # This file
```

## Training Workflow

1. **Data Loading**: Dataset is loaded with model-specific transformations (resize, crop, flip, normalize)
2. **Model Initialization**: Architecture is instantiated and moved to GPU/CPU
3. **Training Loop**: Model trains for specified epochs with progress tracking
4. **Evaluation**: Model is evaluated on test set
5. **Model Saving**: Trained weights saved to `Trained_Models/{model}_{dataset}_{optimizer}_trained_model.pt`

## Data Augmentation

Each model uses optimized preprocessing:
- **VGG19**: 224x224 resize, rotation, horizontal flip, random crop, normalization
- **ResNet50**: 255 resize, 224 center crop, horizontal flip, rotation, normalization
- **ResNet20**: 32x32 random crop, horizontal flip, normalization
- **MLP models**: Tensor conversion only (no augmentation)

## Optimizers

- **[SGD](https://pytorch.org/docs/stable/generated/torch.optim.SGD.html)**: Stochastic Gradient Descent
- **[Adam](https://pytorch.org/docs/stable/generated/torch.optim.Adam.html)**: Adaptive Moment Estimation
- **[Adagrad](https://pytorch.org/docs/stable/generated/torch.optim.Adagrad.html)**: Adaptive Gradient Algorithm
- **MyNesterovMomentumSGD**: Custom Nesterov Momentum implementation

## Model Output

Trained models are saved with the naming convention:
```
{model}_{dataset}_{optimizer}_trained_model.pt
```

Example: `ResNet20_CIFAR10_Adam_trained_model.pt`

## Hardware Support

- Automatically detects and uses CUDA-enabled GPUs
- Falls back to CPU if GPU unavailable
- Multi-worker data loading for improved performance

## Contributing

Feel free to submit issues or pull requests for:
- New model architectures
- Additional datasets
- Performance optimizations
- Bug fixes

## License

This project is provided as-is for research and educational purposes.

## References

- [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
- [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)
- [Wide Residual Networks](https://arxiv.org/abs/1605.07146)
- [CIFAR-10 and CIFAR-100 Datasets](https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf)
