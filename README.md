# Neural Network Performance Comparison on Fashion MNIST Dataset

## Overview

This project explores image classification using multiple neural network architectures on the FashionMNIST dataset, without using advanced techniques such as data augmentation, transfer learning, or pre-trained models (e.g., ResNet or VGG).
The goal is to analyze how different model structures, optimizers, and learning rates affect classification performance.

Three main stages were developed and evaluated:

1. **Baseline Fully Connected Networks (No Convolution)**
2. **Convolutional Neural Networks (CNN)**
3. **Optimizer and Learning Rate Tuning**

Each model was trained, tested, and compared based on testing accuracy.

## Dataset

The FashionMNIST dataset contains images of various clothing categories used to benchmark computer vision models.  
Each image represents one of the following classes:

| Label | Category    |
| ----- | ----------- |
| 0     | T-shirt/top |
| 1     | Trouser     |
| 2     | Pullover    |
| 3     | Dress       |
| 4     | Coat        |
| 5     | Sandal      |
| 6     | Shirt       |
| 7     | Sneaker     |
| 8     | Bag         |
| 9     | Ankle boot  |

- Image Size: 28×28 pixels
- Classes: 10 (T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot)
- Training Samples: 60,000
- Testing Samples: 10,000

## Summary of Results

| Rank | Test Accuracy (%) | Epochs | LR    | Optimizer | Parameters | Description                           |
| ---- | ----------------- | ------ | ----- | --------- | ---------- | ------------------------------------- |
| 1    | 92.64%            | 10     | 0.001 | Adam      | 1,997,034  | Same as Model 3                       |
| 2    | 89.54%            | 10     | 0.005 | SGD       | 1,997,034  | Same as Model 3                       |
| 3    | 87.16%            | 10     | 0.001 | SGD       | 1,997,034  | 6 Conv layers, 3 Pooling layers, 3 FC |
| 4    | 85.54%            | 5      | 0.001 | SGD       | 3,226,474  | 2 Conv layers, 1 Pool, 2 FC           |
| 5    | 85.45%            | 20     | 0.001 | SGD       | 25,874     | 3 FC Layers                           |
| 6    | 83.20%            | 10     | 0.001 | SGD       | 25,474     | Simplest MLP with 2 FC layers         |

➡️ CNNs extracted spatial features effectively, outperforming the fully connected models on both accuracy and generalization.

➡️ Even with the same architecture and number of parameters, switching from SGD to Adam improved accuracy from 89.54% → 92.64%.

## Technologies Used

- Python 3.7
- Libraries:
  - `tensorflow`, `keras`
  - `numpy`, `pandas`
  - `matplotlib`
  - `scikit-learn` for metrics and data handling

## How to Run

```bash
# Clone the repository
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>

# Create Python environment
conda create -n my_env python=3.7
conda activate my_env

# Install dependencies
pip install tensorflow keras matplotlib numpy pandas scikit-learn

# Run the notebook
jupyter notebook Assi.ipynb
```
