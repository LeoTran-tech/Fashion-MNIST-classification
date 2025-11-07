🧠 Neural Network Performance Comparison on MNIST Dataset
📘 Overview

This project explores and compares multiple neural network architectures to evaluate their performance on the MNIST handwritten digit dataset.
The workflow was divided into three main stages, each focusing on different architectural and optimization improvements.

Project Stages

Baseline Model – Build a neural network without convolutional layers to perform classification. Then modify the architecture to enhance performance.

Convolutional Model – Integrate convolutional layers and experiment with the number of filters, layers, and activation functions.

Optimization Tuning – Adjust optimizers and learning rates to evaluate their impact on convergence speed and accuracy.

🧩 Stage 1: Neural Networks Without Convolutional Layers
Model Description Training Accuracy Testing Accuracy
Model 1 Basic Fully Connected Network 84.75% 83.20%
Model 2 Deeper Fully Connected Network 87.29% 85.45%

💡 Increasing network depth improved performance slightly, but convergence remained slow without convolutional feature extraction.

🧠 Stage 2: Neural Networks With Convolutional Layers
Model Description Training Accuracy Testing Accuracy
Model 1 Basic CNN (2 Conv + 1 Pool + 2 FC) 86.77% 85.54%
Model 2 Deeper CNN (6 Conv + 3 Pool + 3 FC) 88.99% 87.16%

🧩 Adding convolutional layers significantly improved accuracy and generalization compared to fully connected networks.

⚙️ Stage 3: Optimization Experiments
Model Optimizer Learning Rate Training Accuracy Testing Accuracy
Model 1 Adam 0.001 97.54% 92.64%
Model 2 SGD 0.005 92.71% 89.54%

⚡ Fine-tuning the optimizer and learning rate had a major impact on model performance.
Adam achieved the highest test accuracy and stable convergence.

🏁 Summary of Results
🥇 Rank Test Accuracy (%) Epochs LR Optimizer Parameters Description
🥇 1 92.64% 10 0.001 Adam 1,997,034 Same as Model 3 but optimized with Adam
🥈 2 89.54% 10 0.005 SGD 1,997,034 Same as Model 3 but with higher learning rate
🥉 3 87.16% 10 0.001 SGD 1,997,034 6 Conv layers, 3 Pooling layers, 3 FC
4 85.54% 5 0.001 SGD 3,226,474 2 Conv layers, 1 Pooling layer, 2 FC
5 85.45% 20 0.001 SGD 25,874 3 FC Layers
6 83.20% 10 0.001 SGD 25,474 Simplest MLP with 2 FC layers

🧠 The Adam-optimized CNN achieved the highest test accuracy of 92.64%, demonstrating the importance of proper optimizer and learning rate tuning.

🧾 Environment & Dependencies
Library Version Purpose
Python 3.7.1 Core environment
TensorFlow 2.17.0 Deep learning backend
Keras 3.5.0 Model building API
NumPy 1.26.3 Numerical operations
Matplotlib 3.8.2 Plotting training accuracy/loss
Pandas 2.1.4 Data handling
Scikit-learn 1.5.1 Metrics and evaluation support
Jupyter Notebook 6.29.0 Interactive experimentation

🧩 Note: This project used an older environment (Python 3.7.x, TensorFlow 2.x) to maintain compatibility with academic systems.
The notebook remains fully reproducible on modern setups with minor version adjustments.

💻 How to Run

# Clone the repository

git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>

# Create a Python environment

conda create -n CSE3CI_env python=3.7
conda activate CSE3CI_env

# Install dependencies

pip install tensorflow keras matplotlib numpy pandas scikit-learn

# Run the notebook

jupyter notebook Assi.ipynb

📚 Technologies Used

Python 3.7

TensorFlow / Keras

NumPy

Matplotlib

Pandas

Scikit-learn

Jupyter Notebook
