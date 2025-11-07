🧠 Neural Network Performance Comparison on MNIST Dataset

📘 Project Overview

In this project, multiple neural network models with different architectures were implemented to evaluate their performance on the MNIST dataset. The process consisted of three main stages:

- Baseline Model: Develop a neural network without convolutional layers to perform the classification task, then modify the model structure to enhance its performance.

- Convolutional Model: Construct a neural network incorporating convolutional layers, and experiment with varying the number of layers, filters, and activation functions to further improve results.

- Optimization Tuning: Adjust the optimizer type and learning rate used in previous models to observe and analyze their impact on overall model performance.

🧩 Stage 1: Neural Networks Without Convolutional Layers

🔹 Model 1 — Basic Fully Connected Network

Training Accuracy: 84.75%

Testing Accuracy: 83.20%

🔹 Model 2 — Deeper Fully Connected Network

Training Accuracy: 87.29%

Testing Accuracy: 85.45%

🧠 Stage 2: Neural Networks With Convolutional Layers
🔹 Model 1 — Basic CNN

Training Accuracy: 86.77%

Testing Accuracy: 85.54%

🔹 Model 2 — Deeper CNN

Training Accuracy: 88.99%

Testing Accuracy: 87.16%

⚙️ Stage 3: Optimization Experiments
🔹 Model 1 — Change Optimizer to Adam

Training Accuracy: 97.54%

Testing Accuracy: 92.64%

🔹 Model 2 — Higher Learning Rate with SGD

Training Accuracy: 92.71%

Testing Accuracy: 89.54%

🏁 Summary of Results
Rank Test Accuracy (%) Epoch Learning Rate Optimizer Parameters Description
🥇 1 92.64% 10 0.001 Adam 1,997,034 Same as Model 3 but optimized with Adam
🥈 2 89.54% 10 0.005 SGD 1,997,034 Same as Model 3 but higher LR
🥉 3 87.16% 10 0.001 SGD 1,997,034 6 Conv layers, 3 Pooling layers, 3 FC
4 85.54% 5 0.001 SGD 3,226,474 2 Conv layers, 1 Pool, 2 FC
5 85.45% 20 0.001 SGD 25,874 3 FC Layers
6 83.20% 10 0.001 SGD 25,474 Simplest MLP with 2 FC layers

🧾 Environment & Dependencies

This project was implemented using the following environment and package versions:

Library Version Purpose
Python 3.7.1 Core environment
TensorFlow 2.17.0 Deep learning backend
Keras 3.5.0 Model building API
NumPy 1.26.3 Numerical operations
Matplotlib 3.8.2 Plotting training accuracy/loss
Pandas 2.1.4 Data handling
Scikit-learn 1.5.1 Metrics and evaluation support
Jupyter Notebook 6.29.0 (via ipykernel) Interactive experimentation

🧠 Note:
Older Python (3.7.x) and TensorFlow 2.x were used to ensure compatibility with legacy systems and assignment constraints.
If you’re running this on a newer environment, minor version mismatches may occur, but the notebook is fully reproducible.

💻 How to Run

Clone the repository

git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>

Create a Python environment

conda create -n CSE3CI_env python=3.7  
conda activate CSE3CI_env

Install dependencies

pip install tensorflow keras matplotlib numpy

Run the notebook

jupyter notebook Assi.ipynb

📚 Dependencies

Python 3.7

TensorFlow / Keras

NumPy

Matplotlib
