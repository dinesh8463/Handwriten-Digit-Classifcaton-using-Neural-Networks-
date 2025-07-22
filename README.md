# Handwritten Digit Classification using Neural Networks

This project implements a fully connected neural network from scratch to classify handwritten digits from the MNIST dataset. 
It does **not** use any external deep learning libraries like TensorFlow or PyTorch.


##  Project Overview

- **Goal**: Classify digits (0–9) using a feedforward neural network.
- **Dataset**: [MNIST handwritten digit dataset](http://yann.lecun.com/exdb/mnist/)
- **Architecture**: Multi-layer perceptron (MLP) with:
  - Input layer: 784 units (28x28 pixel images)
  - Hidden layers: Two layers with ReLU activation
  - Output layer: 10 units with Softmax activation
- **Loss Function**: Cross-entropy loss
- **Training Method**: Mini-batch Stochastic Gradient Descent (SGD)


##  Features

- Custom implementation of:
  - Forward propagation
  - Backward propagation (manual gradient computation)
  - ReLU and Softmax activations
  - Cross-entropy loss
- Mini-batch training with customizable batch size and learning rate
- Accuracy evaluation on validation and test sets

##  Dependencies

- Python 3.x
- NumPy
- Matplotlib (for optional plotting)

Install the required dependencies using:

```bash
pip install numpy matplotlib


## Results
Achieved high classification accuracy without any external machine learning libraries.
Demonstrates the core principles behind deep learning: forward/backward passes, activation functions, gradient descent, and generalization.


