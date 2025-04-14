# Welding
This is an educational repository for neural network development. 

## Introduction
We will develop a neural network that approximates the results of welding given specific input variables. We will then use an optimization algorithm operating over inputs and outputs of the neural network to find the best settings. We will generate sample information using arbitrary functions, and the data is not necessarily reflective of actual or real data. The input variables are as follows:

* electrode force
* electrode contact surface diameter
* squeeze time
* weld time
* hold time
* weld current

The output variabels are as follows:

* leak rate
* explosive force
* leaking
* explosion

Leaking and explosion are binary categories whereas leak rate and explosive force, respectively, provide a quantitative measure of these.

### Organization

This repository is broken into several sections. Jupyter notebooks are available in the notebooks folder. I recommend running them in Google Colab, as that is what I used, because Jupyter notebook was not cooperative. The code found in the Jupyter notebooks is also available in Python files in the main directory. This code will be moved to its own folder at some point. I will also add a LaTeX document which reviews pertinent theory.

## Data Generation, Collection and Labeling

Ordinarily, one would have to gather and label data. This is the most time-consuming and expensive part of training a neural network, because it involves utilizing human labelers and human gatherers of data.

In this case, we may use various methods to generate random data. (The approach described here may be different from the approach taken in the code, as the code is being constantly updated without concurrent updates to the documentation.) The first approach was to generate a linear functional dependence of the output (b) on the the input (x) using a matrix of coefficients (A) to get Ax-b. The line 'output = np.matmul(coefficients, input_row)' in the 'generate_rows' function performs this task. The coefficient matrix is randomly populated with a normal distribution in the 'generate_coefficients' function. In essence, the neural network will have to reproduce this coefficient matrix.

## Pre-processing

Data preprocessing is, arguably, the most important step in machine learning. At least, it is one of the most important steps.

## Deep Learning

The neural network is generated using Keras. Keras is an easy-to-use, high level deep learning API that allows one to easily construct neural networks.

## Optimization
Different possible optimization techniques are highlighted in each chapter of 'AI Application Programming' by M. Tim Jones: simulated annealing, particle swarm optimization, adaptive resonance theory, ant algorithm, and genetic algorithms

1. History of AI
2. Pathfinding Algorithms
3. Simulated Annealing
4. Particle Swarm Optimization
5. Clustering with Adaptive Resonance Theory (ART1)
6. Introduction to Classifier Systems
7. Ant Algorithms
8. Backpropagation Networks
9. Reinforcement Learning
10. Introduction to Genetic Algorithms
11. Artificial Life
12. Rules-Based Systems
13. Fuzzy Logic
14. Natural Language Processing
15. Hidden Markov Models
16. Intelligent Agents
17. AI Today

### Newton-Raphson Method

Newton's method or the Newton-Raphson method is an iterative method for finding the roots of a differentiable function.

### Gradient Descent

Gradient descent is an iterative algorithm that is used to minimize a function by finding the optimal parameters. To implement a gradient descent algorithm, we require the following
- Cost function to minimize
- Number of iterations
- Learning rate
    - Determine step size at each iteration
- Partial derivatives for weight and bias
    - Update parameters at each iteration
- Predict function

### Gauss-Newton Method

The Gauss-Newton algorithm is an extension of Newton's method. It is used to solve non-linear least squares problems, which is equivalent to minimizing a sum of squared function values.

The Gauss-Newton method is used in welding1.py to minimize the leakage rate and explosive force. The neural network serves as the functional model which relates the input parameters to the output parameters - leakage and explosion.
