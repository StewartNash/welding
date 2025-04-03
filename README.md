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

