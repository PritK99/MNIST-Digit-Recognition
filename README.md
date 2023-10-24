# MNIST-Digit-Recognition

<img src = "assets/MNIST.png">

## Table of Contents

- [Project](#MNIST-Digit-Recognition)
  - [Table of Contents](#table-of-contents)
  - [About The Project](#about-the-project)
  - [File Structure](#file-structure)
  - [Description](#Description)
  - [Getting started](#Getting-Started)
  - [Contributors](#contributors)
  - [Acknowledgements and Resources](#acknowledgements-and-references)
  - [License](#license)

## About

Implementing Deep Learning for Handwritten Digit Recognition from scratch

## File Structure

## Description
### Approach 1: Using Vanilla Neural Networks

#### Method 1: Using SGD learning algorithm

##### Description

Implementing <b>stochastic gradient descent</b> learning algorithm for a vanilla neural network. Here, the gradients are calculated using backpropagation algorithm. The model uses the <b>quadratic cost function (Mean Squared Error)</b> and the sigmoid activation function for all the layers except the input layer.

##### Usage

```python vanilla_neural_network.py```

##### Results

* Using two hidden layers with 100 and 30 neurons respectively
<img src = "assets/nn-with-100-30-hidden-layers.png">
<img src = "assets/vanilla-neural-network.png">

Note: The results might vary since the weights are initialized randomly everytime.
#### Method 2: Improving the model

##### Description
##### Usage

```python neural_network.py```

##### Results

## Getting Started

### Prerequisites
To download and use this code, the following python libraries are required:

* ```numpy```
* ```matplotlib```
* ```pickle```
* ```gzip```

### Installation

Clone the project by typing the following command in your Terminal/CommandPrompt

```
git clone https://github.com/PritK99/MNIST-Digit-Recognition
```
Navigate to the MNIST-Digit-Recognition folder

```
cd MNIST-Digit-Recognition
```

### Usage

Once the requirements are satisfied, use the following commands to run the respective python file:

```
python <name_of_the_file.py>
```


## Contributors

* [Prit Kanadiya](https://github.com/PritK99)

## Acknowledgements and Resources

* <a href = "http://neuralnetworksanddeeplearning.com/index.html" >Neural Network and Deep Learning</a> by Michael Nielsen
* ```Improving Deep Neural Networks: Hyperparameter Tuning, Regularization and Optimization``` by ```deeplearning.ai``` delivered through Coursera

## License
[MIT License](https://opensource.org/licenses/MIT)