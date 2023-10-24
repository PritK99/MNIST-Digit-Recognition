import numpy as np
import random
import gzip
import pickle
import matplotlib.pyplot as plt

# loading the MNIST dataset
def load_data() :
    f = gzip.open("data/mnist.pkl.gz", "rb")
    u = pickle._Unpickler( f )
    u.encoding = 'latin1'
    training_data, validation_data, test_data = u.load()
    f.close()
    return training_data, validation_data, test_data

def load_data_wrapper():
    tr_d, va_d, te_d = load_data()
    training_inputs = [np.reshape(x, (784,1)) for x in tr_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data = list(zip(training_inputs, training_results))
    test_inputs = [np.reshape(x, (784,1)) for x in te_d[0]]
    test_data = list(zip(test_inputs, te_d[1]))
    validation_inputs = [np.reshape(x, (784,1)) for x in va_d[0]]
    validation_data = list(zip(validation_inputs,va_d[1]))

    return training_data, validation_data, test_data

def vectorized_result(num):
    output = np.zeros((10,1))
    output[num] = 1.0
    return output

class Network():
    def __init__(self):
        self.weights = np.random.randn(784,1)
        self.bias = np.zeros(1,)
    
    def feedforward(self, input):
        z = self.weights*input + self.bias 
        a = sigmoid(z)

# helper functions
def sigmoid(z):
    return 1.0/(1.0 + np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))

if __name__ == "__main__":
    net = Network
