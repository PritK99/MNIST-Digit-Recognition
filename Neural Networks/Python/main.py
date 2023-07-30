import numpy as np
import random

# loading the MNIST dataset

# driver class for neural networks
def Network ():
    def __init__(self, sizes):
        self.sizes = sizes
        self.num_layers = len(sizes)
        self.weights = [np.random.randn(x,y) for x,y in zip(sizes[1:], sizes[:-1])]
        self.baises = [np.random.randn(y,1) for y in sizes[1:]]
    
    def feedforward(self, a):
        for i in range(self.num_layers - 1):
            z = np.dot(self.weights[i], a) + self.biases[i]
            a = sigmoid(z)
        return a
    
    def SGD(self, training_data, epochs, mini_batch_size, learning_rate, test_data = None):
        n = len(training_data)
        if (test_data):
             n_test = len(test_data)
        for i in range (epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+ mini_batch_size] for k in range (0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, learning_rate)
                if (test_data):
                    print("Completing epoch {0}: {1}/{2}".format(i, self.evaluate(test_data), n_test))
                else:
                    print("Completing epoch {0}".format(i))

    def update_mini_batch():
        return 0
    
    def evaluate(self, test_data):
        test_results = [(np.argmax(feedforward(x)),y) for (x,y) in test_data]
        return sum(int(x==y) for (x,y) in test_results)
    

# helper functions
def sigmoid(z):
        return 1/(1 + np.exp(-z))

def sigmoid_prime(z):
     return sigmoid(z)*(1-sigmoid(z))

# driver code
if __name__ == "__main__":
     training_data, validation_data, test_data = 1