import numpy as np
import random
import gzip
import pickle

# loading the MNIST dataset
def load_data() :
    f = gzip.open("../../data/mnist.pkl.gz", "rb")
    training_data, validation_data, test_data = pickle.load(f)
    f.close()
    return training_data, validation_data, test_data

def load_data_wrapper():
    tr_d, va_d, te_d = load_data()
    training_inputs = [np.reshape(x, (784,1)) for x in tr_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data = zip(training_inputs, training_results)
    test_inputs = [np.reshape(x, (784,1)) for x in te_d[0]]
    test_results = [vectorized_result(y) for y in te_d[1]]
    test_data = zip(test_inputs, test_results)
    validation_inputs = [np.reshape(x, (784,1)) for x in va_d[0]]
    validation_results = [vectorized_result(y) for y in va_d[1]]
    validation_data = zip(validation_inputs,validation_results)

    return training_data, validation_data, test_data

def vectorized_result(num):
    output = np.zeros((10,1))
    output[num] = 1
    return output

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

    def update_mini_batch(self, mini_batch, learning_rate):
        nabla_b = [np.zeros(b.shape()) for b in self.biases]
        nabla_w = [np.zeros(w.shape()) for w in self.weights]
        n = len(mini_batch)
        for (x,y) in mini_batch:
            delta_nabla_w, delta_nabla_b = self.backprop(x,y)
            nabla_b = nabla_b + delta_nabla_b
            nabla_w = nabla_w + delta_nabla_w
        db = nabla_b/n
        dw = nabla_w/n
        self.weights = self.weights - learning_rate*dw
        self.baises = self.biases - learning_rate*db

    def backprop(self, x, y):
        return 0 #pending
    
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
    training_data, validation_data, test_data = load_data_wrapper()
    net = Network([784,30,10])