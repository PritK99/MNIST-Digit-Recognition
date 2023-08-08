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

# plotting graph for accuracy vs epochs
def plot(accuracy):
    x = [round(val*0.0001, 2) for val in accuracy]
    plt.plot(range(len(x)), x)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('logistic-regression')
    plt.savefig("logistic-regression.png")
    plt.show()

# implementing logistic regression
class Network ():
    def __init__(self, sizes):
        self.sizes = sizes
        self.input_size = sizes[0]
        self.output_size = sizes[1]
        self.weights = np.zeros((self.output_size, self.input_size), dtype=float)
        self.bias = np.zeros((1,1), dtype=float)
        self.accuracy = []
    
    def feedforward(self, a):
        z = np.dot(self.weights, a) + self.bias
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
                x = self.evaluate(test_data)
                print("Completing epoch {0}: {1}/{2}".format(i, x, n_test))
                self.accuracy.append(x)
            else:
                print("Completing epoch {0}".format(i))

    def update_mini_batch(self, mini_batch, learning_rate):
        nabla_b = np.zeros(1)
        nabla_w = np.zeros(len(self.weights))
        n = len(mini_batch)
        for (x,y) in mini_batch:
            delta_nabla_w, delta_nabla_b = self.backprop(x,y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(learning_rate/len(mini_batch))*nw for w, nw in zip(self.weights, nabla_w)]
        self.bias = [b-(learning_rate/len(mini_batch))*nb for b, nb in zip(self.bias, nabla_b)]

    def cost_derivative(self, output, y):
        return (output - y)
    
    def backprop(self, x, y):
        delta_nabla_b = np.zeros(1)
        delta_nabla_w = np.zeros(len(self.weights))
        #forward pass
        a = x
        activations = [x]
        z_cache = []

        z = np.dot(self.weights,a) + self.bias
        z_cache.append(z)
        a = sigmoid(z)
        activations.append(a)
        #backward pass
        delta = self.cost_derivative(activations[-1], y)*sigmoid_prime(z_cache[-1])
        delta_nabla_b = delta
        delta_nabla_w = np.dot(delta, activations[-2].transpose())
        return delta_nabla_w, delta_nabla_b

    
    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

# helper functions
def sigmoid(z):
    return 1.0/(1.0 + np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))

# driver code
if __name__ == "__main__":
    training_data, validation_data, test_data = load_data_wrapper()
    net = Network([784,10])
    net.SGD(training_data, 40, 10, 0.1, test_data=test_data)
    plot(net.accuracy)