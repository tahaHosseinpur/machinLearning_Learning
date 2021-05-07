#i am just training what i have underestood here
#it represents a classification problem which find out what number is written

import random

#### Libraries
# Standard library
import pickle as cPickle
import gzip

# Third-party libraries
import numpy as np

#this is actually creats our network
class Network(object):
     #it gets the sizes of our network layers when the obj is called   
        def __init__(self, sizes):
            #our number of layers are included
            self.num_layers = len(sizes)
            #here just we pass the array of sizes
            self.sizes = sizes
            #we generate random biases (but not for input layer) starting from the seccond layer random.randn generates random number in a matrix with n number and 1 column n = size of layer
            self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
            #well here we generate random weights or parameters for layers it works like this (784 , 30) , (30 , 10) where we the first item is columns and seccond is rows 
            self.weights = [np.random.randn(y, x) 
                            for x, y in zip(sizes[:-1], sizes[1:])]
        # it returns an array of our outputs
        def feedTheLayers(self , a):
            for b , w in zip(self.biases , self.weights):    
            a = sigmoid(np.dot(w,a) + b)
            return a;
        
        def SGD(self , training_data , epochs , mini_batch_size , eta ,test_data=None):
            if test_data: n_test = len(test_data)
            n = len(training_data)
            # repeats for n timres
            for j in range(epochs):
                #shuffle our training data so we have randomized inputs for mini_batches
                random.shuffle(training_data)
                # is an array of mini_batch in which randomized trainig_datas are in it
                mini_batches = [training_data[k:k+mini_batch_size] for k in range(0,n,mini_batch_size)]
                for mini_batch in mini_batches:
                    self.update_mini_batche(mini_batch , eta)
                if test_data:
                    print("epoch")
                    print(str(j))
                    print(str(self.evaluate(test_data)))
                    print(str(n_test))
                else:
                    print("Epoch complete")
        
        def update_mini_batche(self, mini_batch , eta):
                    # array of biases and each value is zero
                    nabla_b = [np.zeros(b.shape) for b in self.biases]
                    # same as above
                    nabla_w = [np.zeros(w.shape) for w in self.weights]
                    # do it for each item in mini bach
                    for x, y in mini_batch:
                        delta_nabla_b, delta_nabla_w = self.backprop(x, y)
                        # updates the array 
                        nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
                        nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]    
                    # does the desenct of items      
                    self.weights= [w-(eta/len(mini_batch))*nw for w, nw in zip(self.weights, nabla_w)]
                    self.biases = [b-(eta/len(mini_batch))*nb for b, nb in zip(self.biases, nabla_b)]

        def backprop(self, x, y):
                    # arrays with value of zero
                    nabla_b = [np.zeros(b.shape) for b in self.biases]
                    nabla_w = [np.zeros(w.shape) for w in self.weights]
                    activation = x
                    #array of inputs ot activations
                    activations = [x]
                    zs = [] 
                    #it does it for evey layer
                    for b, w in zip(self.biases, self.weights):
                        #array of Z or inputs of new layer 
                        z = np.dot(w, activation)+b
                        zs.append(z)
                        #array of output of new layer 
                        activation = sigmoid(z)
                        activations.append(activation)
                     # it gets the delta of last layer    
                    delta = self.cost_derivative(activations[-1], y) * \
                        sigmoid_prime(zs[-1])
                    # retuns to last item of array
                    nabla_b[-1] = delta
                    nabla_w[-1] = np.dot(delta, activations[-2].transpose())
                    for l in range(2, self.num_layers):
                        z = zs[-l]
                        sp = sigmoid_prime(z)
                        delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
                        nabla_b[-l] = delta
                        nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
                    return (nabla_b, nabla_w)
         #retuns      
        def cost_derivative(self, output_activations, y):
                    return (output_activations-y)      

        def evaluate(self, test_data):
            test_results = [(np.argmax(self.feedTheLayers(x)), y)
                            for (x, y) in test_data]
            return sum(int(x == y) for (x, y) in test_results)            
        
#here is simply a defined func to return sigmoid function using numpy
def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))


def load_data():
    f = gzip.open('E:\ML projc\data\mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = cPickle.load(f , encoding='latin1')
    f.close()
    return (training_data, validation_data, test_data)

def load_data_wrapper():
    tr_d, va_d, te_d = load_data()
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data = list(zip(training_inputs, training_results))
    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data = list(zip(validation_inputs, va_d[1]))
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = list(zip(test_inputs, te_d[1]))
    return (training_data, validation_data, test_data)

def vectorized_result(j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e


def main():
    training_data, validation_data, test_data = load_data_wrapper()
    net = Network([784, 30, 10])
    net.SGD(training_data, 30, 10, 3.0, test_data=test_data)
    

