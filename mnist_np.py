import random
import sys, os
import math
import pickle
import numpy as np

from typing import List
from struct import pack, unpack
from PIL import Image

class Model: ...

def null(*args):
    pass


# debug_print = print
debug_print = null

BATCH_SIZE = 10
MNIST_DIM = (784, 16, 16, 10)
LEARNING_RATE = 0.01

def load_label(fname) -> np.ndarray[np.float64]:
    with open(fname, 'rb') as f:
        magic = unpack('>I', f.read(4))[0]

        assert(magic == 0x00000801)
        number_of_label = unpack('>I', f.read(4))[0]

        debug_print('Number of label: ', number_of_label)

        # list will convert bytes to number
        return np.array(list(f.read(number_of_label)), dtype=np.int64)

def load_image(fname):
    with open(fname, 'rb') as f:
        magic = unpack('>I', f.read(4))[0]

        assert(magic == 0x00000803)
        number_of_image = unpack('>I', f.read(4))[0]
        number_of_row = unpack('>I', f.read(4))[0]
        number_of_col = unpack('>I', f.read(4))[0]

        debug_print('Number of image: ', number_of_image)
        debug_print('Number of row: ', number_of_row)
        debug_print('Number of col: ', number_of_col)

        assert(number_of_row == 28)
        assert(number_of_col == 28)

        images = []
        for _ in range(number_of_image):
            # list will convert bytes to number
            images.append(list(f.read(number_of_row * number_of_col)))

        return np.array(images, dtype=np.float64)

def normalize_input(input):
    return ((input * (2.0 / 255.0)) - 1.0).reshape(len(input), 1)

def sigmoid(a: np.ndarray[np.float64]):
    # we keep making new array instead of doing inplace operations
    # perf ?
    a = a.clip(-500, 500)

    # perf
    return 1/(1 + np.exp(-a))

def sigmoid_deri(input: np.ndarray[np.float64]):
    s = sigmoid(input)
    return s*(1-s)

af = sigmoid
af_deri = sigmoid_deri

def display_image(image):
    # display the image using PIL
    img = Image.new('L', (28, 28))
    img.putdata(image)
    img = img.resize((128, 128), Image.Resampling.LANCZOS)
    img.show()

class LinearLayer:
    def __init__(self, model: Model, idx: int, size: int, prev_layer_size: int):
        self.idx = idx
        self.size = size
        self.prev_layer_size = prev_layer_size
        self.model = model

        # zero init activations
        # activations is a 1 dimension array
        self.activations: np.ndarray

        # z is the activations before applying activation function
        # we have to save it in order to backprop through our sigmoid function 
        self.z = np.zeros(size)

        # weights
        # each neuron in one layer have multiple connections to neurons in previous layer
        # hence weights is a 2 dimensions array

        # eg: weights[0][3] will be neuron-0 of current layer to neuron-3 of previous layer
        self.weights = np.random.uniform(-0.1, 0.1, size=(size, prev_layer_size))
        
        self.weight_gradients = np.zeros((self.prev_layer_size, self.size))
        
        # column matrix
        '''
        >>> np.ones((3, 1))
        array(
            [[1.],
            [1.],
            [1.]]
        )
        '''
        self.biases = np.random.uniform(0, 0.1, size=(size, 1))

        # For debugging
        # self.weights = np.ones((size, prev_layer_size))
        # self.biases = np.ones((size, 1))

        self.bias_gradients = np.zeros((size, 1))


    def set_input(self, input: np.ndarray[np.float64]):
        assert(self.idx == 0)
        assert(input.size == self.size)

        self.activations = normalize_input(input)

    # input here is activation fron prev_layer
    def forward(self, input):
        assert(self.idx > 0)

        # we have to reset activations of previous input
        self.activations = np.zeros(self.size)

        # calcuate z
        self.z = self.weights @ input  + self.biases

        # apply activation function
        self.activations = af(self.z)

    def average_gradients(self):
        self.bias_gradients /= BATCH_SIZE
        self.weight_gradients /= BATCH_SIZE

    def update_params(self):
        debug_print('update_params')
        
        self.biases -= LEARNING_RATE*self.bias_gradients
        self.bias_gradients[:] = 0

        
        self.weights-= LEARNING_RATE * self.weight_gradients.T        
        self.weight_gradients[:] = 0

    def backward(self, gradient):
        assert(self.idx > 0)

        # backprop through the sigmoid function
        gradient = gradient * af_deri(self.z)
        shape = gradient.shape
        debug_print(f'{gradient.shape=}')

        # Accumulate bias gradient
        self.bias_gradients += gradient

        prev_layer_activations = self.model.layers[self.idx-1].activations

        debug_print(f'{prev_layer_activations.shape=}')

        
        # y = w*x + b 
        # dy/dw = x
        self.weight_gradients += prev_layer_activations @ gradient.T


        # dy/dx = w
        new_gradient = self.weights.T @ gradient

        return new_gradient

class MNISTModel:
    def __init__(self, model_dim):
        # should layers be a ndarray too ?
        # if layers is a ndarray we have to get rid of the Layer class ?
        # will it give better perf ?
        self.layers: List[LinearLayer] = list()

        self.model_dim = model_dim

        for idx, layer_size in enumerate(self.model_dim):
            self.layers.append(LinearLayer(self, idx, layer_size, self.model_dim[idx-1] if idx > 0 else 0))

        self.output_layer = self.layers[-1]
        self.input_layer = self.layers[0]

    def feed_forward(self):
        for i in range(1, len(self.layers)):
            self.layers[i].forward(self.layers[i-1].activations)

    def set_input(self, input):
        self.layers[0].set_input(input)

    def print_output(self):
        debug_print(self.layers[-1].activations)

    def cost(self, label: int):
        result = np.zeros((10, 1))
        result[label] = 1

        return sum((self.output_layer.activations - result)**2)       
 
    def backward(self, label: int):
        debug_print('backward')
        
        result = np.zeros((10, 1))
        result[label] = 1

        gradient = 2*(self.output_layer.activations - result)
        debug_print(f'{gradient.shape=}')


        for i in range(len(self.layers)-1, 0, -1):
            gradient = self.layers[i].backward(gradient)

    def print_output(self):
        print(self.layers[-1].activations)

    def get_result_label(self):
        return np.argmax(self.layers[-1].activations)

RUNNING = True
def train():
    global RUNNING
    global BATCH_SIZE

    labels = load_label(os.path.join('data', 'train-labels.idx1-ubyte'))
    images = load_image(os.path.join('data', 'train-images.idx3-ubyte'))

    training_data = list(zip(images, labels))

    # Create a training batch
    if len(sys.argv) == 3:
        model_file = sys.argv[2]
        with open(model_file, 'rb') as f:
            model = pickle.load(f)
    else:
        model = MNISTModel(MNIST_DIM)

    count = 0
    while RUNNING:
        random.shuffle(training_data)
        for batch_idx in range(0, len(training_data), BATCH_SIZE):
            try:
                debug_print(f'Batch {batch_idx}/{len(training_data)}')
                batch = training_data[batch_idx:batch_idx+BATCH_SIZE]

                # debug_print(batch[0][1])

                for image, label in batch:
                    model.set_input(image)
                    # debug_print('input: ', model.layers[0].activations)
                    model.feed_forward()
            
                    count += 1
                    if count % 1000 == 0:
                        print('cost: ', model.cost(label))
                    model.backward(label)

                    # Test to make sure that const function is actually decreasing
                    
                    # model.update_params()
                    # model.set_input(image)
                    # debug_print('input: ', model.layers[0].activations)
                    # model.feed_forward()
                    # debug_print('output: ', model.layers[-1].activations)
                    # debug_print('cost: ', model.cost(label))

                    # input('>')

                # Now we to average the gradients and update the weights and biases
                # If we have keyboard interrupt here ignore it because it will break our model
                try:
                    for i in range(1, len(model.layers)):
                        model.layers[i].average_gradients()

                    for i in range(1, len(model.layers)):
                        model.layers[i].update_params()
                except KeyboardInterrupt:
                    pass
            except KeyboardInterrupt:
                RUNNING = False
                break

    print ('saving model')
    with open('model_np.pkl', 'wb') as f:
        pickle.dump(model, f)    


if __name__ == '__main__':
    if sys.argv[1] == 't':
        train()
    
    elif sys.argv[1] == 'test':
        model = MNISTModel()

        labels = load_label('data\\t10k-labels.idx1-ubyte')
        images = load_image('data\\t10k-images.idx3-ubyte')

        model.set_input(images[0])
        model.feed_forward()

        model.print_output()

    elif sys.argv[1] == 'i':
        with open(sys.argv[2], 'rb') as f:
            model = pickle.load(f)

        labels = load_label(os.path.join('data', 't10k-labels.idx1-ubyte'))
        images = load_image(os.path.join('data', 't10k-images.idx3-ubyte'))

        test_data = list(zip(images, labels))
        random.shuffle(test_data)
        for image, label in test_data:
            display_image(image)
            model.set_input(image)
            model.feed_forward()
            model.print_output()
            print('predict:', model.get_result_label())
            print('label: ', label)
            print('cost: ', model.cost(label))

            input('>')