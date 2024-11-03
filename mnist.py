import random
from PIL import Image
import math
from struct import pack, unpack

random.seed(1337)

debug_print = print

LEARNING_RATE = 0.01

# Sigmoid(0) == 0.5
# So if we initialize the weights and biases to 0, the output of the neuron will be 0.5
def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x)*(1 - sigmoid(x))

def load_label(fname) -> list[int]:
    with open(fname, 'rb') as f:
        magic = unpack('>I', f.read(4))[0]

        assert(magic == 0x00000801)
        number_of_label = unpack('>I', f.read(4))[0]

        debug_print('Number of label: ', number_of_label)

        # list will convert bytes to number
        return list(f.read(number_of_label))

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

        return images

def display_image(image):
    # display the image using PIL
    img = Image.new('L', (28, 28))
    img.putdata(image)
    img = img.resize((128, 128), Image.Resampling.LANCZOS)
    img.show()


def matrix_add(a, b):
    assert(len(a) == len(b))
    assert(len(a[0]) == len(b[0]))

    result = [[0 for _ in range(len(a[0]))] for _ in range(len(a))]
    for i in range(len(a)):
        for j in range(len(a[0])):
            result[i][j] = a[i][j] + b[i][j]

    return result

def matrix_sub(a, b):
    assert(len(a) == len(b))
    assert(len(a[0]) == len(b[0]))

    result = [[0 for _ in range(len(a[0]))] for _ in range(len(a))]
    for i in range(len(a)):
        for j in range(len(a[0])):
            result[i][j] = a[i][j] - b[i][j]

    return result

def array_add(a, b):
    assert(len(a) == len(b))

    result = [0]*len(a)
    for i in range(len(a)):
        result[i] = a[i] + b[i]

    return result

def array_sub(a, b):
    assert(len(a) == len(b))

    result = [0]*len(a)
    for i in range(len(a)):
        result[i] = a[i] - b[i]

    return result


class MNISTModel: ...

class Layer:
    def __init__(self, model: MNISTModel, idx: int, dim: int, prev_dim: int):
        self.idx = idx
        self.dim = dim
        self.prev_dim = prev_dim
        self.model = model

        # The size of the current layer will be the number of columns of the weight matrix
        # The size of the previous layer will be the number of rows of the weight matrix

        self.weights = [[random.random() for _ in range(dim)] for _ in range(prev_dim)]
        self.biases = [random.random() for _ in range(dim)]

        # z is the output of the neuron before applying the activation function
        self.z = [0]*dim
        self.activations = [0]*dim

        # Gradient of the biases and weights
        self.bias_gradients = [0]*dim
        self.weight_gradients = [[0 for _ in range(dim)] for _ in range(prev_dim)]

    def init(self):
        pass

    def update_params(self):
        for i in range(self.dim):
            self.biases[i] -= LEARNING_RATE * self.bias_gradients[i]
            self.bias_gradients[i] = 0

            for j in range(self.prev_dim):
                self.weights[j][i] -= LEARNING_RATE * self.weight_gradients[j][i]
                self.weight_gradients[j][i] = 0

    def set_input(self, input):
        assert(self.idx == 0)
        assert(len(input) == self.dim)

        for i in range(self.dim):
            self.activations[i] = input[i]/255
    # Feed forward will take in the activations of the previous layer
    # And weights and biases of the connections between the previous layer and the current layer
    # And calculate the activations of the current layer
    def feed_forward_neuron(self, neuron_idx, input):
        self.activations[neuron_idx] = 0
        
        # It is reset below anyway but just to make sure we don't change the code and messup later
        self.z[neuron_idx] = 0

        # Input is the activations of the previous layer
        for i in range(self.prev_dim):
            self.activations[neuron_idx] += input[i]*self.weights[i][neuron_idx]
        
        # Add the bias
        self.activations[neuron_idx] += self.biases[neuron_idx]

        # Save the output of the neuron before applying the activation function
        self.z[neuron_idx] = self.activations[neuron_idx]

        # Apply the activation function
        self.activations[neuron_idx] = sigmoid(self.activations[neuron_idx])

    def feed_forward(self, input):
        assert(self.idx > 0)

        for i in range(self.dim):
            self.feed_forward_neuron(i, input)


    # Gradient is the value idealy we want to add to the activation of this neuron
    # In math term it is the derivative of the cost function with respect to the activation of this neuron
    def back_propagation_neuron(self, neuron_idx, gradient, new_gradient):
        # Update biases gradient
        bias_gradient = gradient * sigmoid_derivative(self.z[neuron_idx])
        self.bias_gradients[neuron_idx] += bias_gradient

        # Update weights gradient
        for i in range(self.prev_dim):
            weight_gradient = sigmoid_derivative(self.z[neuron_idx]) * gradient * self.model.layers[self.idx-1].activations[i]
            self.weight_gradients[i][neuron_idx] += weight_gradient

        # Calcuate new gradient for the previous layer
        for i in range(self.prev_dim):
            new_gradient[i] += self.weights[i][neuron_idx] * sigmoid_derivative(self.z[neuron_idx]) * gradient

    # Gradient is the desired change in the activations of the current layer
    def back_propagation(self, gradient):
        assert(self.idx > 0)

        new_gradient = [0]*self.prev_dim
        for i in range(self.dim):
            self.back_propagation_neuron(i, gradient[i], new_gradient)
        return new_gradient

class MNISTModel:
    MODEL_DIM = (784, 16, 16, 10)
    def __init__(self):
        self.layers = list()

        for idx, dim in enumerate(self.MODEL_DIM):
            self.layers.append(Layer(self, idx, dim, self.MODEL_DIM[idx-1] if idx > 0 else 0))

    def init(self):
        pass

    def set_input(self, input):
        assert(len(input) == self.MODEL_DIM[0])

        self.layers[0].set_input(input)

    def feed_forward(self):
        for i in range(1, len(self.layers)):
            self.layers[i].feed_forward(self.layers[i-1].activations)

    def train(self):
        pass

    def print_output(self):
        debug_print(self.layers[-1].activations)

    def print_input(self):
        debug_print(self.layers[0].activations)

    def cost(self, label: int):
        cost = 0
        for i in range(10):
            if i == label:
                cost += (1 - self.layers[-1].activations[i])**2
            else:
                cost += (0 - self.layers[-1].activations[i])**2
        return cost

    def back_propagation(self, label: int):
        result = [0]*10
        result[label] = 1

        gradient = [2 * (self.layers[-1].activations[i] - result[i]) for i in range(len(result))]

        # debug_print(result)
        # debug_print(self.layers[-1].activations)
 
        for i in range(len(self.layers)-1, 0, -1):
            debug_print('gradient: ', gradient)
            gradient = self.layers[i].back_propagation(gradient)

    def update_params(self):
        for i in range(1, len(self.layers)):
            self.layers[i].update_params()

# Set to 1 for debugging
BATCH_SIZE = 1

if __name__ == '__main__':
    labels = load_label('data\\train-labels.idx1-ubyte')
    images = load_image('data\\train-images.idx3-ubyte')

    training_data = list(zip(images, labels))

    # Create a training batch
    random.shuffle(training_data)

    model = MNISTModel()
    for i in range(0, len(training_data), BATCH_SIZE):
        batch = training_data[i:i+BATCH_SIZE]

        debug_print(batch[0][1])

        for image, label in batch:
            model.set_input(image)
            # debug_print('input: ', model.layers[0].activations)
            model.feed_forward()
    
            debug_print('output: ', model.layers[-1].activations)
            debug_print('cost: ', model.cost(label))
            model.back_propagation(label)
            model.update_params()

            model.set_input(image)
            # debug_print('input: ', model.layers[0].activations)
            model.feed_forward()

            debug_print('output: ', model.layers[-1].activations)
            debug_print('cost: ', model.cost(label))
            break

        break


    # display_image(images[2])

    # model = MNISTModel()
    # model.set_input(images[2])
    # model.print_input()
    # model.feed_forward()
    # model.print_output()
    # debug_print('label: ', labels[2])
    # debug_print('cost: ', model.cost(labels[2]))