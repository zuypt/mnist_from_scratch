import random
import sys, os
import math
import pickle
import numpy as np
import curses
import pygame

from typing import List
from struct import pack, unpack
from PIL import Image

class Model: ...

def null(*args):
    pass


# debug_print = print
debug_print = null

BATCH_SIZE = 64
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

def display_grayscale(image):
    # display the image using PIL
    img = Image.new('L', (28, 28))
    img.putdata(image)
    img = img.resize((128, 128), Image.Resampling.LANCZOS)
    img.show()

def display_rgb(image):

    
    pixel_data = [tuple(pixel) for pixel in image.reshape(-1, 3)]  # Convert each row to a tuple
    
    # display the image using PIL
    img = Image.new('RGB', (28, 28))
    img.putdata(pixel_data)
    img = img.resize((128, 128), Image.Resampling.NEAREST)
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

    def visualize(self):
        model = self

        layer_1 = model.layers[1]
        layer_2 = model.layers[2]
        layer_3 = model.layers[3]
        
        # print(f'{layer_1.weights.shape=}')
        # print(f'{layer_2.weights.shape=}')
        
        for i in range(16):
            pass
            # pygame_disp_rbg(normalize_to_rgb(layer_1.weights[i]).reshape(28, 28, 3), 30*(i//4), (30*i)%120)

        layer_1_visual = np.zeros((16, 784))
        for i in range(16):
            for j in range(16):
                layer_1_visual[i] += layer_1.weights[j]*layer_2.weights[i][j]     
            # pygame_disp_rbg(normalize_to_rgb(layer_1_visual[i]).reshape(28, 28, 3), 30*(i//4), (30*i)%120)
            
        layer_2_visual = np.zeros((10, 784))
        for i in range(10):
            for j in range(16):
                layer_2_visual[i] += layer_1_visual[j]*layer_3.weights[i][j]
            pygame_disp_rbg(normalize_to_rgb(layer_2_visual[i]).reshape(28, 28, 3), 30*(i//4), (30*i)%120)

        # running = True
        # clock = pygame.time.Clock()
        # while running:
        #     for event in pygame.event.get():
        #         if event.type == pygame.QUIT:
        #             running = False
        #     clock.tick(30)
        # pygame.quit()

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
                
                model.visualize()
                pygame.event.get()
            except KeyboardInterrupt:
                RUNNING = False
                break

    print ('saving model')
    with open('model_np.pkl', 'wb') as f:
        pickle.dump(model, f)    


def inference():
    with open(sys.argv[2], 'rb') as f:
        model = pickle.load(f)

    labels = load_label(os.path.join('data', 't10k-labels.idx1-ubyte'))
    images = load_image(os.path.join('data', 't10k-images.idx3-ubyte'))

    test_data = list(zip(images, labels))
    random.shuffle(test_data)
    for image, label in test_data:
        display_grayscale(image)
        model.set_input(image)
        model.feed_forward()
        model.print_output()
        print('predict:', model.get_result_label())
        print('label: ', label)
        print('cost: ', model.cost(label))

        input('>')


def normalize_to_grayscale(weights):
    """
    Normalize an array of weights to grayscale [0, 255].
    
    Parameters:
    -----------
    weights : numpy.ndarray
        A NumPy array (or similar) of floating-point weight values.
    
    Returns:
    --------
    np.ndarray
        A NumPy array of uint8 values in [0, 255] representing grayscale.
    """
    # Convert to a NumPy array if it's not already
    weights = np.array(weights, dtype=np.float64)
    
    w_min = np.min(weights)
    w_max = np.max(weights)
    
    # If all weights are identical, avoid division by zero:
    if np.isclose(w_min, w_max):
        return np.full_like(weights, 128, dtype=np.uint8)
    
    # Normalize to [0,1]
    normalized = (weights - w_min) / (w_max - w_min)
    # Scale to [0,255]
    grayscale = (normalized * 255).astype(np.uint8)
    
    return grayscale

def normalize_to_rgb(weights):
    """
    Map an array of floating-point values to RGB colors:
      - Negative values -> Green channel (the more negative, the brighter)
      - Positive values -> Red channel  (the more positive, the brighter)
      - Zero -> Black

    Parameters
    ----------
    weights : array-like
        A NumPy array (or similar) of floating-point weight values.

    Returns
    -------
    np.ndarray
        A NumPy array of shape (..., 3) with dtype=np.uint8, 
        where each entry is an [R, G, B] color.
    """
    # Convert to float64 NumPy array
    weights = np.array(weights, dtype=np.float64)

    # Prepare output with shape (..., 3) for RGB
    # e.g. for a 2D weights array shape=(H, W), output shape=(H, W, 3)
    output_shape = weights.shape + (3,)
    rgb = np.zeros(output_shape, dtype=np.uint8)

    # Handle the case where all values are zero to avoid a divide-by-zero
    max_abs = np.max(np.abs(weights))
    if np.isclose(max_abs, 0.0):
        return rgb  # already zero => black

    # Compute brightness as a fraction of the max absolute value
    # brightness[i] = 255 * |weights[i]| / max_abs
    brightness = np.abs(weights) / max_abs * 255
    brightness = brightness.astype(np.uint8)

    # Create boolean masks
    neg_mask = (weights < 0)
    pos_mask = (weights > 0)

    # For negative values, fill the Green channel with brightness
    rgb[neg_mask, 1] = brightness[neg_mask]
    # For positive values, fill the Red channel with brightness
    rgb[pos_mask, 0] = brightness[pos_mask]
    # Zero stays black [0,0,0]

    return rgb


def disp_grayscale(stdscr, arr: np.ndarray, x=0, y=0):
    if arr.shape != (28, 28):
        raise ValueError("Array must be exactly 28×28.")

    # The standard xterm 24-step grayscale range:
    # 232 = near-black ... 255 = near-white
    start_gray = 232
    end_gray   = 255
    num_levels = end_gray - start_gray + 1  # typically 24

    # Initialize color pairs for these 24 grayscale steps
    max_pairs = curses.COLOR_PAIRS
    usable_pairs = min(max_pairs - 1, num_levels)  # e.g. up to 24
    for i in range(usable_pairs):
        color_idx = start_gray + i
        if color_idx > end_gray:
            break
        # pair i+1 => (foreground=COLOR_BLACK, background=color_idx)
        curses.init_pair(i+1, curses.COLOR_BLACK, color_idx)

    # Draw each pixel (two spaces wide)
    for row in range(28):
        for col in range(28):
            val = int(arr[row, col]) & 0xFF  # clamp to [0..255]

            # Map val (0..255) to a grayscale level in 0..(num_levels-1)
            # e.g. val=0 -> 0 (dark), val=255 -> num_levels-1 (light)
            level = (val * num_levels) // 256
            if level == num_levels:
                level = num_levels - 1

            # Also ensure we don't exceed 'usable_pairs'
            level = min(level, usable_pairs - 1)

            pair_id = level + 1
            try:
                stdscr.addstr(row+x, (col+y) * 2, "  ", curses.color_pair(pair_id))
            except curses.error:
                # If the terminal is too small and we go out of bounds
                pass

    stdscr.refresh()


COLOR_MAP = {}  # Maps color_index to a curses pair ID

def rgb_to_256_color_index(r, g, b):
    # If the pixel is black, return curses' black color index
    if r == 0 and g == 0 and b == 0:
        return -1
    # Otherwise, use the 6x6x6 color cube conversion.
    r6 = r * 6 // 256
    g6 = g * 6 // 256
    b6 = b * 6 // 256
    return 16 + 36 * r6 + 6*g6 + b6

def curse_disp_rgb(stdscr, arr: np.ndarray, x=0, y=0):
    """
    Display a 28×28×3 RGB NumPy array in curses using the 256-color palette.

    Parameters
    ----------
    stdscr : curses.window
        The curses window object.
    arr : np.ndarray
        A 28×28×3 array of RGB data in [0..255].
    x : int
        Top row offset in the curses window.
    y : int
        Left column offset in the curses window.
    """
    # 1. Validate shape
    if arr.shape != (28, 28, 3):
        raise ValueError("Input array must be exactly 28×28×3 for RGB.")

    # 2. Prepare for color mapping
    pair_id_counter = 1
    max_pairs = curses.COLOR_PAIRS  # Total pairs supported by terminal

    # 3. Draw each pixel
    for row in range(28):
        for col in range(28):
            r, g, b = arr[row, col]


            color_idx = rgb_to_256_color_index(r, g, b)

            # If we haven't seen this color_idx, create a new color pair (foreground=black)
            if color_idx not in COLOR_MAP:
                if pair_id_counter < max_pairs:
                    print (color_idx)
                    curses.init_pair(pair_id_counter, curses.COLOR_BLACK, color_idx)
                    COLOR_MAP[color_idx] = pair_id_counter
                    pair_id_counter += 1
                else:
                    assert(False)

            pair_id = COLOR_MAP[color_idx]

            # Safely attempt to add the colored text
            try:
                # Multiply col by 2 to widen each cell to 2 spaces
                stdscr.addstr(x + row, (y + col) * 2, "@@", curses.color_pair(pair_id))
            except curses.error:
                raise Exception("Terminal too small to display image.")

# Initialize Pygame once in your program.
pygame.init()
_display_initialized = False
_screen = None

def init_screen(scale=10, grid_cols=4, grid_rows=4):
    """
    Initializes the global display. The screen size will be large enough
    to contain a grid of squares of size 28x28 scaled by `scale`.
    """
    global _display_initialized, _screen
    width = 30 * scale * grid_cols
    height = 30 * scale * grid_rows
    _screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Pygame 4x4 Grid")
    _display_initialized = True

def pygame_disp_rbg(rgb_array, pos_x, pos_y, scale=6):
    """
    Draws a 28x28 RGB image using Pygame at a given starting (x,y) position.
    The drawing is non-blocking and updates the display immediately.
    
    Parameters:
        rgb_array (np.ndarray): A NumPy array of shape (28, 28, 3) with dtype uint8.
        pos_x (int): The x-coordinate (in grid units, not pixels) where the square will be drawn.
        pos_y (int): The y-coordinate (in grid units, not pixels) where the square will be drawn.
        scale (int): Scaling factor to enlarge the 28x28 image for better visibility.
                     Default is 10.
                     
    The screen is assumed to be large enough to contain a 4x4 grid of squares.
    """
    global _display_initialized, _screen

    # Ensure the image is a 28x28x3 array.
    rgb_array = np.array(rgb_array, dtype=np.uint8)
    if rgb_array.shape != (28, 28, 3):
        raise ValueError("Input array must have shape (28, 28, 3)")

    # Initialize the screen if it hasn't been created yet.
    if not _display_initialized:
        # The grid is fixed at 4x4 squares.
        init_screen(scale=scale, grid_cols=4, grid_rows=4)

    # Create a surface from the image.
    # Note: pygame.surfarray.make_surface expects an array of shape (width, height, 3)
    # so we need to transpose our array from (height, width, 3).
    surface = pygame.surfarray.make_surface(np.transpose(rgb_array, (1, 0, 2)))
    # Scale the surface.
    window_square_size = (28 * scale, 28 * scale)
    surface = pygame.transform.scale(surface, window_square_size)

    # Convert grid coordinates (pos_x, pos_y) to pixel coordinates.
    pixel_x = pos_x*scale
    pixel_y = pos_y*scale

    # Blit the image onto the screen at the computed position.
    _screen.blit(surface, (pixel_x, pixel_y))
    pygame.display.update(pygame.Rect(pixel_x, pixel_y, window_square_size[0], window_square_size[1]))

    
if __name__ == '__main__':
    if sys.argv[1] == 't':
        train()
    
    elif sys.argv[1] == 'i':
        inference()
        
    elif sys.argv[1] == 'v':
        visualize()