# Goals
- Follow
     - [But what is a neuron network? | Chapter 1, Deep learning](https://www.youtube.com/watch?v=aircAruvnKk&t=147s)
     - [Gradient descent, how neural networks learn | Chapter 2, Deep learning](https://www.youtube.com/watch?v=IHZwWFHWa-w)
     - [What is backpropagation really doing? | Chapter 3, Deep learning](https://www.youtube.com/watch?v=Ilg3gGewQ5U)
- Copilot was used to help with completion
- Don't reference any code
- Only 1 dep PIL - for parsing/displaying image

# Notes
- Download mnist dataset from https://www.kaggle.com/datasets/hojjatk/mnist-dataset?resource=download
- Figureout the dataset format
- Write a dataset loader.
- A neuron takes in multiple inputs and produce a single output - activation
- Each neuron from one layer is connected to all neurons from a previous layer
- output = sigmoid(w*a+b)
    - `a` activation from previous layer
    - `w` weights of the connection between neurons
    - `b` is the bias of this neuron

- The sigmoid function is only there to normalize the output
- Sigmoid(0) == 0.5 so if we initialize the weights and biases to 0, the output of the neuron will be 0.5
- Our model will have 4 layers 784 - 16 - 16 - 10
- The first layer is the input layer 784 == 28*28
- Last layer is the output layer for 10 digits.
- Neuron from one layer will connect to all neuron from its previous layer so we will have:
    - 784*16 + 16*16 + 16*10 = 12960 weights
    - 16 + 16 + 10 = 42 biases

    ![Model](model.png)

- How to feed forward 

    ![Feed forward](feed_forward.png)

- The number of weights between 2 layers of size and `m` and `n` will `m*n`
- The number of biases of layer of size `m` wil be m
- The activations of a layer will be the input of the next layer

- Back propagation

    ![Back propagation](back_prop.png)


# Pit Falls
- Input need to be normalized 
- Remember to reset the `activations` in feed forward
~~~python
# This is RIGHT
self.biases = [random.random() for i in range(self.dim)]

# This is WRONG
self.biases = [random()]*self.dim
~~~
- Becareful with the sign of `gradient`