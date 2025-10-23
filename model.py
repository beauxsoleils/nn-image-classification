
from torch.nn import Module, Linear, ReLU, Sequential, Flatten

class Model(Module):
    """
    Class Model defines a forward-feed neural network with an input layer, three hidden layers and a single output layer.
    Here we use it to perform image classification on the 10-digit MNIST dataset.
    """

    def __init__(self):
        super().__init__()
        self.net = Sequential(
            Flatten(),
            Linear(784, 128), # input layer (784 features) to hidden layer (128 neurons)
            ReLU(), # activation function: rectified linear unit
            Linear(128, 64), # hidden layer
            ReLU(), # activation function
            Linear(64, 10) # hidden layer to output layer (10 classes for digits 0-9)
        )

    def forward(self, x):
        return self.net(x)

if __name__ == '__main__': pass



