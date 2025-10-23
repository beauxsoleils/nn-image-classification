# Image Classification Using a Foward-Feed Neural Network
This repository serves as a base for my learning journey throughout deep learning and artificial intelligence. 
I designed the codebase to be as readable and modular as possible for quick modification during learning projects.
It defines a multilayer perceptron with three hidden layers. We use classic backpropagation on MNIST features for training the model's weights.

The only dependencies are pytorch and tqdm.

for now, you only need to run
```sh
python run.py
```


with the following optional arguments:

```sh
[--help] [--epochs EPOCHS] [--learning_rate LEARNING_RATE] [--save SAVE]
```


## TODO:
- eval.py -- for confirming and testing our model's accuracy and displaying results
- Build a CNN for more elaborate image classification tasks using the Fashion-MNIST dataset and eventually SVHN.

 
