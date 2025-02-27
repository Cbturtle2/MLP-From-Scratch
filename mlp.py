import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from typing import Tuple


def batch_generator(train_x, train_y, batch_size):
    """
    Generator that yields batches of train_x and train_y.

    :param train_x (np.ndarray): Input features of shape (n, f).
    :param train_y (np.ndarray): Target values of shape (n, q).
    :param batch_size (int): The size of each batch.

    :return tuple: (batch_x, batch_y) where batch_x has shape (B, f) and batch_y has shape (B, q). The last batch may be smaller.
    """
    sample_count = train_x.shape[0]
    indices = np.arange(sample_count)
    np.random.shuffle(indices)

    for start_idx in range(0, sample_count, batch_size):
        end_idx = min(start_idx + batch_size, sample_count)
        batch_indices = indices[start_idx:end_idx]
        yield train_x[batch_indices], train_y[batch_indices]


class ActivationFunction(ABC):
    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the output of the activation function, evaluated on x

        Input args may differ in the case of softmax

        :param x (np.ndarray): input
        :return: output of the activation function
        """
        pass

    @abstractmethod
    def derivative(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the derivative of the activation function, evaluated on x
        :param x (np.ndarray): input
        :return: activation function's derivative at x
        """
        pass


class Sigmoid(ActivationFunction):
    def forward(self, x):
        return 1 / (1 + np.exp(-x))

    def derivative(self, x):
        return self.forward(x) * (1 - self.forward(x))


class Tanh(ActivationFunction):
    def forward(self, x):
        return np.tanh(x)

    def derivative(self, x):
        return 1 - np.square(np.tanh(x))


class Relu(ActivationFunction):
    def forward(self, x):
        return np.maximum(0, x)

    def derivative(self, x):
        return np.where(x > 0, 1, 0)


class Softmax(ActivationFunction):
    def forward(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def derivative(self, x):
        return self.forward(x) * (1 - self.forward(x))


class Linear(ActivationFunction):
    def forward(self, x):
        return x

    def derivative(self, x):
        return np.ones_like(x)
        # This just returns 1s in the same shape as x since nothing is changing.


class LossFunction(ABC):
    @abstractmethod
    def loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def derivative(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        pass


class SquaredError(LossFunction):
    def loss(self, y_true, y_pred):
        return 0.5 * np.mean(np.square(y_true - y_pred), axis=0)

    def derivative(self, y_true, y_pred):
        return (y_pred - y_true) / y_true.shape[0]
        # This is the percentage of error in the predictions.


class CrossEntropy(LossFunction):
    def loss(self, y_true, y_pred):
        return -np.mean(np.where(y_true == 0, 0, y_true * np.log(y_pred)), axis=0)

    def derivative(self, y_true, y_pred):
        return (y_pred - y_true) / y_true.shape[0]


class Layer:
    def __init__(self, fan_in: int, fan_out: int, activation_function: ActivationFunction):
        """
        Initializes a layer of neurons

        :param fan_in: number of neurons in previous (presynpatic) layer
        :param fan_out: number of neurons in this layer
        :param activation_function: instance of an ActivationFunction
        """
        self.fan_in = fan_in
        self.fan_out = fan_out
        self.activation_function = activation_function

        # this will store the activations (forward prop)
        self.activations = None
        # this will store the delta term (dL_dPhi, backward prop)
        self.delta = None

        limit = np.sqrt(6 / (fan_in + fan_out))
        self.W = np.random.uniform(-limit, limit, (fan_in, fan_out))
        self.b = np.zeros((1, fan_out))

    def forward(self, h: np.ndarray):
        """
        Computes the activations for this layer

        :param h: input to layer
        :return: layer activations
        """
        self.z = np.dot(h, self.W) + self.b

        self.activations = self.activation_function.forward(self.z)

        return self.activations

    def backward(self, h: np.ndarray, delta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply backagation to this layer and return the weight and bias gradients

        :param h: input to this layer
        :param delta: delta term from layer above
        :return: (weight gradients, bias gradients)
        """
        # Replace all NoneType with with 0s
        if self.delta is None:
            self.delta = np.zeros_like(self.activations)

        dL_dW = np.dot(h.T, self.delta)
        dL_db = np.sum(self.delta, axis=0, keepdims=True)
        self.delta = None
        return dL_dW, dL_db


class MultilayerPerceptron:
    def __init__(self, layers: Tuple[Layer]):
        """
        Create a multilayer perceptron (densely connected multilayer neural network)
        :param layers: list or Tuple of layers
        """
        self.layers = layers

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        This takes the network input and computes the network output (forward propagation)
        :param x: network input
        :return: network output
        """

        h = x

        # Process through each layer
        for layer in self.layers:
            h = layer.forward(h)

        return h

    def backward(self, loss_grad: np.ndarray, input_data: np.ndarray) -> Tuple[list, list]:
        """
        Applies backpropagation to compute the gradients of the weights and biases for all layers in the network
        :param loss_grad: gradient of the loss function
        :param input_data: network's input data
        :return: (List of weight gradients for all layers, List of bias gradients for all layers)
        """
        dl_dw_all = []
        dl_db_all = []

        # Start with the loss gradient for the output layer
        delta = loss_grad

        # Store layer inputs for backward pass
        h = input_data
        h_values = [h]

        # Get inputs for each layer without modifying the activations
        for i in range(len(self.layers) - 1):
            h = self.layers[i].activations
            h_values.append(h)

        # Backward pass
        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            h = h_values[i]

            # Compute gradients for this layer
            dL_dw, dL_db = layer.backward(h, delta)

            # Store gradients
            dl_dw_all.insert(0, dL_dw)
            dl_db_all.insert(0, dL_db)

        return dl_dw_all, dl_db_all

    def train(self, train_x: np.ndarray, train_y: np.ndarray, val_x: np.ndarray, val_y: np.ndarray, loss_func: LossFunction, learning_rate: float = 1E-3, batch_size: int = 16, epochs: int = 32) -> Tuple[np.ndarray, np.ndarray]:
        """
        Train the multilayer perceptron

        :param train_x: full training set input of shape (n x d) n = number of samples, d = number of features
        :param train_y: full training set output of shape (n x q) n = number of samples, q = number of outputs per sample
        :param val_x: full validation set input
        :param val_y: full validation set output
        :param loss_func: instance of a LossFunction
        :param learning_rate: learning rate for parameter updates
        :param batch_size: size of each batch
        :param epochs: number of epochs
        :return:
        """
        training_losses = []
        validation_losses = []

        for epoch in range(epochs):
            # Reset loss since we are only interested in the loss per epoch
            training_loss = 0
            for input, target in batch_generator(train_x, train_y, batch_size):
                # Forward pass
                output = self.forward(input)

                # Compute loss
                batch_loss = loss_func.loss(target, output)
                training_loss += np.sum(batch_loss)

                # Backward pass
                loss_grad = loss_func.derivative(target, output)
                dl_dw_all, dl_db_all = self.backward(loss_grad, input)

                # Update weights and biases
                for i in range(len(self.layers)):
                    layer = self.layers[i]
                    layer.W -= learning_rate * dl_dw_all[i]
                    layer.b -= learning_rate * dl_db_all[i]
            training_loss /= train_x.shape[0]
            training_losses.append(training_loss)

            # Validation loss
            val_output = self.forward(val_x)
            validation_loss = np.mean(loss_func.loss(val_y, val_output))
            validation_losses.append(validation_loss)
            print(
                f"Epoch {epoch + 1}/{epochs}, Training Loss: {training_loss}, Validation Loss: {validation_loss}")

        return training_losses, validation_losses
