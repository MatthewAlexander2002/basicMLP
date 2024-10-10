import numpy as np

class Layer:
    def __init__(self, input_size, output_size, activation="relu"):
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.randn(input_size, output_size) * 0.01
        self.biases = np.zeros((1, output_size))
        self.activation = activation

class MultilayerPerceptron:
    # def __init__(self, layer_sizes, learning_rate=0.01, n_iterations=1000):
    #     # Initialize learning rate, iterations, and layer sizes.
    #     # Initialize weights and biases for each layer randomly (use small values).
    #     self.layer_sizes = layer_sizes
    #     self.learning_rate = learning_rate
    #     self.n_iterations = n_iterations

    #     # Initialize weights and biases
    #     self.weights = []
    #     self.biases = []
    #     for i in range(len(layer_sizes) - 1):
    #         weight = np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * 0.01
    #         bias = np.zeros((1, layer_sizes[i + 1]))
    #         self.weights.append(weight)
    #         self.biases.append(bias)
    def __init__(self, layers, learning_rate=0.01, n_iterations=1000):
        self.layers = layers
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.training_loss_ = []  # Initialize an empty list to store training loss

    def _initialize_weights(self, layer_sizes):
        self.weights = []
        self.biases = []
        for i in range(len(layer_sizes) - 1):
            weight = np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * 0.01
            bias = np.zeros((1, layer_sizes[i + 1]))
            self.weights.append(weight)
            self.biases.append(bias)
            
    def _activation_function(self, z, activation="sigmoid"):
        if activation == "sigmoid":
            return 1 / (1 + np.exp(-z))
        elif activation == "relu":
            return np.maximum(0, z)
        elif activation == "softmax":
            exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))  # For numerical stability
            return exp_z / np.sum(exp_z, axis=1, keepdims=True)
        else:
            raise ValueError("Unsupported activation function")

    def _activation_derivative(self, a, activation):
        if activation == "sigmoid":
            return a * (1 - a)
        elif activation == "relu":
            return np.where(a > 0, 1, 0)
        else:
            raise ValueError("Unsupported activation function")

    def _forward_propagation(self, X):
        activations = [X]
        for layer in self.layers:
            z = np.dot(activations[-1], layer.weights) + layer.biases
            a = self._activation_function(z, activation=layer.activation)
            activations.append(a)
        return activations

    def _backward_propagation(self, X, y, activations):
        # Initialize deltas
        deltas = [None] * len(self.layers)
        
        # Compute the delta for the output layer
        y_pred = activations[-1]
        if y.ndim == 1:
            y = np.eye(y_pred.shape[1])[y]
        deltas[-1] = y_pred - y
        
        # Compute the deltas for the hidden layers
        for i in reversed(range(len(deltas) - 1)):
            deltas[i] = np.dot(deltas[i + 1], self.layers[i + 1].weights.T) * self._activation_derivative(activations[i + 1], self.layers[i].activation)
        
        # Update the weights and biases
        m = X.shape[0]
        for i in range(len(self.layers)):
            self.layers[i].weights -= self.learning_rate * np.dot(activations[i].T, deltas[i]) / m
            self.layers[i].biases -= self.learning_rate * np.sum(deltas[i], axis=0, keepdims=True) / m
            
    def fit(self, X, y):
        self.training_loss_ = []  # Initialize an empty list to store training loss
        for _ in range(self.n_iterations):
            activations = self._forward_propagation(X)
            self._backward_propagation(X, y, activations)
            
            # Compute and store the loss
            y_pred = activations[-1]
            loss = self._loss(y, y_pred)
            self.training_loss_.append(loss)

    def predict(self, X):
        activations = self._forward_propagation(X)
        return activations[-1]

    def _loss(self, y_true, y_pred):
        if y_true.ndim == 1:
            y_true = np.eye(y_pred.shape[1])[y_true]
        
        epsilon = 1e-15  # To avoid log(0)
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return loss
