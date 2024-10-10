import numpy as np

class MultilayerPerceptron:
    def __init__(self, layer_sizes, learning_rate=0.01, n_iterations=1000):
        # Initialize learning rate, iterations, and layer sizes.
        # Initialize weights and biases for each layer randomly (use small values).
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations

        # Initialize weights and biases
        self.weights = []
        self.biases = []
        for i in range(len(layer_sizes) - 1):
            weight = np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * 0.01
            bias = np.zeros((1, layer_sizes[i + 1]))
            self.weights.append(weight)
            self.biases.append(bias)

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
        else:
            raise ValueError("Unsupported activation function")

    def _activation_derivative(self, z, activation="sigmoid"):
        if activation == "sigmoid":
            sig = 1 / (1 + np.exp(-z))
            return sig * (1 - sig)
        elif activation == "relu":
            return np.where(z > 0, 1, 0)
        else:
            raise ValueError("Unsupported activation function")

    def _forward_propagation(self, X):
        activations = [X]
        input = X

        for i in range(len(self.weights)):
            z = np.dot(input, self.weights[i]) + self.biases[i]
            input = self._activation_function(z)
            activations.append(input)

        return activations

    def _backward_propagation(self, X, y, activations):
        m = y.shape[0]
        deltas = [None] * len(self.weights)
        
        # Compute the delta for the output layer
        deltas[-1] = activations[-1] - y.reshape(-1, 1)
        
        # Compute the deltas for the hidden layers
        for i in reversed(range(len(deltas) - 1)):
            deltas[i] = np.dot(deltas[i + 1], self.weights[i + 1].T) * self._activation_derivative(activations[i + 1])
        
        # Update the weights and biases
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * np.dot(activations[i].T, deltas[i]) / m
            self.biases[i] -= self.learning_rate * np.mean(deltas[i], axis=0, keepdims=True)

    def fit(self, X, y):
        for _ in range(self.n_iterations):
            activations = self._forward_propagation(X)
            self._backward_propagation(X, y, activations)

    def predict(self, X):
        activations = self._forward_propagation(X)
        return activations[-1]

    def _loss(self, y_true, y_pred):
        epsilon = 1e-15  # To avoid log(0)
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return loss
