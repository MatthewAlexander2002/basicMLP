import numpy as np

class Perceptron:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        """
        Initialize the Perceptron with learning rate and the number of iterations.
        
        Parameters:
        learning_rate (float): The step size for weight updates.
        n_iterations (int): The number of times to iterate over the training data.
        """
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None

    def _unit_step(self, x):
        """
        Unit step function to determine the output label based on the score.

        Parameters:
        x (float): The linear combination of input features and weights.

        Returns:
        int: 1 if the score is greater than or equal to 0, -1 otherwise.
        """
        return np.where(x >= 0, 1, -1)

    def fit(self, X, y):
        """
        Train the Perceptron model.

        Parameters:
        X (numpy.ndarray): Training data, shape (n_samples, n_features).
        y (numpy.ndarray): Target values, shape (n_samples,).
        """
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Training loop for the perceptron
        for _ in range(self.n_iterations):
            for idx, x_i in enumerate(X):
                # Calculate linear combination (score)
                linear_output = np.dot(x_i, self.weights) + self.bias
                predicted = self._unit_step(linear_output)

                # Weight update rule: w = w + learning_rate * (target - predicted) * x_i
                update = self.learning_rate * (y[idx] - predicted)
                self.weights += update * x_i
                self.bias += update

    def predict(self, X):
        """
        Make predictions on the input data.

        Parameters:
        X (numpy.ndarray): Input data, shape (n_samples, n_features).

        Returns:
        numpy.ndarray: Predicted class labels (1 or -1).
        """
        linear_output = np.dot(X, self.weights) + self.bias
        return self._unit_step(linear_output)

# Example usage:
if __name__ == "__main__":
    # Generate some dummy data (replace with real data)
    X = np.array([[1, 1], [2, 1], [2, 2], [1, 0], [0, 1], [0, 0]])  # Features
    y = np.array([1, 1, 1, -1, -1, -1])  # Binary class labels

    # Initialize and train the Perceptron
    perceptron = Perceptron(learning_rate=0.1, n_iterations=10)
    perceptron.fit(X, y)

    # Test the Perceptron model
    predictions = perceptron.predict(X)
    print("Predictions:", predictions)