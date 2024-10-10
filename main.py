from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from custom_scaler import CustomScaler 
from multilayer_perceptron import MultilayerPerceptron  

if __name__ == "__main__":
    # Load the iris dataset
    iris = load_iris()
    X, y = iris.data, iris.target

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize the features
    scaler = CustomScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Initialize the MLP with a list of layer sizes (e.g., input layer, hidden layers, output layer)
    mlp = MultilayerPerceptron(layer_sizes=[4, 5, 3], learning_rate=0.01, n_iterations=1000)

    # Train the model using the `fit` function
    mlp.fit(X_train, y_train)

    # Use the `predict` function to classify new data
    predictions = mlp.predict(X_test)

    # Print the predictions
    print(predictions)