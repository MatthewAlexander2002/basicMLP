from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from custom_scaler import CustomScaler 
from multilayer_perceptron import Layer, MultilayerPerceptron  
import numpy as np

if __name__ == "__main__":
    # Load the iris dataset
    iris = load_iris()
    X, y = iris.data, iris.target
    print(f"Features shape: {X.shape}, Labels shape: {y.shape}")

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Training set shape: {X_train.shape}, Testing set shape: {X_test.shape}")

    # Standardize the features
    scaler = CustomScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    print(f"First 5 rows of scaled training data:\n{X_train[:5]}")

    # Define the layers for the MLP
    layers = [
        Layer(input_size=4, output_size=5, activation="relu"),
        Layer(input_size=5, output_size=3, activation="softmax")  # Output layer with softmax
    ]

    # Initialize the MLP with the defined layers
    mlp = MultilayerPerceptron(layers)
    print(f"MLP structure: {[{'input_size': layer.input_size, 'output_size': layer.output_size, 'activation': layer.activation} for layer in layers]}")

    # Train the model using the `fit` function
    mlp.fit(X_train, y_train)
    print(f"First 5 training losses: {mlp.training_loss_[:5]}")

    # Use the `predict` function to classify new data
    predictions = mlp.predict(X_test)
    print(f"Predictions: {predictions}")
    
    # Convert probabilities to class labels
    class_labels = np.argmax(predictions, axis=1)
    
    for i, (prob, pred_label, true_label) in enumerate(zip(predictions, class_labels, y_test)):
        print(f"Test instance {i+1}:")
        print(f"  Predicted probabilities: {prob}")
        print(f"  Predicted label: {pred_label}, True label: {true_label}\n")

    # Print the predicted class labels and the true labels
    print(f"Predicted labels: {class_labels}")
    print(f"True labels:      {y_test}")

    # Calculate and print the accuracy
    accuracy = np.mean(class_labels == y_test)
    print(f"Accuracy: {accuracy * 100:.2f}%")