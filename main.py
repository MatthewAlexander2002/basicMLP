from perceptron import Perceptron

def main():
    # Example training data for AND logic gate
    training_inputs = [
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ]
    labels = [0, 0, 0, 1]

    perceptron = Perceptron(input_size=2)
    perceptron.train(training_inputs, labels)

    # Test the perceptron
    inputs = [1, 1]
    print(f"Prediction for input {inputs}: {perceptron.predict(inputs)}")

if __name__ == "__main__":
    main()