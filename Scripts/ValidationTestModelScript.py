import os
import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import load_model


def load_preprocessed_data(data_file):
    print("Loading preprocessed data from", data_file)
    data = np.load(data_file)
    train_data, train_labels = data['train_data'], data['train_labels']
    val_data, val_labels = data['val_data'], data['val_labels']
    test_data, test_labels = data['test_data'], data['test_labels']
    return train_data, train_labels, val_data, val_labels, test_data, test_labels


def plot_history(history):
    # Plot training & validation accuracy values
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    # Plot training & validation loss values
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    plt.tight_layout()
    plt.show()


def evaluate_model(model, test_data, test_labels):
    print("Evaluating model on test data")
    test_loss, test_accuracy = model.evaluate(test_data, test_labels)
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test Loss: {test_loss:.4f}")

    predictions = (model.predict(test_data) > 0.5).astype("int32")
    print("Classification Report")
    print(classification_report(test_labels, predictions))
    print("Confusion Matrix")
    print(confusion_matrix(test_labels, predictions))


def main(data_file, model_path):
    # Load data
    train_data, train_labels, val_data, val_labels, test_data, test_labels = load_preprocessed_data(data_file)

    # Define paths
    model_file = os.path.join(model_path, 'disables_persons_detector_model.h5')
    history_file = os.path.join(model_path, 'history.npy')

    # Load model
    if os.path.exists(model_file):
        model = load_model(model_file)
    else:
        print("Model file not found. Exiting.")
        sys.exit(1)

    # Evaluate model
    evaluate_model(model, test_data, test_labels)

    # Plot training history
    if os.path.exists(history_file):
        history = np.load(history_file, allow_pickle=True).item()
        plot_history(history)
    else:
        print("History file not found. Skipping plotting.")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python test.py <data_file> <model_file>")
        sys.exit(1)

    data_file = sys.argv[1]
    model_path = sys.argv[2]
    main(data_file, model_path)
