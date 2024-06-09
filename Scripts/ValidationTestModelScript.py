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


def print_preprocessed_data_info(train_data, val_data, test_data):
    print("\nPreprocessed data loaded")
    print("Data shapes")
    print("Train data shape:", train_data.shape)
    print("Validation data shape:", val_data.shape)
    print("Test data shape:", test_data.shape)

    print("\nData sizes")
    print("Train data size:", len(train_data))
    print("Validation data size:", len(val_data))
    print("Test data size:", len(test_data))


def plot_history(history, plot_file):
    # Plot training & validation accuracy values
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history['accuracy'])
    plt.plot(history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    # Plot training & validation loss values
    plt.subplot(1, 2, 2)
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    plt.tight_layout()
    plt.savefig(plot_file)
    plt.show()


def plot_confusion_matrix(cm, plot_file, classes, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], fmt), ha='center', va='center',
                 color='white' if cm[i, j] > thresh else 'black')

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(plot_file)
    plt.show()


def evaluate_model(model, test_data, test_labels, cm_plot_file):
    print("Evaluating model on test data")
    test_loss, test_accuracy = model.evaluate(test_data, test_labels)
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test Loss: {test_loss:.4f}")

    predictions = (model.predict(test_data) > 0.5).astype("int32")
    print("Classification Report")
    print(classification_report(test_labels, predictions))

    cm = confusion_matrix(test_labels, predictions)
    print("Confusion Matrix")
    print(cm)

    plot_confusion_matrix(cm, cm_plot_file, classes=['Non-Disabled', 'Disabled'])


def main(data_file, model_path, plots_path):
    # Load data
    train_data, train_labels, val_data, val_labels, test_data, test_labels = load_preprocessed_data(data_file)
    print_preprocessed_data_info(train_data, val_data, test_data)

    # Define paths
    model_file = os.path.join(model_path, 'disabled_persons_detector_model.keras')
    history_file = os.path.join(model_path, 'history.npy')
    history_plot_file = os.path.join(plots_path, 'history_plot.png')
    cm_plot_file = os.path.join(plots_path, 'confusion_matrix.png')


    # Load model
    if os.path.exists(model_path):
        model = load_model(model_file)
    else:
        print("Model file not found. Exiting.")
        sys.exit(1)

    # Evaluate model
    evaluate_model(model, test_data, test_labels, cm_plot_file)

    # Load and plot training history
    if os.path.exists(history_file):
        history = np.load(history_file, allow_pickle=True).item()
        plot_history(history, history_plot_file)
    else:
        print("History file not found. Skipping plotting.")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python test.py <data_file> <model_path> <plots_path>")
        sys.exit(1)

    data_file = sys.argv[1]
    model_path = sys.argv[2]
    plots_path = sys.argv[3]
    main(data_file, model_path, plots_path)
