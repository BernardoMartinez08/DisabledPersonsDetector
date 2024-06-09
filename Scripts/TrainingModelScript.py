import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Input
from tensorflow.keras.callbacks import EarlyStopping


def load_preprocessed_data(data_file):
    print("Loading preprocessed data from", data_file)
    data = np.load(data_file)
    train_data, train_labels = data['train_data'], data['train_labels']
    val_data, val_labels = data['val_data'], data['val_labels']
    return train_data, train_labels, val_data, val_labels


def build_model(input_shape):
    print("Building model")
    model = Sequential([
        Input(shape=input_shape),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Dropout(0.5),
        Flatten(),
        Dense(100, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def train_model(model, train_data, train_labels, val_data, val_labels, batch_size, epochs):
    print("Starting training")
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    history = model.fit(
        train_data, train_labels,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(val_data, val_labels),
        callbacks=[early_stopping]
    )
    return history


def save_model(model, model_path):
    print("Saving model to", model_path)
    model.save(model_path)


def save_history(history, history_path):
    print("Saving history to", history_path)
    np.save(history_path, history.history)


def main(data_file, model_save_path):
    # Load preprocessed data
    train_data, train_labels, val_data, val_labels = load_preprocessed_data(data_file)

    # Build and train model
    model = build_model(train_data.shape[1:])
    history = train_model(model, train_data, train_labels, val_data, val_labels, 32, 60)

    # Save model and history
    os.makedirs(model_save_path, exist_ok=True)
    model_path = os.path.join(model_save_path, 'disabled_persons_detector_model.keras')
    history_path = os.path.join(model_save_path, 'history.npy')

    save_model(model, model_path)
    save_history(history, history_path)
    print("Model trained and saved at", model_path)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python train.py <data_file> <model_save_path>")
        sys.exit(1)

    data_file = sys.argv[1]
    model_save_path = sys.argv[2]
    main(data_file, model_save_path)
