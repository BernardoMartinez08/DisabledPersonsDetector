import os
import sys
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model


def load_and_preprocess_image(img_path, image_size=(128, 128)):
    try:
        with Image.open(img_path) as img:
            img = img.convert('L')
            img = img.resize(image_size)
            img = np.array(img).astype('float32') / 255.0
            img = np.expand_dims(img, axis=-1)  # Add channel dimension
            return img
    except (OSError, ValueError) as e:
        print(f"Error loading image {img_path}: {e}")
        return None


def predict_images(model, folder, image_size=(128, 128)):
    print(f"Predicting images in folder: {folder}")
    file_list = [f for f in os.listdir(folder) if f.endswith(".jpg") or f.endswith(".png")]
    images = []
    file_paths = []
    for file in file_list:
        img_path = os.path.join(folder, file)
        img = load_and_preprocess_image(img_path, image_size)
        if img is not None:
            images.append(img)
            file_paths.append(img_path)
    images = np.array(images)

    if len(images) > 0:
        predictions = (model.predict(images) > 0.5).astype("int32")
        for img_path, prediction in zip(file_paths, predictions):
            label = "Disabled" if prediction == 1 else "Non-Disabled"
            print(f"{img_path}: {label}")
    else:
        print("No valid images found in the folder.")


def main(model_path, image_folder):
    # Load model
    if os.path.exists(model_path):
        model = load_model(model_path)
    else:
        print("Model file not found. Exiting.")
        sys.exit(1)

    # Predict images
    predict_images(model, image_folder)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python predict.py <model_path> <image_folder>")
        sys.exit(1)

    model_path = sys.argv[1]
    image_folder = sys.argv[2]
    main(model_path, image_folder)
