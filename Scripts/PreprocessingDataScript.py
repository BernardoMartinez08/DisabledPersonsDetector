import os
import sys
import pandas as pd
from PIL import Image
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import base64
import io


def decode_image(image_str):
    image_data = base64.b64decode(image_str)
    image = Image.open(io.BytesIO(image_data))
    return np.array(image)


def load_data(data_file, image_size_x, image_size_y):
    print("\n\nLoading data from", data_file)
    data = pd.read_csv(data_file)
    images = []
    labels = data['labels'].tolist()
    print(f"Converting images to Gray scale and resizing to {image_size_x}x{image_size_y}")
    for img_str in data['images']:
        print(f"Processing image {len(images) + 1}/{len(data)}", end='\r')
        try:
            img = decode_image(img_str)
            img = Image.fromarray(img).convert('L')
            img = img.resize((image_size_x, image_size_y))
            img = np.array(img)
            images.append(img)
        except (OSError, ValueError) as e:
            print(f"Skipping image due to error: {e}")
    return np.array(images), np.array(labels)


def preprocess_data(images, labels, augmentation_factor):
    print("\nPreprocessing data")
    print("\nNormalizing images")
    images = images.astype('float32') / 255
    print("\nAdding channel dimension")
    images = np.expand_dims(images, axis=-1)

    print("\nApplying data augmentation")
    datagen = ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.25,
        height_shift_range=0.25,
        zoom_range=[0.5, 1.5],
        shear_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    augmented_images = []
    augmented_labels = []
    for i in range(len(images)):
        print(f"Augmenting image {i + 1}/{len(images)}", end='\r')
        img = images[i].reshape((1,) + images[i].shape)  # Reshape to (1, 128, 128, 1)
        label = labels[i]
        aug_iter = datagen.flow(img, batch_size=1)
        for _ in range(augmentation_factor):  # Generate augmented images per original image
            aug_img = next(aug_iter)[0]
            augmented_images.append(aug_img)
            augmented_labels.append(label)

    augmented_images = np.array(augmented_images)
    augmented_labels = np.array(augmented_labels)
    return augmented_images, augmented_labels


def save_preprocessed_data(train_data, train_labels, val_data, val_labels, test_data, test_labels, output_file):
    print("\nSaving preprocessed data to", output_file)
    np.savez(output_file,
             train_data=train_data, train_labels=train_labels,
             val_data=val_data, val_labels=val_labels,
             test_data=test_data, test_labels=test_labels)


def main(input_dir, output_dir):
    # Define paths
    os.makedirs(input_dir, exist_ok=True)
    train_path = os.path.join(input_dir, 'train_data.csv')
    validation_path = os.path.join(input_dir, 'validation_data.csv')
    test_path = os.path.join(input_dir, 'test_data.csv')

    os.makedirs(output_dir, exist_ok=True)
    preprocessed_data_path = os.path.join(output_dir, 'preprocessed_data.npz')

    # Load data
    train_images, train_labels = load_data(train_path, 128, 128)
    val_images, val_labels = load_data(validation_path, 128, 128)
    test_images, test_labels = load_data(test_path, 128, 128)

    # Preprocess data
    train_images, train_labels = preprocess_data(train_images, train_labels, 5)
    val_images, val_labels = preprocess_data(val_images, val_labels, 5)
    test_images, test_labels = preprocess_data(test_images, test_labels, 5)

    # Save preprocessed data
    save_preprocessed_data(train_images, train_labels, val_images, val_labels, test_images, test_labels,
                           preprocessed_data_path)
    print("Preprocessing complete. Data saved to", output_dir)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python preprocess_images.py <input_dir> <output_dir>")
        sys.exit(1)

    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    main(input_dir, output_dir)
