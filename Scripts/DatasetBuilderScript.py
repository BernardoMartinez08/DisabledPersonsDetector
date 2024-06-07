import os
import sys
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
import base64
import io


def encode_image(image_array):
    pil_img = Image.fromarray(image_array)
    buffer = io.BytesIO()
    pil_img.save(buffer, format="PNG")
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return img_str


def load_images(folder, label, image_size=(128, 128), batch_size=1000):
    print(f"\n\nLoading images from {folder}")
    images = []
    labels = []
    file_list = [f for f in os.listdir(folder) if f.endswith(".jpg") or f.endswith(".png")]

    for i in range(0, len(file_list), batch_size):
        print(f"Processing batch {i // batch_size + 1}/{len(file_list) // batch_size + 1}")
        print(f"Percentage of photos loaded: {i / len(file_list) * 100:.2f}%")
        batch_files = file_list[i:i + batch_size]
        batch_images = []
        for file in batch_files:
            filepath = os.path.join(folder, file)
            try:
                with Image.open(filepath) as img:
                    img = img.convert('L').resize(image_size)
                    batch_images.append(np.array(img))
            except (OSError, ValueError) as e:
                print(f"Skipping file {filepath} due to error: {e}")
        images.extend(batch_images)
        labels.extend([label] * len(batch_images))

    return images, labels


def create_dataset(images, labels):
    print("\nCreating dataset")
    data = pd.DataFrame()
    data["images"] = [encode_image(img) for img in images]
    data["labels"] = labels
    return data


def join_datasets(positive_data, negative_data):
    print("\nJoining datasets")
    data = pd.concat([positive_data, negative_data])
    data = data.sample(frac=1).reset_index(drop=True)
    return data


def split_dataset(data, train_size, validation_size, test_size):
    print("\nSplitting dataset")
    train_data, temp_data = train_test_split(data, train_size=train_size)
    validation_size_adjusted = validation_size / (validation_size + test_size)
    validation_data, test_data = train_test_split(temp_data, test_size=1 - validation_size_adjusted)
    return train_data, validation_data, test_data


def save_dataset(data, filename):
    print("\nSaving dataset to", filename)
    data.to_csv(filename, index=False)


def main(positive_folder, negative_folder, output_folder):
    # Load images
    positive_images, positive_labels = load_images(positive_folder, 1, image_size=(128, 128))
    negative_images, negative_labels = load_images(negative_folder, 0, image_size=(128, 128))

    # Create datasets
    positive_data = create_dataset(positive_images, positive_labels)
    negative_data = create_dataset(negative_images, negative_labels)

    # Join datasets
    joined_data = join_datasets(positive_data, negative_data)

    # Define paths
    os.makedirs(output_folder, exist_ok=True)
    data_path = os.path.join(output_folder, 'data.csv')
    train_path = os.path.join(output_folder, 'train_data.csv')
    validation_path = os.path.join(output_folder, 'validation_data.csv')
    test_path = os.path.join(output_folder, 'test_data.csv')

    # Save datasets
    save_dataset(joined_data, data_path)

    # Split dataset
    train_data, validation_data, test_data = split_dataset(joined_data, 0.7, 0.15, 0.15)

    # Save datasets
    save_dataset(train_data, train_path)
    save_dataset(validation_data, validation_path)
    save_dataset(test_data, test_path)

    print("\nTraining set size:", len(train_data))
    print("Validation set size:", len(validation_data))
    print("Test set size:", len(test_data))


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python DatasetBuilderScript.py <positive_folder> <negative_folder> <output_folder>")
        sys.exit(1)

    positive_folder = sys.argv[1]
    negative_folder = sys.argv[2]
    output_folder = sys.argv[3]
    main(positive_folder, negative_folder, output_folder)
