import os
import sys

from PIL import Image
import pandas as pd
from sklearn.model_selection import train_test_split


def load_images(folder, label):
    print("\n\nLoading images from", folder)
    images = []
    labels = []
    for file in os.listdir(folder):
        filepath = os.path.join(folder, file)
        if file.endswith(".jpg") or file.endswith(".png"):
            with Image.open(filepath) as img:
                images.append(img.copy())
            labels.append(label)
    return images, labels


def create_dataset(images, labels):
    print("\nCreating dataset")
    data = pd.DataFrame()
    data["images"] = images
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
    positive_images, positive_labels = load_images(positive_folder, 1)
    negative_images, negative_labels = load_images(negative_folder, 0)

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

    print("Training set size:", len(train_data))
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
