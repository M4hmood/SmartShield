import json
import numpy as np
import os
import torch
from PIL import Image
from torchvision import transforms


def load_data(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)

    # Numeric data
    numeric_data = np.array(data['numeric_data']).reshape(-1, 1)

    # String data: Example preprocessing - convert strings to their lengths
    string_data = np.array([len(s) for s in data['string_data']]).reshape(-1, 1)

    # Mixed data scores
    mixed_data_scores = np.array([item['scores'] for item in data['mixed_data']]).reshape(-1, len(
        data['mixed_data'][0]['scores']))

    # Image data: Example preprocessing - convert images to flattened arrays
    image_data = []
    for img_path in data.get('image_data', []):
        if os.path.exists(img_path):
            image = Image.open(img_path).convert('L')  # Convert to grayscale
            image = image.resize((28, 28))  # Resize to 28x28
            image = np.array(image).flatten()  # Flatten the image
            image_data.append(image)
    image_data = np.array(image_data) if image_data else np.zeros((0, 28 * 28))

    # URL data: Example preprocessing - convert URLs to lengths
    url_data = np.array([len(url) for url in data['url_data']]).reshape(-1, 1)

    # Find the maximum number of rows among all data arrays
    max_rows = max(len(numeric_data), len(string_data), len(mixed_data_scores), len(image_data), len(url_data))

    # Function to pad arrays to the maximum number of rows
    def pad_array(arr, max_rows):
        if arr.shape[0] < max_rows:
            pad_width = ((0, max_rows - arr.shape[0]), (0, 0))
            arr = np.pad(arr, pad_width, mode='constant', constant_values=0)
        return arr

    # Pad all arrays to have the same number of rows
    numeric_data = pad_array(numeric_data, max_rows)
    string_data = pad_array(string_data, max_rows)
    mixed_data_scores = pad_array(mixed_data_scores, max_rows)
    image_data = pad_array(image_data, max_rows)
    url_data = pad_array(url_data, max_rows)

    # Combine all data
    combined_data = np.hstack([
        numeric_data,
        string_data,
        mixed_data_scores,
        image_data,
        url_data
    ])

    return combined_data


# Main script
if __name__ == "__main__":
    # Load data
    data_path = os.path.join('data', 'data.json')
    X_train = load_data(data_path)

    # Initialize and train the model
    from models.autoencoder import AutoEncoder

    clf = AutoEncoder(contamination=0.05, epoch_num=50, batch_size=64)
    clf.fit(X_train)

    # Save the model
    model_save_path = os.path.join('models', 'autoencoder.pth')
    torch.save(clf.model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")
