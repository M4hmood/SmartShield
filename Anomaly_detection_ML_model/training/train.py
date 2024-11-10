import os
import torch
from models.autoencoder import AutoEncoder
from utils.data_loader import load_data

if __name__ == "__main__":
    # Load data
    data_path = os.path.join('data', 'data.json')
    X_train = load_data(data_path)

    # Initialize and train the model
    clf = AutoEncoder(contamination=0.05, epoch_num=50, batch_size=64)
    clf.fit(X_train)

    # Save the model
    model_save_path = os.path.join('models', 'autoencoder.pth')
    torch.save(clf.model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")
