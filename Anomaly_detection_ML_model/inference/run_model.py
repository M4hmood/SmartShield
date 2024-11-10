import os
import torch
import numpy as np
from models.autoencoder import AutoEncoderModel
from utils.data_loader import load_data

# Load data
data_path = r"C:\Users\bouma\desktop\SmartShield\Anomaly_detection_ML_model\data\data.json"

X_test = load_data(data_path)

# Load the trained model
model_path = r"C:\Users\bouma\desktop\SmartShield\Anomaly_detection_ML_model\models\autoencoder.pth"
feature_size = X_test.shape[1]
model = AutoEncoderModel(feature_size)
model.load_state_dict(torch.load(model_path))
model.eval()

# Predict anomalies
X_test_tensor = torch.from_numpy(X_test).float()
with torch.no_grad():
    reconstructions = model(X_test_tensor)
scores = np.mean((X_test_tensor.numpy() - reconstructions.numpy()) ** 2, axis=1)

# Assuming contamination is 0.05 as in training
threshold = np.percentile(scores, 95)
predictions = (scores > threshold).astype(int)

print("Anomaly scores:\n", scores)
print("Predictions:\n", predictions)
print("Threshold:\n", threshold)
