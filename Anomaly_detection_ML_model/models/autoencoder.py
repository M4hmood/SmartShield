import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from opensearchpy import OpenSearch
from pyod.models.base_dl import BaseDeepLearningDetector
from pyod.utils.stat_models import pairwise_distances_no_broadcast
from pyod.utils.torch_utility import LinearBlock

from datetime import datetime
from dotenv import load_dotenv
import os

load_dotenv()  # This loads variables from .env into the environment

username = os.getenv('OPENSEARCH_USERNAME')
password = os.getenv('OPENSEARCH_PASSWORD')



# Define your dataset and model classes
class LargeDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class AutoEncoder(BaseDeepLearningDetector):
    def __init__(self, contamination=0.1, preprocessing=True, lr=1e-3, epoch_num=10, batch_size=32,
                 optimizer_name='adam', device=None, random_state=42, use_compile=False, compile_mode='default',
                 verbose=1, optimizer_params=None, hidden_neuron_list=None,
                 hidden_activation_name='relu', batch_norm=True, dropout_rate=0.2):
        if optimizer_params is None:
            optimizer_params = {'weight_decay': 1e-5}
        if hidden_neuron_list is None:
            hidden_neuron_list = [64, 32]
        super(AutoEncoder, self).__init__(contamination=contamination, preprocessing=preprocessing, lr=lr,
                                          epoch_num=epoch_num,
                                          batch_size=batch_size, optimizer_name=optimizer_name, criterion_name='mse',
                                          device=device, random_state=random_state, use_compile=use_compile,
                                          compile_mode=compile_mode, verbose=verbose, optimizer_params=optimizer_params)
        self.hidden_neuron_list = hidden_neuron_list
        self.hidden_activation_name = hidden_activation_name
        self.batch_norm = batch_norm
        self.dropout_rate = dropout_rate
        self.model = None
        self.optimizer = None
        self.criterion = None
        self.decision_scores_ = None
        self.feature_size = None  # Initialize feature_size to None

    def build_model(self):
        self.model = AutoEncoderModel(self.feature_size, hidden_neuron_list=self.hidden_neuron_list,
                                      hidden_activation_name=self.hidden_activation_name, batch_norm=self.batch_norm,
                                      dropout_rate=self.dropout_rate)

    def training_forward(self, batch_data):
        x = batch_data
        x = x.to(self.device)
        self.optimizer.zero_grad()
        x_recon = self.model(x)
        loss = self.criterion(x_recon, x)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def evaluating_forward(self, batch_data):
        x = batch_data
        x_gpu = x.to(self.device)
        x_recon = self.model(x_gpu)
        score = pairwise_distances_no_broadcast(x.numpy(), x_recon.cpu().numpy())
        return score

    def fit(self, X, y=None):
        X = self._preprocess_data(X)
        self._set_n_classes(y)
        self.device = self._get_device()

        # Set feature_size based on input data dimensions
        self.feature_size = X.shape[1]  # Assuming X is a 2D array with shape (num_samples, num_features)

        self.build_model()
        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, **self.optimizer_params)
        self.criterion = nn.MSELoss()
        dataset = LargeDataset(torch.from_numpy(X).float())
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True, num_workers=0)

        for epoch in range(self.epoch_num):
            self.model.train()
            train_loss = 0.0
            for batch in train_loader:
                batch_data = batch.to(self.device)
                loss = self.training_forward(batch_data)
                train_loss += loss
            if self.verbose:
                print(f'Epoch {epoch + 1}/{self.epoch_num}, Loss: {train_loss / len(train_loader)}')
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.from_numpy(X).float().to(self.device)
            reconstructions = self.model(X_tensor)
            self.decision_scores_ = pairwise_distances_no_broadcast(X, reconstructions.cpu().numpy())
        self._process_decision_scores()

    def decision_function(self, X):
        X = self._preprocess_data(X)
        X_tensor = torch.from_numpy(X).float().to(self.device)
        self.model.eval()
        with torch.no_grad():
            reconstructions = self.model(X_tensor)
            scores = pairwise_distances_no_broadcast(X, reconstructions.cpu().numpy())
        return scores

    def _preprocess_data(self, X):
        # Assuming you need to normalize or scale your data
        # Replace with actual preprocessing steps as needed
        return (X - X.mean(axis=0)) / X.std(axis=0)

    def _get_device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

class AutoEncoderModel(nn.Module):
    def __init__(self, feature_size, hidden_neuron_list=None, hidden_activation_name='relu', batch_norm=True,
                 dropout_rate=0.2):
        if hidden_neuron_list is None:
            hidden_neuron_list = [64, 32]
        super(AutoEncoderModel, self).__init__()
        self.feature_size = feature_size
        self.hidden_neuron_list = hidden_neuron_list
        self.hidden_activation_name = hidden_activation_name
        self.batch_norm = batch_norm
        self.dropout_rate = dropout_rate
        self.encoder = self._build_encoder()
        self.decoder = self._build_decoder()

    def _build_encoder(self):
        encoder_layers = []
        last_neuron_size = self.feature_size
        for neuron_size in self.hidden_neuron_list:
            encoder_layers.append(
                LinearBlock(last_neuron_size, neuron_size, activation_name=self.hidden_activation_name,
                            batch_norm=self.batch_norm, dropout_rate=self.dropout_rate))
            last_neuron_size = neuron_size
        return nn.Sequential(*encoder_layers)

    def _build_decoder(self):
        decoder_layers = []
        last_neuron_size = self.hidden_neuron_list[-1]
        for neuron_size in reversed(self.hidden_neuron_list[:-1]):
            decoder_layers.append(
                LinearBlock(last_neuron_size, neuron_size, activation_name=self.hidden_activation_name,
                            batch_norm=self.batch_norm, dropout_rate=self.dropout_rate))
            last_neuron_size = neuron_size
        decoder_layers.append(nn.Linear(last_neuron_size, self.feature_size))
        return nn.Sequential(*decoder_layers)

    def forward(self, x):
        return self.decoder(self.encoder(x))
    
# Connect to OpenSearch and retrieve data
client = OpenSearch(
    hosts=[{'host': 'localhost', 'port': 9200}],
    http_auth=(username, password),
    use_ssl=False,
    verify_certs=False,
)
query = {
    "query": 
    {
        "match_all": {}
    }
}
# Generate the index name with the current date
index_name = f"wazuh-alerts-{datetime.now().strftime('%Y.%m.7')}"

response = client.search(index=index_name, body=query)
data = [hit["_source"] for hit in response['hits']['hits']]
X = np.array([[value for value in record.values()] for record in data])
X = (X - X.mean(axis=0)) / X.std(axis=0)

# Initialize, train, and use the model to detect anomalies
model = AutoEncoder(contamination=0.1)
model.fit(X)
scores = model.decision_function(X)

# Store anomaly scores back in OpenSearch
for i, record in enumerate(data):
    anomaly_document = record.copy()
    anomaly_document["anomaly_score"] = float(scores[i])
    client.index(index="anomaly-index", body=anomaly_document)
