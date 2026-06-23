import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

class scDenoiseNet(nn.Module):
    def __init__(self, num_genes, latent_dim=32):
        super(scDenoiseNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(num_genes, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, latent_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_genes),
            nn.ReLU()
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

def train_and_impute(data_matrix, latent_dim=32, epochs=15, batch_size=64, lr=0.001):
    num_cells, num_genes = data_matrix.shape
    tensor_data = torch.tensor(data_matrix, dtype=torch.float32)
    dataset = TensorDataset(tensor_data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model = scDenoiseNet(num_genes=num_genes, latent_dim=latent_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    model.train()
    for epoch in range(epochs):
        for batch in dataloader:
            inputs = batch[0]
            optimizer.zero_grad()
            loss = criterion(model(inputs), inputs)
            loss.backward()
            optimizer.step()
            
    model.eval()
    with torch.no_grad():
        return model(tensor_data).numpy()
