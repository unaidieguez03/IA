import torch
import torch.nn as nn
import torch.optim as optim
from .custom_dataloader import LazyFrameDataset
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm
class ConvolutionalAutoencoder():
    def __init__(self, autoencoder, training_set:LazyFrameDataset,batch_size:int, device):
        self.autoencoder = autoencoder
        self.device = device
        self.optimizer = optim.AdamW(self.autoencoder.parameters(), lr=1e-3)
        self.criterion = nn.MSELoss()
        self.dataloader = DataLoader(
            dataset=training_set,
            batch_size=batch_size,
            shuffle=True
        )
    def _train_epoch(self):
        self.autoencoder.train()
        total_loss = 0.0

        for batch in tqdm(self.dataloader, desc="Training process"):
            batch = batch.to(self.device)
                
            self.optimizer.zero_grad()

            reconstructed = self.autoencoder(batch)
                
            loss = self.criterion(reconstructed, batch)
            loss.backward()
            self.optimizer.step()
                
            total_loss += loss.item()
        avg_loss = total_loss / len(training_set)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')
        return avg_loss

    def train(self, num_epochs: int):

        for epoch in range(num_epochs):
            avr_loss = self._train_epoch()
            print(avr_loss)
        
        return self.autoencoder, self.autoencoder.encoder
    def autoencode(self, x):
        return self.autoencoder(x)
    
    def encode(self, x):
        encoder = self.autoencoder.encoder
        return encoder(x)

    def decode(self, x):
        decoder = self.autoencoder.decoder
        return decoder(x)