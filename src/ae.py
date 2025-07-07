import torch

class Autoencoder(torch.nn.Module):
    def __init__(self, learning_rate=1e-3, epochs=10, verbose=1):
        super(Autoencoder, self).__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(1, 8, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.AvgPool2d(2),
            torch.nn.Conv2d(8, 16, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.AvgPool2d(2)
        )
        self.bottleneck = torch.nn.Linear(16 * 7 * 7, 16 * 7 * 7)
        self.decoder = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(16, 8, 2, stride=2),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(8, 1, 2, stride=2),
            torch.nn.Sigmoid()
        )
        self.loss_function = torch.nn.MSELoss()
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.verbose = verbose

    def encode(self, x):
        x = self.encoder(x)
        x = torch.flatten(x, start_dim=1)
        return x
    
    def decode(self, x):
        x = x.view(-1, 16, 7, 7)
        x = self.decoder(x)
        return x
    
    def forward(self, x):
        x = self.encode(x)
        x = self.bottleneck(x)
        x = self.decode(x)
        return x
    
    def fit(self, X):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        dataloader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X, X),
            batch_size=64, 
            shuffle=True
        )
        for epoch in range(self.epochs):
            cumulative_loss = 0.0
            for X_batch, y_batch in dataloader:
                optimizer.zero_grad()
                X_pred = self(X_batch)
                loss = self.loss_function(X_pred, y_batch)
                loss.backward()
                optimizer.step()
                cumulative_loss += loss.item()
            average_loss = cumulative_loss / len(dataloader)
            if self.verbose >= 1:
                print(f"\rEpoch {epoch + 1}/{self.epochs} - Loss: {average_loss:.4f}", end='', flush=True)
        
    def reconstruct(self, X):
        return self.forward(X)