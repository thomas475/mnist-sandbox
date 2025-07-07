import torch

class VAE(torch.nn.Module):
    def __init__(self, latent_dim=256, learning_rate=1e-3, epochs=10, verbose=1):
        super(VAE, self).__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(1, 8, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.AvgPool2d(2),
            torch.nn.Conv2d(8, 16, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.AvgPool2d(2)
        )
        self.fc_mu = torch.nn.Linear(16 * 7 * 7, latent_dim)
        self.fc_logvar = torch.nn.Linear(16 * 7 * 7, latent_dim)
        self.fc_decode = torch.nn.Linear(latent_dim, 16 * 7 * 7)
        self.decoder = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(16, 8, 2, stride=2),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(8, 1, 2, stride=2),
            torch.nn.Sigmoid()
        )
        self.loss_function = self.vae_loss
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.verbose = verbose

    def encode(self, x):
        x = self.encoder(x)
        x = torch.flatten(x, start_dim=1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        x = self.fc_decode(z)
        x = x.view(-1, 16, 7, 7)
        x = self.decoder(x)
        return x
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x = self.decode(z)
        return x, mu, logvar
    
    def vae_loss(self, X_reconstructed, X, mu, logvar):
        reconstruction_loss = torch.nn.functional.mse_loss(X_reconstructed, X, reduction='sum')
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return reconstruction_loss + kl_loss
    
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
                X_pred, mu, logvar = self(X_batch)
                loss = self.loss_function(X_pred, y_batch, mu, logvar)
                loss.backward()
                optimizer.step()
                cumulative_loss += loss.item()
            average_loss = cumulative_loss / len(dataloader)
            if self.verbose >= 1:
                print(f"\rEpoch {epoch + 1}/{self.epochs} - Loss: {average_loss:.4f}", end='', flush=True)
        
    def reconstruct(self, X):
        X_reconstructed, _, _ = self.forward(X)
        return X_reconstructed
