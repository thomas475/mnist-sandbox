import torch

    
class BetaCVAE(torch.nn.Module):
    def __init__(self, num_classes=10, beta=1, latent_dim=256, learning_rate=1e-3, epochs=10, verbose=1):
        super(BetaCVAE, self).__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(1, 8, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.AvgPool2d(2),
            torch.nn.Conv2d(8, 16, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.AvgPool2d(2)
        )
        self.fc_mu = torch.nn.Linear(16 * 7 * 7 + num_classes, latent_dim)
        self.fc_logvar = torch.nn.Linear(16 * 7 * 7 + num_classes, latent_dim)
        self.fc_decode = torch.nn.Linear(latent_dim + num_classes, 16 * 7 * 7)
        self.decoder = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(16, 8, 2, stride=2),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(8, 1, 2, stride=2),
            torch.nn.Sigmoid()
        )
        self.loss_function = self.vae_loss
        self.num_classes = num_classes
        self.beta = beta
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.verbose = verbose

    def encode(self, x, y):
        x = self.encoder(x)
        x = torch.flatten(x, start_dim=1)
        y_onehot = torch.nn.functional.one_hot(y, self.num_classes).float()
        x = torch.cat([x, y_onehot], dim=1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z, y):
        y_onehot = torch.nn.functional.one_hot(y, self.num_classes).float()
        z = torch.cat([z, y_onehot], dim=1)
        x = self.fc_decode(z)
        x = x.view(-1, 16, 7, 7)
        x = self.decoder(x)
        return x
    
    def forward(self, x, y):
        mu, logvar = self.encode(x, y)
        z = self.reparameterize(mu, logvar)
        x = self.decode(z, y)
        return x, mu, logvar
    
    def vae_loss(self, X_reconstructed, X, mu, logvar):
        reconstruction_loss = torch.nn.functional.mse_loss(X_reconstructed, X, reduction='sum')
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return reconstruction_loss + self.beta * kl_loss
    
    def fit(self, X, y):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        dataloader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X, y),
            batch_size=64, 
            shuffle=True
        )
        for epoch in range(self.epochs):
            cumulative_loss = 0.0
            for X_batch, y_batch in dataloader:
                optimizer.zero_grad()
                X_pred, mu, logvar = self.forward(X_batch, y_batch)
                loss = self.loss_function(X_pred, X_batch, mu, logvar)
                loss.backward()
                optimizer.step()
                cumulative_loss += loss.item()
            average_loss = cumulative_loss / len(dataloader)
            if self.verbose >= 1:
                print(f"\rEpoch {epoch + 1}/{self.epochs} - Loss: {average_loss:.4f}", end='', flush=True)
        
    def reconstruct(self, X, y):
        X_reconstructed, _, _ = self.forward(X, y)
        return X_reconstructed  
    

class CVAE(BetaCVAE):
    def __init__(self, num_classes=10, latent_dim=256, learning_rate=1e-3, epochs=10, verbose=1):
        super(CVAE, self).__init__(beta=1.0, num_classes=num_classes, latent_dim=latent_dim, learning_rate=learning_rate, epochs=epochs, verbose=verbose)
