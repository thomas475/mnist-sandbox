import torch

class GAN(torch.nn.Module):
    def __init__(self, latent_dim=128, learning_rate=1e-3, epochs=10, verbose=1):
        super(GAN, self).__init__()
        self.generator = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, 64 * 7 * 7),
            torch.nn.ReLU(),
            torch.nn.Unflatten(1, (64, 7, 7)),
            torch.nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(32, 1, kernel_size=2, stride=2)
        )
        self.discriminator = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.AvgPool2d(2, 2),
            torch.nn.Conv2d(32, 64, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.AvgPool2d(2, 2),
            torch.nn.Flatten(1),
            torch.nn.Linear(64 * 7 * 7, 1),
            torch.nn.Sigmoid()
        )
        self.loss_function = torch.nn.BCEWithLogitsLoss()
        self.latent_dim = latent_dim
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.verbose = verbose

    def forward(self, z, discriminator=False):
        out = self.generator(z)
        if discriminator:
            score = self.discriminator(out)
            return out, score
        return out
    
    def fit(self, X):
        optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=self.learning_rate)
        optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=self.learning_rate)
        dataloader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X),
            batch_size=64, 
            shuffle=True
        )
        for epoch in range(self.epochs):
            cumulative_g_loss = 0.0
            cumulative_d_loss = 0.0
            for (real_batch,) in dataloader:
                batch_size = real_batch.size(0)
                real_labels = torch.ones(batch_size, 1)
                fake_labels = torch.zeros(batch_size, 1)

                z = torch.randn(batch_size, self.latent_dim)
                fake_batch = self.generator(z)

                # train discriminator
                real_preds = self.discriminator(real_batch)
                fake_preds = self.discriminator(fake_batch.detach())
                d_loss = self.loss_function(real_preds, real_labels) + self.loss_function(fake_preds, fake_labels)
                optimizer_D.zero_grad()
                d_loss.backward()
                optimizer_D.step()

                # train generator
                fake_preds = self.discriminator(fake_batch)
                g_loss = self.loss_function(fake_preds, real_labels)
                optimizer_G.zero_grad()
                g_loss.backward()
                optimizer_G.step()

                cumulative_g_loss += g_loss.item()
                cumulative_d_loss += d_loss.item()
            average_g_loss = cumulative_g_loss / len(dataloader)
            average_d_loss = cumulative_d_loss / len(dataloader)
            if self.verbose >= 1:
                print(f"\rEpoch {epoch + 1}/{self.epochs} - G Loss: {average_g_loss:.4f} - D Loss: {average_d_loss:.4f}", end='', flush=True)
                