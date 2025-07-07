import torch

class NeuralNetwork(torch.nn.Module):
    def __init__(self, learning_rate=1e-3, epochs=10, verbose=1):
        super(NeuralNetwork, self).__init__()
        self.linear1 = torch.nn.Linear(28 * 28, 64)
        self.activation = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(64, 10)
        self.softmax = torch.nn.Softmax()
        self.loss_function = torch.nn.CrossEntropyLoss()
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.verbose = verbose
    
    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        return x
    
    def fit(self, X, y):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate)
        dataloader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X, y), 
            batch_size=64, 
            shuffle=True
        )
        for epoch in range(self.epochs):
            cumulative_loss = 0.0
            for X_batch, y_batch in dataloader:
                optimizer.zero_grad()
                y_pred = self(X_batch)
                loss = self.loss_function(y_pred, y_batch)
                loss.backward()
                optimizer.step()
                cumulative_loss += loss.item()
            average_loss = cumulative_loss / len(dataloader)
            if self.verbose >= 1:
                print(f"\rEpoch {epoch + 1}/{self.epochs} - Loss: {average_loss:.4f}", end='', flush=True)
        
    def predict(self, X):
        return torch.argmax(self.forward(X), dim=1)
    