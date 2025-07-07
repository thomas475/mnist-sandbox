import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from torchvision import datasets, transforms

def load_mnist():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    return train_dataset, test_dataset

def display_number(image, label=None, path=None):
    plt.imshow(image.detach().squeeze(), cmap='gray')
    if label:
        plt.title(f"Label: {label}")
    plt.axis('off')
    _store(path)
    plt.show()
    
def display_numbers(images, label=None, path=None):
    n = len(images)
    _, axes = plt.subplots(1, n, figsize=(n * 2, 2))

    if n == 1:
        axes = [axes]

    for i, sample in enumerate(images):
        img = sample.detach().squeeze().cpu().numpy()
        axes[i].imshow(img, cmap='gray')
        axes[i].axis('off')

    if label:
        plt.suptitle(f"Label: {label}")

    _store(path)
    plt.show()

def display_numbers_grid(image_dict, path=None):
    labels = sorted(image_dict.keys())
    num_labels = len(labels)
    num_samples = len(next(iter(image_dict.values())))

    _, axes = plt.subplots(num_labels, num_samples, figsize=(num_samples * 2, num_labels * 2))
    if num_labels == 1:
        axes = [axes]

    for row_idx, label in enumerate(labels):
        samples = image_dict[label]
        for col_idx, image in enumerate(samples):
            img = image.squeeze().cpu().numpy()
            ax = axes[row_idx][col_idx] if num_labels > 1 else axes[col_idx]
            ax.imshow(img, cmap='gray')
            ax.axis('off')

    plt.tight_layout()
    _store(path)
    plt.show()

def display_reconstruction(original, reconstruction, label, path=None):
    plt.figure(figsize=(6, 3))

    plt.subplot(1, 2, 1)
    plt.imshow(original.squeeze(), cmap='gray')
    plt.title(f"Original {label}")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(reconstruction.squeeze().detach().cpu(), cmap='gray')
    plt.title(f"Reconstructed {label}")
    plt.axis('off')

    _store(path)
    plt.show()

def display_reconstructions_grid(pairs, path=None):
    num_rows = len(pairs)

    _, axes = plt.subplots(num_rows, 2, figsize=(4, 2 * num_rows))
    if num_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    for row, (orig, recon) in enumerate(pairs):
        axes[row][0].imshow(orig.detach().squeeze().cpu().numpy(), cmap='gray')
        axes[row][0].axis('off')
        axes[row][1].imshow(recon.detach().squeeze().cpu().numpy(), cmap='gray')
        axes[row][1].axis('off')

    axes[0][0].set_title("Original", fontsize=12)
    axes[0][1].set_title("Reconstruction", fontsize=12)

    plt.tight_layout()
    _store(path)
    plt.show()

def generate_samples(model, n_samples, label=None, variance_scaling=1.0, mode="cvae"):
    samples = []
    
    if mode == "cvae":
        for _ in range(n_samples):
            with torch.no_grad():
                z = torch.randn(1, model.fc_mu.out_features) * variance_scaling
                sample = model.decode(z, torch.tensor([label], dtype=torch.long))
                samples.append(sample.squeeze(0))
    elif mode == "gan":
        z = torch.randn(n_samples, 128) * variance_scaling
        samples = model(z)

    return samples

def explore_latent_space(model, X, original, grid_size=5, span=3.0, mode="ae", path=None):
    device = original.device
    latent_variables = []

    with torch.no_grad():
        for x in X:
            x = x.to(device)
            x = x if x.ndim == 4 else x.unsqueeze(0)
            if mode == "ae":
                z = model.bottleneck(model.encode(x))
            elif mode == "vae":
                z, _ = model.encode(x)
            latent_variables.append(z.squeeze().cpu().numpy())

    latent_matrix = np.stack(latent_variables)
    pca = PCA(n_components=2).fit(latent_matrix)
    pc1, pc2 = pca.components_

    with torch.no_grad():
        x = original.to(device)
        x = x if x.ndim == 4 else x.unsqueeze(0)
        if mode == "ae":
            z_orig = model.bottleneck(model.encode(x))
        elif mode == "vae":
            z_orig, _ = model.encode(x)
        z_orig = z_orig.squeeze().cpu().numpy()

    lin = np.linspace(-span, span, grid_size)
    decoded_images = []
    for dy in lin:
        row = []
        for dx in lin:
            z = z_orig + dx * pc1 + dy * pc2
            z_tensor = torch.tensor(z).unsqueeze(0).float().to(device)
            image = model.decode(z_tensor).squeeze().cpu()
            row.append(image)
        decoded_images.append(row)

    _, axs = plt.subplots(grid_size, grid_size, figsize=(grid_size, grid_size))
    for i in range(grid_size):
        for j in range(grid_size):
            image = decoded_images[i][j]
            if image.ndim == 3:
                image = image.permute(1, 2, 0)
            axs[i, j].imshow(image.detach().numpy(), cmap='gray' if image.ndim == 2 else None)
            axs[i, j].axis('off')

    center = grid_size // 2
    axs[center, center].set_title("Original", fontsize=8, color='red')
    plt.tight_layout()
    _store(path)
    plt.show()

def _store(path):
    if path is not None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.savefig(path, bbox_inches='tight', pad_inches=0.1)
