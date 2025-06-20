!pip install torch torchvision matplotlib seaborn scikit-learn numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np
import seaborn as sns

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load MNIST
transform = transforms.ToTensor()
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# Autoencoder
class Autoencoder(nn.Module):
    # Corrected constructor name from _init_ to __init__
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Flatten(), nn.Linear(28*28, 64), nn.ReLU())
        self.decoder = nn.Sequential(nn.Linear(64, 28*28), nn.Sigmoid(), nn.Unflatten(1, (1, 28, 28)))

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat

# Variational Autoencoder
class VAE(nn.Module):
    # Corrected constructor name from _init_ to __init__
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc_mu = nn.Linear(128, 32)
        self.fc_logvar = nn.Linear(128, 32)
        self.fc2 = nn.Linear(32, 128)
        self.fc3 = nn.Linear(128, 28*28)

    def encode(self, x):
        h = F.relu(self.fc1(x))
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = F.relu(self.fc2(z))
        return torch.sigmoid(self.fc3(h))

    def forward(self, x):
        x = x.view(-1, 28*28)
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z).view(-1, 1, 28, 28), mu, logvar

# Loss Functions
def vae_loss(recon, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

# Train Function
def train(model, is_vae=False, epochs=10):
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    losses = []

    for epoch in range(epochs):
        epoch_loss = 0
        for data, _ in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            if is_vae:
                output, mu, logvar = model(data)
                loss = vae_loss(output, data, mu, logvar)
            else:
                output = model(data)
                loss = F.mse_loss(output, data)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(train_loader.dataset)
        losses.append(avg_loss)
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")
    return losses

# Run AE and VAE
ae = Autoencoder()
vae = VAE()

loss_ae = train(ae, is_vae=False)
loss_vae = train(vae, is_vae=True)

#  Plot loss curves
plt.plot(loss_ae, label='AE')
plt.plot(loss_vae, label='VAE')
plt.title("Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()

#  Visualize latent space
def extract_latents(model, is_vae=False):
    model.eval()
    latents, labels = [], []
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            if is_vae:
                x = data.view(-1, 28*28)
                mu, _ = model.encode(x)
                latents.append(mu.cpu())
            else:
                z = model.encoder(data)
                latents.append(z.cpu())
            labels.extend(target.numpy())
    return torch.cat(latents).numpy(), np.array(labels)

z_ae, y_ae = extract_latents(ae)
z_vae, y_vae = extract_latents(vae, is_vae=True)

#  PCA/t-SNE Visualization
def plot_latent(z, y, title):
    z_pca = PCA(n_components=2).fit_transform(z)
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    z_tsne = tsne.fit_transform(z)

    plt.figure(figsize=(12,5))
    plt.subplot(1, 2, 1)
    sns.scatterplot(x=z_pca[:,0], y=z_pca[:,1], hue=y, palette="tab10", s=10, legend=False)
    plt.title(title + " (PCA)")

    plt.subplot(1, 2, 2)
    sns.scatterplot(x=z_tsne[:,0], y=z_tsne[:,1], hue=y, palette="tab10", s=10, legend=False)
    plt.title(title + " (t-SNE)")
    plt.show()

plot_latent(z_ae, y_ae, "AE Latent Space")
plot_latent(z_vae, y_vae, "VAE Latent Space")

#  Reconstruction comparison
def compare_reconstructions(model_ae, model_vae):
    model_ae.eval()
    model_vae.eval()
    data, _ = next(iter(test_loader))
    data = data.to(device)
    with torch.no_grad():
        recon_ae = model_ae(data)
        recon_vae, _, _ = model_vae(data)

    fig, axes = plt.subplots(3, 10, figsize=(15, 4))
    for i in range(10):
        axes[0, i].imshow(data[i].cpu().squeeze(), cmap="gray")
        axes[1, i].imshow(recon_ae[i].cpu().squeeze(), cmap="gray")
        axes[2, i].imshow(recon_vae[i].cpu().squeeze(), cmap="gray")
        for ax in axes[:, i]:
            ax.axis('off')
    axes[0, 0].set_ylabel("Original")
    axes[1, 0].set_ylabel("AE")
    axes[2, 0].set_ylabel("VAE")
    plt.suptitle("Reconstruction Comparison")
    plt.show()

compare_reconstructions(ae, vae)

#  Latent vector arithmetic
def latent_arithmetic(model, idx1, idx2, alpha=0.5):
    model.eval()
    data, _ = next(iter(test_loader))
    data = data.to(device)
    with torch.no_grad():
        # The VAE model requires a flattened input for encoding
        mu1, _ = model.encode(data[idx1].view(-1, 28*28))
        mu2, _ = model.encode(data[idx2].view(-1, 28*28))
        interp = alpha * mu1 + (1 - alpha) * mu2
        recon = model.decode(interp).view(1, 1, 28, 28)
        return data[idx1].cpu(), data[idx2].cpu(), recon.cpu()

img1, img2, img_interp = latent_arithmetic(vae, 0, 1)
fig, ax = plt.subplots(1, 3, figsize=(9, 3))
for i, img in enumerate([img1, img_interp, img2]):
    ax[i].imshow(img.squeeze(), cmap="gray")
    ax[i].axis('off')
ax[0].set_title("Image A")
ax[1].set_title("Interpolation")
ax[2].set_title("Image B")
plt.show()
