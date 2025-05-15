import os
import glob
import numpy as np
from plyfile import PlyData
from scipy.spatial.transform import Rotation
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn.functional as F
import shutil

# --- Gaussian loader with covariance extraction ---
def extract_covariance_for_vertex(vertex_data, idx):
    scales = np.array([
        vertex_data['scale_0'][idx],
        vertex_data['scale_1'][idx],
        vertex_data['scale_2'][idx]
    ], dtype=np.float32)
    quat = np.array([
        vertex_data['rot_0'][idx],
        vertex_data['rot_1'][idx],
        vertex_data['rot_2'][idx],
        vertex_data['rot_3'][idx]
    ], dtype=np.float32)
    rot = Rotation.from_quat(quat).as_matrix()
    Lambda = np.diag(scales)
    covariance = rot @ Lambda @ rot.T
    return np.array([
        covariance[0, 0], covariance[0, 1], covariance[0, 2],
        covariance[1, 1], covariance[1, 2], covariance[2, 2]
    ], dtype=np.float32)

def load_gaussians_from_ply(ply_path):
    plydata = PlyData.read(ply_path)
    vertex_data = plydata['vertex'].data
    N = len(vertex_data)
    means = np.zeros((N, 3), dtype=np.float32)
    covs = np.zeros((N, 6), dtype=np.float32)
    for i in range(N):
        means[i] = np.array([vertex_data['x'][i], vertex_data['y'][i], vertex_data['z'][i]], dtype=np.float32)
        covs[i] = extract_covariance_for_vertex(vertex_data, i)
    return means, covs

# --- Dataset with chunking ---
class GaussianChunkDataset(Dataset):
    def __init__(self, ply_folder, chunk_size=1000):
        self.file_paths = glob.glob(os.path.join(ply_folder, '*.ply'))
        self.chunk_size = chunk_size

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        means, covs = load_gaussians_from_ply(self.file_paths[idx])
        N = means.shape[0]
        chunks = []
        for start in range(0, N, self.chunk_size):
            end = min(start + self.chunk_size, N)
            chunk_means = means[start:end]
            chunk_covs = covs[start:end]
            features = np.concatenate([chunk_means, chunk_covs], axis=1)  # (chunk_len, 9)
            chunks.append(features)
        # Return list of chunks (arrays)
        return chunks

# --- Collate function to flatten batches ---
def collate_fn(batch):
    # batch is list of samples (each sample is list of chunks)
    # Flatten all chunks from all samples into one big batch tensor
    all_chunks = []
    for sample_chunks in batch:
        for chunk in sample_chunks:
            all_chunks.append(torch.tensor(chunk))
    return torch.cat(all_chunks, dim=0)  # (total_chunks_points, 9)

# --- Normalize dataset features ---
def compute_dataset_stats(ply_folder):
    # Compute mean and std per feature across all data for normalization
    file_paths = glob.glob(os.path.join(ply_folder, '*.ply'))
    all_features = []
    for p in file_paths:
        means, covs = load_gaussians_from_ply(p)
        features = np.concatenate([means, covs], axis=1)
        all_features.append(features)
    all_features = np.vstack(all_features)
    mean = np.mean(all_features, axis=0)
    std = np.std(all_features, axis=0) + 1e-6  # add small epsilon to avoid div zero
    return mean.astype(np.float32), std.astype(np.float32)

# --- VQ-VAE model ---

class Encoder(nn.Module):
    def __init__(self, input_dim=9, hidden_dim=256, latent_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)

class Decoder(nn.Module):
    def __init__(self, latent_dim=64, hidden_dim=256, output_dim=9):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, z):
        return self.net(z)

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings=512, embedding_dim=64, commitment_cost=0.25):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        # Init embeddings uniform between -1 and 1:
        self.embedding.weight.data.uniform_(-1, 1)
        self.commitment_cost = commitment_cost

    def forward(self, inputs):
        input_shape = inputs.shape
        flat_input = inputs.view(-1, self.embedding_dim)  # (B, D)

        distances = (
            torch.sum(flat_input**2, dim=1, keepdim=True)
            - 2 * torch.matmul(flat_input, self.embedding.weight.t())
            + torch.sum(self.embedding.weight**2, dim=1)
        )  # (B, num_embeddings)

        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)  # (B,1)
        encodings = torch.zeros(encoding_indices.size(0), self.num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        quantized = torch.matmul(encodings, self.embedding.weight)  # (B, D)
        quantized = quantized.view(input_shape)

        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        quantized = inputs + (quantized - inputs).detach()

        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return quantized, loss, perplexity

class VQVAE(nn.Module):
    def __init__(self, input_dim=9, hidden_dim=256, latent_dim=64, num_embeddings=512):
        super().__init__()
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim)
        self.vq = VectorQuantizer(num_embeddings, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim)

    def forward(self, x):
        z_e = self.encoder(x)
        z_q, vq_loss, perplexity = self.vq(z_e)
        x_recon = self.decoder(z_q)
        return x_recon, vq_loss, perplexity

# --- Training loop with checkpoint saving/loading ---
def train_vqvae(model, dataloader, device, mean, std, num_epochs=20, lr=1e-4, save_path='vqvae_checkpoint.pth'):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train()
    model.to(device)

    mse_loss = nn.MSELoss()

    for epoch in range(num_epochs):
        total_loss = 0
        total_recon_loss = 0
        total_vq_loss = 0
        total_perplexity = 0
        count = 0

        for batch in dataloader:
            batch = batch.to(device).float()
            # Normalize input batch:
            batch = (batch - mean.to(device)) / std.to(device)

            optimizer.zero_grad()

            x_recon, vq_loss, perplexity = model(batch)
            recon_loss = mse_loss(x_recon, batch)
            loss = recon_loss + 0.25 * vq_loss

            if torch.isnan(loss) or torch.isinf(loss):
                print("NaN or Inf in loss!")
                break

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_vq_loss += vq_loss.item()
            total_perplexity += perplexity.item()
            count += 1

            if count % 10 == 0:
                print(f"Epoch {epoch+1} Batch {count} | Loss: {loss.item():.4f} | Recon: {recon_loss.item():.4f} | VQ Loss: {vq_loss.item():.4f} | Perplexity: {perplexity.item():.4f}")

        print(f"Epoch {epoch+1}/{num_epochs} | Avg Loss: {total_loss/count:.4f} | Avg Recon: {total_recon_loss/count:.4f} | Avg VQ Loss: {total_vq_loss/count:.4f} | Avg Perplexity: {total_perplexity/count:.4f}")

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'mean': mean,
            'std': std,
        }, save_path)

def load_vqvae_checkpoint(model, optimizer, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    return model, optimizer, start_epoch

@torch.no_grad()
def extract_latent_codes(model, ply_path, chunk_size, mean, std, device):
    model.eval()
    means, covs = load_gaussians_from_ply(ply_path)
    N = means.shape[0]
    latent_codes = []

    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        chunk_means = means[start:end]
        chunk_covs = covs[start:end]
        chunk_features = np.concatenate([chunk_means, chunk_covs], axis=1)  # (chunk_len, 9)
        chunk_tensor = torch.tensor(chunk_features, dtype=torch.float32, device=device)

        chunk_tensor = (chunk_tensor - mean.to(device)) / std.to(device)
        z_e = model.encoder(chunk_tensor)
        z_q, _, _ = model.vq(z_e)
        latent_codes.append(z_q.cpu().numpy())

    latent_codes = np.concatenate(latent_codes, axis=0)
    return latent_codes


def main():
    ply_folder = "./splats"
    batch_size = 2
    chunk_size = 512
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Compute or load normalization stats
    stats_cache = "stats.npz"
    if os.path.exists(stats_cache):
        print(f"Removing existing stats cache at {stats_cache}")
        os.remove(stats_cache)

    print("Computing dataset stats...")
    mean_np, std_np = compute_dataset_stats(ply_folder)
    np.savez(stats_cache, mean=mean_np, std=std_np)
    mean = torch.tensor(mean_np, dtype=torch.float32)
    std = torch.tensor(std_np, dtype=torch.float32)

    # Prepare dataset and dataloader
    print("Preparing dataset")
    dataset = GaussianChunkDataset(ply_folder, chunk_size=chunk_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    # Initialize model and optimizer
    print("Initializing model")
    model = VQVAE(input_dim=9, hidden_dim=128, latent_dim=64, num_embeddings=512)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    model.to(device)

    # Try to load checkpoint if exists
    checkpoint_path = 'vqvae_checkpoint.pth'
    start_epoch = 0
    if os.path.exists(checkpoint_path):
        print(f"Removing existing checkpoint at {checkpoint_path}")
        os.remove(checkpoint_path)
        """
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        print(f"Resuming from epoch {start_epoch}")
        """

    # Train model (train_vqvae internally runs epochs from 0 to num_epochs)
    print("Training model")
    train_vqvae(
        model,
        dataloader,
        device,
        mean=mean.to(device),
        std=std.to(device),
        num_epochs=20,
        lr=1e-4,
        save_path=checkpoint_path
    )

    # Set model to eval after training
    model.eval()

    # Folder to save latent codes
    latent_folder = './latents'
    os.makedirs(latent_folder, exist_ok=True)

    # Extract and save latent codes for all ply files
    for ply_path in dataset.file_paths:
        latent_codes = extract_latent_codes(model, ply_path, chunk_size, mean=mean.to(device), std=std.to(device), device=device)
        filename = os.path.basename(ply_path).replace('.ply', '_latent.pt')
        save_path = os.path.join(latent_folder, filename)
        torch.save(latent_codes.cpu(), save_path)
        print(f"Saved latent codes for {ply_path} to {save_path}")

if __name__ == "__main__":
    main()
