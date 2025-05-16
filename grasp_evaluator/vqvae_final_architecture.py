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
from tqdm import tqdm
import matplotlib.pyplot as plt

# ----------- Gaussian Loader & Dataset -----------

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

    if not np.isfinite(quat).all():
        raise ValueError(f"Invalid quaternion data: {quat}")

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
    means = np.empty((N, 3), dtype=np.float32)
    covs = np.empty((N, 6), dtype=np.float32)

    for i in range(N):
        means[i] = np.array([vertex_data['x'][i], vertex_data['y'][i], vertex_data['z'][i]], dtype=np.float32)
        covs[i] = extract_covariance_for_vertex(vertex_data, i)

    return means, covs


class GaussianChunkDataset(Dataset):
    def __init__(self, ply_path, chunk_size=512, mean=None, std=None):
        if mean is None or std is None:
            raise ValueError("Mean and std must be provided for normalization.")
        
        self.chunk_size = chunk_size
        self.mean = mean
        self.std = std

        means, covs = load_gaussians_from_ply(ply_path)
        features = np.concatenate([means, covs], axis=1)
        features = (features - self.mean) / self.std
        N = features.shape[0]
        self.chunks = []

        for start_idx in range(0, N, self.chunk_size):
            end_idx = min(start_idx + self.chunk_size, N)
            chunk = features[start_idx:end_idx]

            if chunk.shape[0] < self.chunk_size:
                pad_size = self.chunk_size - chunk.shape[0]
                padding = np.full((pad_size, features.shape[1]), fill_value=np.mean(features, axis=0), dtype=np.float32)
                chunk = np.vstack([chunk, padding])

            self.chunks.append(chunk)

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        return torch.tensor(self.chunks[idx], dtype=torch.float32)


# ----------- Model Architecture -----------

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, z):
        z = F.relu(self.fc1(z))
        z = F.relu(self.fc2(z))
        return self.fc3(z)


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VectorQuantizer, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost

        self.embeddings = nn.Embedding(num_embeddings, embedding_dim)
        self.embeddings.weight.data.uniform_(-1 / num_embeddings, 1 / num_embeddings)

    def forward(self, z):
        z_flattened = z.view(-1, self.embedding_dim)
        distances = (torch.sum(z_flattened**2, dim=1, keepdim=True) 
                     + torch.sum(self.embeddings.weight**2, dim=1)
                     - 2 * torch.matmul(z_flattened, self.embeddings.weight.t()))

        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        z_q = self.embeddings(encoding_indices).view(z.shape)

        loss = F.mse_loss(z_q.detach(), z) + self.commitment_cost * F.mse_loss(z_q, z.detach())

        z_q = z + (z_q - z).detach()
        return z_q, loss, encoding_indices


class VQVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, num_embeddings, commitment_cost):
        super(VQVAE, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim)
        self.quantizer = VectorQuantizer(num_embeddings, latent_dim, commitment_cost)
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim)

    def forward(self, x):
        z = self.encoder(x)
        z_q, vq_loss, encoding_indices = self.quantizer(z)
        x_recon = self.decoder(z_q)
        recon_loss = F.mse_loss(x_recon, x)
        total_loss = recon_loss + vq_loss
        return x_recon, total_loss, vq_loss, z_q


# ----------- Visualization -----------

def visualize_reconstruction(original, reconstructed, num_samples=5):
    for i in range(num_samples):
        plt.figure(figsize=(8, 4))
        plt.subplot(1, 2, 1)
        plt.title("Original Gaussian")
        plt.plot(original[i])
        plt.subplot(1, 2, 2)
        plt.title("Reconstructed Gaussian")
        plt.plot(reconstructed[i])
        plt.show()


# ----------- Main Training Loop -----------

def compute_dataset_mean_std(ply_folder):
    all_features = []
    file_paths = glob.glob(os.path.join(ply_folder, '*.ply'))
    for ply_path in tqdm(file_paths, desc="Computing mean/std over all data"):
        means, covs = load_gaussians_from_ply(ply_path)
        features = np.concatenate([means, covs], axis=1)
        all_features.append(features)
    all_features = np.vstack(all_features)
    mean = np.mean(all_features, axis=0)
    std = np.std(all_features, axis=0) + 1e-9
    return mean, std

def get_num_workers(max_workers=8):
    try:
        cpu_count = os.cpu_count() or 1
        return min(cpu_count, max_workers)
    except Exception:
        return 1  # fallback

def train_vqvae(
    ply_folder,
    chunk_size=512,
    input_dim=9,
    hidden_dim=256,
    latent_dim=256,
    num_embeddings=512,
    commitment_cost=0.25,
    batch_size=64,
    num_epochs=30,
    lr=1e-3,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    checkpoint_dir='checkpoints',
    latent_save_dir='latent_per_object'
):
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(latent_save_dir, exist_ok=True)

    print(f"Using device: {device}")

    # Compute dataset mean and std for normalization
    mean, std = compute_dataset_mean_std(ply_folder)

    model = VQVAE(input_dim, hidden_dim, latent_dim, num_embeddings, commitment_cost).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    ply_paths = glob.glob(os.path.join(ply_folder, '*.ply'))

    # Training loop
    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_loss = 0.0
        total_batches = 0

        for ply_path in tqdm(ply_paths, desc=f"Epoch {epoch} Training"):
            dataset = GaussianChunkDataset(ply_path, chunk_size, mean, std)
            num_workers = get_num_workers(max_workers=8)

            dataloader = DataLoader(
                dataset, 
                batch_size=batch_size, 
                shuffle=True, 
                drop_last=True, 
                num_workers=num_workers, 
                pin_memory=True
            )


            for batch in dataloader:
                batch = batch.to(device)
                optimizer.zero_grad()
                x_recon, total_loss, vq_loss, _ = model(batch)
                total_loss.backward()
                optimizer.step()

                epoch_loss += total_loss.item()
                total_batches += 1

        avg_loss = epoch_loss / max(total_batches, 1)
        print(f"Epoch {epoch}/{num_epochs} - Avg Loss: {avg_loss:.6f}")

        # Save checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, f"vqvae_epoch_{epoch}.pt")
        torch.save(model.state_dict(), checkpoint_path)

    # After training, extract and save latent vectors per object
    model.eval()
    with torch.no_grad():
        for ply_path in tqdm(ply_paths, desc="Extracting Latents per Object"):
            dataset = GaussianChunkDataset(ply_path, chunk_size, mean, std)
            dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

            all_latents = []
            for chunk in dataloader:
                chunk = chunk.to(device)
                _, _, _, z_q = model(chunk)
                # z_q shape: (batch_size=1, chunk_size, latent_dim)
                # Remove batch dim:
                z_q = z_q.squeeze(0)
                all_latents.append(z_q.cpu().numpy())

            all_latents = np.vstack(all_latents)  # (num_points, latent_dim)
            latent_vector = np.mean(all_latents, axis=0)  # (latent_dim,)

            # Save latent vector
            base_name = os.path.basename(ply_path).replace('.ply', '.npy')
            save_path = os.path.join(latent_save_dir, base_name)
            np.save(save_path, latent_vector)

    # Visualize some reconstruction samples from the last object
    dataset = GaussianChunkDataset(ply_paths[-1], chunk_size, mean, std)
    dataloader = DataLoader(dataset, batch_size=5, shuffle=False)
    originals = []
    reconstructions = []
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            x_recon, _, _, _ = model(batch)
            originals.append(batch.cpu().numpy())
            reconstructions.append(x_recon.cpu().numpy())
    originals = np.vstack(originals)
    reconstructions = np.vstack(reconstructions)
    visualize_reconstruction(originals, reconstructions, num_samples=5)


if __name__ == "__main__":
    # Set your dataset path here
    ply_folder = './splats'

    train_vqvae(
        ply_folder=ply_folder,
        chunk_size=512,
        input_dim=9,  # 3 means + 6 covariances
        hidden_dim=256,
        latent_dim=256,
        num_embeddings=512,
        commitment_cost=0.25,
        batch_size=64,
        num_epochs=30,
        lr=1e-3,
        checkpoint_dir='./checkpoints_new',
        latent_save_dir='./latent_per_object_new'
    )