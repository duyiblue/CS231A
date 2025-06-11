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
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
import json

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
        
        # Calculate embedding usage statistics
        unique_indices, counts = torch.unique(encoding_indices, return_counts=True)
        embedding_usage = len(unique_indices) / self.num_embeddings
        
        return z_q, loss, encoding_indices, embedding_usage


class VQVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, num_embeddings, commitment_cost):
        super(VQVAE, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim)
        self.quantizer = VectorQuantizer(num_embeddings, latent_dim, commitment_cost)
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim)

    def forward(self, x):
        z = self.encoder(x)
        z_q, vq_loss, encoding_indices, embedding_usage = self.quantizer(z)
        x_recon = self.decoder(z_q)
        recon_loss = F.mse_loss(x_recon, x)
        total_loss = recon_loss + vq_loss
        return x_recon, total_loss, vq_loss, recon_loss, embedding_usage, z_q


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
        plt.savefig(f"reconstruction_{i}.png")
        plt.close()


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
    experiment_root_dir='experiments'
):
    # Create unified experiment directory with timestamp
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    experiment_dir = os.path.join(experiment_root_dir, f"experiment_{timestamp}")
    
    # Create subdirectories for different outputs
    checkpoint_dir = os.path.join(experiment_dir, "checkpoints")
    latent_save_dir = os.path.join(experiment_dir, "latents")
    tb_log_dir = os.path.join(experiment_dir, "tensorboard_logs")
    
    # Create all directories
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(latent_save_dir, exist_ok=True)
    os.makedirs(tb_log_dir, exist_ok=True)
    
    # Initialize TensorBoard writer
    writer = SummaryWriter(tb_log_dir)
    
    print(f"Using device: {device}")
    print(f"Experiment directory: {experiment_dir}")
    print(f"├── Checkpoints: {checkpoint_dir}")
    print(f"├── Latents: {latent_save_dir}")
    print(f"└── TensorBoard logs: {tb_log_dir}")

    # Compute dataset mean and std for normalization
    mean, std = compute_dataset_mean_std(ply_folder)
    
    # Save experiment configuration
    config = {
        'experiment_name': f"experiment_{timestamp}",
        'timestamp': timestamp,
        'ply_folder': os.path.abspath(ply_folder),
        'model_params': {
            'chunk_size': chunk_size,
            'input_dim': input_dim,
            'hidden_dim': hidden_dim,
            'latent_dim': latent_dim,
            'num_embeddings': num_embeddings,
            'commitment_cost': commitment_cost
        },
        'training_params': {
            'batch_size': batch_size,
            'num_epochs': num_epochs,
            'learning_rate': lr,
            'device': str(device)
        },
        'dataset_stats': {
            'mean': mean.tolist(),
            'std': std.tolist()
        }
    }
    
    config_path = os.path.join(experiment_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Experiment configuration saved to: {config_path}")
    
    # Create README for the experiment
    readme_content = f"""# VQ-VAE Experiment: {config['experiment_name']}

## Experiment Overview
- **Timestamp**: {timestamp}
- **Dataset**: {ply_folder}
- **Device**: {device}

## Model Architecture
- **Input Dimension**: {input_dim} (3 means + 6 covariances)
- **Hidden Dimension**: {hidden_dim}
- **Latent Dimension**: {latent_dim}
- **Codebook Size**: {num_embeddings} embeddings
- **Commitment Cost**: {commitment_cost}

## Training Configuration
- **Chunk Size**: {chunk_size}
- **Batch Size**: {batch_size}
- **Number of Epochs**: {num_epochs}
- **Learning Rate**: {lr}

## Directory Structure
```
{experiment_dir}/
├── config.json              # Complete experiment configuration
├── README.md               # This file
├── checkpoints/            # Model checkpoints per epoch
├── latents/               # Extracted latent vectors per object
└── tensorboard_logs/      # TensorBoard logging data
```

## Usage
To view training progress:
```bash
tensorboard --logdir={tb_log_dir}
```

To load a checkpoint:
```python
checkpoint = torch.load('checkpoints/vqvae_epoch_<N>.pt')
model.load_state_dict(checkpoint['model_state_dict'])
```
"""
    
    readme_path = os.path.join(experiment_dir, 'README.md')
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    print(f"Experiment documentation saved to: {readme_path}")
    
    # Log hyperparameters to TensorBoard
    writer.add_hparams({
        'chunk_size': chunk_size,
        'input_dim': input_dim,
        'hidden_dim': hidden_dim,
        'latent_dim': latent_dim,
        'num_embeddings': num_embeddings,
        'commitment_cost': commitment_cost,
        'batch_size': batch_size,
        'learning_rate': lr,
        'num_epochs': num_epochs
    }, {})

    model = VQVAE(input_dim, hidden_dim, latent_dim, num_embeddings, commitment_cost).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    ply_paths = glob.glob(os.path.join(ply_folder, '*.ply'))

    # Training loop
    global_step = 0
    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_total_loss = 0.0
        epoch_recon_loss = 0.0
        epoch_vq_loss = 0.0
        epoch_embedding_usage = 0.0
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

            for batch_idx, batch in enumerate(dataloader):
                batch = batch.to(device)
                optimizer.zero_grad()
                x_recon, total_loss, vq_loss, recon_loss, embedding_usage, _ = model(batch)
                total_loss.backward()
                optimizer.step()

                # Accumulate losses for epoch averaging
                epoch_total_loss += total_loss.item()
                epoch_recon_loss += recon_loss.item()
                epoch_vq_loss += vq_loss.item()
                epoch_embedding_usage += embedding_usage  # embedding_usage is already a Python float
                total_batches += 1
                global_step += 1

                # Log batch-level metrics every 100 steps
                if global_step % 100 == 0:
                    writer.add_scalar('Loss/Total_Batch', total_loss.item(), global_step)
                    writer.add_scalar('Loss/Reconstruction_Batch', recon_loss.item(), global_step)
                    writer.add_scalar('Loss/VQ_Batch', vq_loss.item(), global_step)
                    writer.add_scalar('Metrics/Embedding_Usage_Batch', embedding_usage, global_step)  # embedding_usage is already a Python float
                    writer.add_scalar('Metrics/Learning_Rate', optimizer.param_groups[0]['lr'], global_step)

        # Calculate and log epoch averages
        avg_total_loss = epoch_total_loss / max(total_batches, 1)
        avg_recon_loss = epoch_recon_loss / max(total_batches, 1)
        avg_vq_loss = epoch_vq_loss / max(total_batches, 1)
        avg_embedding_usage = epoch_embedding_usage / max(total_batches, 1)
        
        # Log epoch metrics
        writer.add_scalar('Loss/Total_Epoch', avg_total_loss, epoch)
        writer.add_scalar('Loss/Reconstruction_Epoch', avg_recon_loss, epoch)
        writer.add_scalar('Loss/VQ_Epoch', avg_vq_loss, epoch)
        writer.add_scalar('Metrics/Embedding_Usage_Epoch', avg_embedding_usage, epoch)
        
        print(f"Epoch {epoch}/{num_epochs} - Avg Total Loss: {avg_total_loss:.6f}, "
              f"Recon Loss: {avg_recon_loss:.6f}, VQ Loss: {avg_vq_loss:.6f}, "
              f"Embedding Usage: {avg_embedding_usage:.4f}")

        # Save checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, f"vqvae_epoch_{epoch}.pt")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_total_loss,
        }, checkpoint_path)

    # After training, extract and save latent vectors per object
    model.eval()
    with torch.no_grad():
        for ply_path in tqdm(ply_paths, desc="Extracting Latents per Object"):
            dataset = GaussianChunkDataset(ply_path, chunk_size, mean, std)
            dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

            all_latents = []
            for chunk in dataloader:
                chunk = chunk.to(device)
                _, _, _, _, _, z_q = model(chunk)
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

    # Visualize some reconstruction samples from the last object and log to TensorBoard
    dataset = GaussianChunkDataset(ply_paths[-1], chunk_size, mean, std)
    dataloader = DataLoader(dataset, batch_size=5, shuffle=False)
    originals = []
    reconstructions = []
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            x_recon, _, _, _, _, _ = model(batch)
            originals.append(batch.cpu().numpy())
            reconstructions.append(x_recon.cpu().numpy())
            break  # Only take the first batch for visualization
    
    if originals and reconstructions:
        originals_batch = originals[0]  # Shape: (batch_size, chunk_size, input_dim)
        reconstructions_batch = reconstructions[0]
        
        # Log reconstruction samples to TensorBoard
        for i in range(min(3, originals_batch.shape[0])):  # Log first 3 samples
            # Create matplotlib figure
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            
            # Plot original
            ax1.plot(originals_batch[i].flatten())
            ax1.set_title('Original Gaussian Features')
            ax1.set_xlabel('Feature Index')
            ax1.set_ylabel('Value')
            
            # Plot reconstruction
            ax2.plot(reconstructions_batch[i].flatten())
            ax2.set_title('Reconstructed Gaussian Features')
            ax2.set_xlabel('Feature Index')
            ax2.set_ylabel('Value')
            
            plt.tight_layout()
            writer.add_figure(f'Reconstruction/Sample_{i}', fig, num_epochs)
            plt.close(fig)
        
        # Visualize and save reconstruction images
        originals_all = np.vstack(originals)
        reconstructions_all = np.vstack(reconstructions)
        visualize_reconstruction(originals_all, reconstructions_all, num_samples=5)
    
    # Close TensorBoard writer
    writer.close()
    print(f"\n" + "="*60)
    print(f"🎉 Training completed! Experiment saved to: {experiment_dir}")
    print(f"📁 Experiment contents:")
    print(f"├── config.json          # Experiment configuration")
    print(f"├── README.md           # Experiment documentation")
    print(f"├── checkpoints/        # Model checkpoints ({num_epochs} files)")
    print(f"├── latents/           # Latent vectors ({len(ply_paths)} files)")
    print(f"└── tensorboard_logs/  # Training metrics")
    print(f"\n📊 To view training progress:")
    print(f"tensorboard --logdir={tb_log_dir}")
    print(f"\n📖 View experiment details:")
    print(f"cat {os.path.join(experiment_dir, 'README.md')}")
    print("="*60)


if __name__ == "__main__":
    # Set your dataset path here
    ply_folder = './splats'

    train_vqvae(
        ply_folder=ply_folder,
        chunk_size=512,
        input_dim=9,  # 3 means + 6 covariances
        hidden_dim=4096,
        latent_dim=4096,
        num_embeddings=2048,
        commitment_cost=0.25,
        batch_size=64,
        num_epochs=30,
        lr=1e-3,
        experiment_root_dir='./experiments'
    )