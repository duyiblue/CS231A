import os
import glob
import numpy as np
from plyfile import PlyData
from scipy.spatial.transform import Rotation
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from vqvae_v4 import VQVAE, GaussianChunkDataset, load_gaussians_from_ply, compute_dataset_mean_std, get_num_workers


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


def load_model(checkpoint_path, input_dim, hidden_dim, latent_dim, num_embeddings, commitment_cost, device):
    print(f"Loading model from checkpoint: {checkpoint_path}")
    model = VQVAE(input_dim, hidden_dim, latent_dim, num_embeddings, commitment_cost).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model


def extract_latents(
    ply_folder,
    checkpoint_path,
    chunk_size=512,
    input_dim=9,
    hidden_dim=256,
    latent_dim=256,
    num_embeddings=512,
    commitment_cost=0.25,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    latent_save_dir='latent_per_object'
):
    os.makedirs(latent_save_dir, exist_ok=True)

    print(f"Using device: {device}")

    # Compute dataset mean and std for normalization
    mean, std = compute_dataset_mean_std(ply_folder)

    # Load the model
    model = load_model(checkpoint_path, input_dim, hidden_dim, latent_dim, num_embeddings, commitment_cost, device)

    ply_paths = glob.glob(os.path.join(ply_folder, '*.ply'))

    # Extract and save latent vectors per object
    with torch.no_grad():
        for ply_path in tqdm(ply_paths, desc="Extracting Latents per Object"):
            dataset = GaussianChunkDataset(ply_path, chunk_size, mean, std)
            dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

            all_latents = []
            for chunk in dataloader:
                chunk = chunk.to(device)
                _, _, _, z_q = model(chunk)
                z_q = z_q.squeeze(0)
                all_latents.append(z_q.cpu().numpy())

            # Concatenate all chunks and compute the mean
            all_latents = np.vstack(all_latents)
            latent_vector = np.mean(all_latents, axis=0)  # 1D latent vector of size `latent_dim`

            # Save latent vector
            base_name = os.path.basename(ply_path).replace('.ply', '.npy')
            save_path = os.path.join(latent_save_dir, base_name)
            np.save(save_path, latent_vector)
            print(f"Saved latent vector to: {save_path}")


if __name__ == "__main__":
    # Set your paths here
    ply_folder = './splats'
    checkpoint_path = './checkpoints_new/vqvae_epoch_30.pt'  # Update this to the desired checkpoint

    extract_latents(
        ply_folder=ply_folder,
        checkpoint_path=checkpoint_path,
        chunk_size=512,
        input_dim=9,  # 3 means + 6 covariances
        hidden_dim=256,
        latent_dim=256,
        num_embeddings=512,
        commitment_cost=0.25,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        latent_save_dir='./latent_per_object_at_checkpoint_30'
    )