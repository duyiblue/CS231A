import os
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import shutil

latent_dir = "/iris/u/duyi/cs231a/encode_gaussian/experiments/experiment_20250607_172942/latents"
grasp_dir = "/iris/u/duyi/cs231a/get_a_grip/data/dataset/small_random/final_evaled_grasp_config_dicts/"

def rotation_matrix_to_6d(rot_matrix):
    """Convert 3x3 rotation matrix to 6D representation (first 2 columns)"""
    # rot_matrix: (batch_size, 3, 3)
    return rot_matrix[..., :2].flatten(-2)  # (batch_size, 6)

def rotation_6d_to_matrix(rot_6d):
    """Convert 6D representation back to 3x3 rotation matrix"""
    # rot_6d: (batch_size, 6)
    x_raw = rot_6d[..., 0:3]  # (batch_size, 3)
    y_raw = rot_6d[..., 3:6]  # (batch_size, 3)
    
    x = x_raw / torch.norm(x_raw, dim=-1, keepdim=True)
    z = torch.cross(x, y_raw, dim=-1)
    z = z / torch.norm(z, dim=-1, keepdim=True)
    y = torch.cross(z, x, dim=-1)
    
    return torch.stack([x, y, z], dim=-1)  # (batch_size, 3, 3)

class Grasp:
    def __init__(self, latent_path, grasp_dict, grasp_idx):
        self.latent_path = latent_path
        
        self.trans = grasp_dict['trans'][grasp_idx]
        self.rot = grasp_dict['rot'][grasp_idx]
        self.joint_angles = grasp_dict['joint_angles'][grasp_idx]
        self.grasp_orientations = grasp_dict['grasp_orientations'][grasp_idx]
        self.y_pick = grasp_dict['y_pick'][grasp_idx]
        self.y_coll = grasp_dict['y_coll'][grasp_idx]
        self.y_pgs = grasp_dict['y_PGS'][grasp_idx]

    def get(self):
        # Convert rotation matrix to 6D representation
        rot_6d = rotation_matrix_to_6d(torch.from_numpy(self.rot).unsqueeze(0)).squeeze(0).numpy()
        
        # Convert grasp orientations to more compact representation
        # Extract only the first column of each 3x3 matrix (primary direction)
        grasp_dirs = self.grasp_orientations[:, :, 0].flatten()  # (4, 3) -> (12,)
        
        dict_to_return = {
            'trans': self.trans,
            'rot_6d': rot_6d,  # 6D instead of 9D
            'joint_angles': self.joint_angles,
            'grasp_dirs': grasp_dirs,  # 12D instead of 36D
            'y_pick': self.y_pick,     # Return all 3 objectives
            'y_coll': self.y_coll,
            'y_pgs': self.y_pgs,
            'object_latent': np.load(self.latent_path).flatten()
        }
        return dict_to_return

class GraspEvaluatorDataset(Dataset):
    def __init__(self, latent_dir, grasp_dir, split='train', split_ratio=(0.8, 0.1, 0.1), seed=42):
        self.latent_dir = latent_dir
        self.grasp_dir = grasp_dir
        self.split = split
        self.split_ratio = split_ratio
        self.seed = seed

        # List of all objects by matching filenames in latent_dir and grasp_dir
        latent_files = os.listdir(latent_dir)
        latent_files = [file for file in latent_files if file.endswith('.npy')]
        grasp_files = os.listdir(grasp_dir)
        grasp_files = [file for file in grasp_files if file.endswith('.npy')]

        assert len(latent_files) == len(grasp_files), f"Number of latent files ({len(latent_files)}) does not match number of grasp files ({len(grasp_files)})"

        print(f"Found {len(grasp_files)} objects with latent+grasp files\n")

        all_grasps = []

        for grasp_file in grasp_files:
            object_name = grasp_file[:-4]  # remove .npy
            latent_name = 'splat_' + object_name + '.npy'
            assert latent_name in latent_files, f"Latent file {latent_name} not found in latent_dir"

            d = np.load(os.path.join(self.grasp_dir, grasp_file), allow_pickle=True).item()
            num_grasps = d['y_pick'].shape[0]
            print(f"- Found {num_grasps} grasps for object {object_name}")

            for idx in range(num_grasps):
                all_grasps.append(Grasp(os.path.join(self.latent_dir, latent_name), d, idx))

        print(f"\nFound {len(all_grasps)} grasps in total")

        # Shuffle and split
        np.random.seed(self.seed)
        indices = np.random.permutation(len(all_grasps))
        n_total = len(all_grasps)
        n_train = int(self.split_ratio[0] * n_total)
        n_val = int(self.split_ratio[1] * n_total)
        n_test = n_total - n_train - n_val

        if split == 'train':
            selected_indices = indices[:n_train]
        elif split == 'val':
            selected_indices = indices[n_train:n_train + n_val]
        elif split == 'test':
            selected_indices = indices[n_train + n_val:]
        else:
            raise ValueError(f"Unknown split: {split}")

        self.grasps = [all_grasps[i] for i in selected_indices]
        print(f"Split '{split}': {len(self.grasps)} grasps")

    def __len__(self):
        return len(self.grasps)

    def __getitem__(self, idx):
        grasp_dict = self.grasps[idx].get()

        # Convert numpy arrays to torch tensors
        return {
            'trans': torch.from_numpy(grasp_dict['trans']).float(),
            'rot_6d': torch.from_numpy(grasp_dict['rot_6d']).float(),
            'joint_angles': torch.from_numpy(grasp_dict['joint_angles']).float(),
            'grasp_dirs': torch.from_numpy(grasp_dict['grasp_dirs']).float(),
            'y_pick': torch.tensor(grasp_dict['y_pick']).float(),
            'y_coll': torch.tensor(grasp_dict['y_coll']).float(),
            'y_pgs': torch.tensor(grasp_dict['y_pgs']).float(),
            'object_latent': torch.from_numpy(grasp_dict['object_latent']).float()
        }

class FCResBlock(nn.Module):
    """
    ResBlock matching get_a_grip's FCResBlock architecture.
    Adapted from: github.com/qianbot/FFHNet/blob/main/FFHNet/models/networks.py#L78
    """
    def __init__(self, Fin, Fout, n_neurons=256, use_bn=True):
        super().__init__()
        self.Fin = Fin
        self.Fout = Fout

        self.fc1 = nn.Linear(Fin, n_neurons)
        self.bn1 = nn.BatchNorm1d(n_neurons)
        self.fc2 = nn.Linear(n_neurons, Fout)
        self.bn2 = nn.BatchNorm1d(Fout)

        # Skip connection projection if input/output dims don't match
        if Fin != Fout:
            self.fc3 = nn.Linear(Fin, Fout)

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.use_bn = use_bn

    def forward(self, x, final_nl=True):
        # Skip connection (project if needed)
        Xin = x if self.Fin == self.Fout else self.leaky_relu(self.fc3(x))

        # Main path
        Xout = self.fc1(x)
        if self.use_bn:
            Xout = self.bn1(Xout)
        Xout = self.leaky_relu(Xout)

        Xout = self.fc2(Xout)
        if self.use_bn:
            Xout = self.bn2(Xout)
        
        # Residual connection
        Xout = Xin + Xout

        if final_nl:
            return self.leaky_relu(Xout)
        return Xout

class GraspEvaluatorModel(nn.Module):
    """
    Evaluator matching get_a_grip's BpsEvaluatorModel architecture.
    Flexible latent_dim (can be 256 for our latents or 4096 for BPS).
    Uses 3 objectives collectively: y_pick, y_coll, y_pgs.
    """
    def __init__(self, latent_dim, grasp_dim=37, n_neurons=2048, internal_neurons=2048):
        super().__init__()
        input_dim = latent_dim + grasp_dim

        # Architecture matching their BpsEvaluatorModel
        self.bn1 = nn.BatchNorm1d(input_dim)
        self.rb1 = FCResBlock(input_dim, n_neurons, n_neurons=internal_neurons)
        self.rb2 = FCResBlock(input_dim + n_neurons, n_neurons, n_neurons=internal_neurons)
        self.rb3 = FCResBlock(input_dim + n_neurons, n_neurons, n_neurons=internal_neurons)
        self.out_success = nn.Linear(n_neurons, 3)  # [y_coll, y_pick, y_pgs]
        self.dropout = nn.Dropout(0.2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, latent, grasp_feats):
        """
        Args:
            latent: (batch_size, latent_dim) - object features
            grasp_feats: (batch_size, grasp_dim) - grasp features
        Returns:
            (y_pick, y_coll, y_pgs): tuple of predictions
        """
        # Concatenate object and grasp features
        X = torch.cat([latent, grasp_feats], dim=1)  # (batch_size, latent_dim + grasp_dim)
        
        # Initial processing
        X0 = self.bn1(X)
        
        # ResBlock 1
        X = self.rb1(X0)
        X = self.dropout(X)
        
        # ResBlock 2 with skip connection from input
        X = self.rb2(torch.cat([X, X0], dim=1))
        X = self.dropout(X)
        
        # ResBlock 3 with skip connection from input
        X = self.rb3(torch.cat([X, X0], dim=1), final_nl=False)
        X = self.dropout(X)
        
        # Output layer
        X = self.out_success(X)
        predictions = self.sigmoid(X)  # (batch_size, 3)
        
        # Return as tuple: (y_pick, y_coll, y_pgs)
        return predictions[:, 1], predictions[:, 0], predictions[:, 2]  # Reorder to match our convention

def train_improved(num_epochs=100, batch_size=128):
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"experiment_2048dim_BCE_large_lr_{current_time}"

    log_file = f"./logs/{experiment_name}.txt"
    os.makedirs("./logs", exist_ok=True)
    if os.path.exists(log_file):
        os.remove(log_file)
    
    epoch_avg_loss_log_file = f"./logs/{experiment_name}_epoch_avg_loss.txt"
    if os.path.exists(epoch_avg_loss_log_file):
        os.remove(epoch_avg_loss_log_file)
    
    os.makedirs("./model_checkpoints", exist_ok=True)
    model_checkpoints_dir = f"./model_checkpoints/{experiment_name}"
    if os.path.exists(model_checkpoints_dir):
        shutil.rmtree(model_checkpoints_dir)
    os.makedirs(model_checkpoints_dir, exist_ok=True)
    
    # Create datasets
    split_ratio = (0.8, 0.1, 0.1)
    train_dataset = GraspEvaluatorDataset(latent_dir, grasp_dir, split='train', split_ratio=split_ratio)
    val_dataset = GraspEvaluatorDataset(latent_dir, grasp_dir, split='val', split_ratio=split_ratio)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Get dimensions from sample
    sample = train_dataset[0]
    print("\nBPS-style model dimensions:")
    print(f"Object latent shape: {sample['object_latent'].shape}")
    print(f"Translation shape: {sample['trans'].shape}")
    print(f"Rotation 6D shape: {sample['rot_6d'].shape}")
    print(f"Joint angles shape: {sample['joint_angles'].shape}")
    print(f"Grasp directions shape: {sample['grasp_dirs'].shape}")
    print(f"Labels - Pick: {sample['y_pick']}, Coll: {sample['y_coll']}, PGS: {sample['y_pgs']}")

    latent_dim = sample['object_latent'].shape[0]
    grasp_dim = sample['trans'].numel() + sample['rot_6d'].numel() + sample['joint_angles'].numel() + sample['grasp_dirs'].numel()
    
    print(f"Total input: {latent_dim} (object) + {grasp_dim} (grasp) = {latent_dim + grasp_dim}")
    print(f"Grasp features: {grasp_dim}D (matching their 37D exactly!)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GraspEvaluatorModel(latent_dim, grasp_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)  # Using AdamW like they do
    criterion = nn.BCELoss()  # Experiment with BCE or MSE

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    model.train()
    for epoch in range(num_epochs):
        epoch_losses = []
        for batch_idx, batch in enumerate(train_loader):
            latent = batch['object_latent'].to(device)
            
            # Concatenate grasp features (37D total)
            grasp_feats = torch.cat([
                batch['trans'].view(batch['trans'].shape[0], -1),        # (batch_size, 3)
                batch['rot_6d'].view(batch['rot_6d'].shape[0], -1),      # (batch_size, 6)
                batch['joint_angles'].view(batch['joint_angles'].shape[0], -1),  # (batch_size, 16)
                batch['grasp_dirs'].view(batch['grasp_dirs'].shape[0], -1),      # (batch_size, 12)
            ], dim=1).to(device)  # Total: 37 dimensions

            # Ground truth labels
            y_pick_true = batch['y_pick'].to(device).float()
            y_coll_true = batch['y_coll'].to(device).float()
            y_pgs_true = batch['y_pgs'].to(device).float()

            optimizer.zero_grad()
            y_pick_pred, y_coll_pred, y_pgs_pred = model(latent, grasp_feats)
            
            # Combined loss using all 3 objectives as hints
            # y_pick and y_coll serve as hints to help learn y_pgs better
            loss_pick = criterion(y_pick_pred, y_pick_true)
            loss_coll = criterion(y_coll_pred, y_coll_true) 
            loss_pgs = criterion(y_pgs_pred, y_pgs_true)
            
            # Weighted combination - emphasize PGS but use others as hints
            loss = 0.5 * loss_pgs + 0.25 * loss_pick + 0.25 * loss_coll
            
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())
            
            if batch_idx % 100 == 0:
                with open(log_file, 'a') as f:
                    f.write(f"[Epoch {epoch+1} | Batch {batch_idx+1}] Loss: {loss.item():.4f} (PGS: {loss_pgs.item():.4f}, Pick: {loss_pick.item():.4f}, Coll: {loss_coll.item():.4f})\n")

        avg_loss = sum(epoch_losses) / len(epoch_losses)
        print(f"[Epoch {epoch+1}] Average Loss: {avg_loss:.4f}")

        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                latent = batch['object_latent'].to(device)
                grasp_feats = torch.cat([
                    batch['trans'].view(batch['trans'].shape[0], -1),
                    batch['rot_6d'].view(batch['rot_6d'].shape[0], -1),
                    batch['joint_angles'].view(batch['joint_angles'].shape[0], -1),
                    batch['grasp_dirs'].view(batch['grasp_dirs'].shape[0], -1),
                ], dim=1).to(device)
                
                y_pick_true = batch['y_pick'].to(device).float()
                y_coll_true = batch['y_coll'].to(device).float()
                y_pgs_true = batch['y_pgs'].to(device).float()
                
                y_pick_pred, y_coll_pred, y_pgs_pred = model(latent, grasp_feats)
                
                loss_pick = criterion(y_pick_pred, y_pick_true)
                loss_coll = criterion(y_coll_pred, y_coll_true)
                loss_pgs = criterion(y_pgs_pred, y_pgs_true)
                loss = 0.5 * loss_pgs + 0.25 * loss_pick + 0.25 * loss_coll
                
                val_losses.append(loss.item())
        
        val_loss = sum(val_losses) / len(val_losses)
        print(f"[Epoch {epoch+1}] Validation Loss: {val_loss:.4f}")
        
        model.train()

        # Save checkpoint
        checkpoint_path = os.path.join(model_checkpoints_dir, f"checkpoint_epoch_{epoch+1}.pt")
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }, checkpoint_path)

        with open(epoch_avg_loss_log_file, 'a') as epoch_log:
            epoch_log.write(f"Epoch {epoch+1}: {avg_loss:.4f} (val: {val_loss:.4f})\n")

if __name__ == "__main__":
    train_improved() 