import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class GraspDataset(Dataset):
    def __init__(self, latent_folder, grasp_annotations):
        """
        latent_folder: Path to folder containing latent codes (.pt) for each object.
        grasp_annotations: Path to .npy file containing grasp descriptors and labels.
        """
        self.latent_folder = latent_folder
        self.data = np.load(grasp_annotations, allow_pickle=True).item()
        
        # Extract grasp descriptors (concatenate rotation, translation, and joint angles)
        self.grasps = []
        self.labels = []
        self.filenames = []

        for filename, _ in self.data.items():
            self.filenames.append(filename)
            grasp_descriptor = np.concatenate([
                self.data[filename]['rot'].flatten(),   # Rotation matrix (9 elements)
                self.data[filename]['trans'].flatten(), # Translation vector (3 elements)
                self.data[filename]['joint_angles']     # Joint angles (n elements, depends on your setup)
            ])
            self.grasps.append(grasp_descriptor)

            # Collect labels
            self.labels.append([
                self.data[filename]['y_coll'],
                self.data[filename]['y_pick'],
                self.data[filename]['y_PGS']
            ])
        
        self.grasps = np.array(self.grasps, dtype=np.float32)
        self.labels = np.array(self.labels, dtype=np.float32)

        # Pre-load all latent representations into memory
        self.latents = []
        for filename in self.filenames:
            if isinstance(filename, bytes):
                filename = filename.decode('utf-8')
            latent_path = os.path.join(self.latent_folder, filename + '_latent.pt')
            latent_rep = torch.load(latent_path)
            if latent_rep.dim() != 1:
                latent_rep = latent_rep.squeeze()
            self.latents.append(latent_rep)

    def __len__(self):
        return len(self.grasps)
    
    def __getitem__(self, idx):
        grasp_descriptor = torch.tensor(self.grasps[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        latent_rep = self.latents[idx]
        return grasp_descriptor, latent_rep, label


class ResidualBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x):
        return F.relu(self.block(x) + x)


class GraspEvaluator(nn.Module):
    def __init__(self, grasp_dim, latent_dim, hidden_dim=128):
        super().__init__()
        self.input_dim = grasp_dim + latent_dim
        
        # Residual Blocks
        self.res_block1 = ResidualBlock(self.input_dim, hidden_dim)
        self.res_block2 = ResidualBlock(self.input_dim, hidden_dim)
        self.res_block3 = ResidualBlock(self.input_dim, hidden_dim)

        # Final classifiers
        self.collision_head = nn.Linear(self.input_dim, 1)
        self.pick_head = nn.Linear(self.input_dim, 1)
        self.pgs_head = nn.Linear(self.input_dim, 1)

    def forward(self, grasp_descriptor, latent_representation):
        # Concatenate grasp and latent along feature dimension (dim=1)
        x = torch.cat((grasp_descriptor, latent_representation), dim=1)
        
        # Pass through residual blocks
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)

        # Predict logits for each task
        y_coll = self.collision_head(x)
        y_pick = self.pick_head(x)
        y_pgs = self.pgs_head(x)

        return y_coll, y_pick, y_pgs


def train_grasp_evaluator(model, dataloader, device, num_epochs=20, lr=1e-4):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()  # Using logits

    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        for grasp_descriptor, latent_rep, labels in dataloader:
            grasp_descriptor = grasp_descriptor.to(device)
            latent_rep = latent_rep.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            y_coll, y_pick, y_pgs = model(grasp_descriptor, latent_rep)

            # Use squeeze(-1) to avoid dimension mismatch when batch size = 1
            loss_coll = criterion(y_coll.squeeze(-1), labels[:, 0])
            loss_pick = criterion(y_pick.squeeze(-1), labels[:, 1])
            loss_pgs = criterion(y_pgs.squeeze(-1), labels[:, 2])

            loss = loss_coll + loss_pick + loss_pgs
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch + 1}/{num_epochs}] - Loss: {avg_loss:.4f}")
    
    print("Training Complete.")
    return model


def main():
    # Hyperparameters
    latent_folder = "./latents"
    grasp_annotations = "./grasp_annotations.npy"
    batch_size = 8
    num_epochs = 20
    lr = 1e-4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset and Dataloader
    print("Loading Dataset...")
    dataset = GraspDataset(latent_folder, grasp_annotations)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    # Model initialization
    print("Initializing Model...")
    grasp_dim = dataset.grasps.shape[1]
    latent_dim = dataset.latents[0].shape[0]
    model = GraspEvaluator(grasp_dim, latent_dim)

    # Train the model
    print("Training Model...")
    trained_model = train_grasp_evaluator(model, dataloader, device, num_epochs=num_epochs, lr=lr)

    # Save the model
    torch.save(trained_model.state_dict(), "grasp_evaluator.pth")
    print("Model Saved.")


if __name__ == "__main__":
    main()
