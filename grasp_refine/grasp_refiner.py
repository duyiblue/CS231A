import os
import shutil
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime
import logging
import time
from collections import defaultdict
import matplotlib.pyplot as plt
import pypose as pp

# Configure logging
logging.basicConfig(
   level=logging.INFO,
   format='[%(asctime)s] %(levelname)s: %(message)s',
   datefmt='%Y-%m-%d %H:%M:%S'
)

# Paths — update checkpoint_path to your model file
latent_dir = "/iris/u/duyi/cs231a/encode_gaussian/experiments/experiment_20250607_172942/latents"
grasp_dir = "/iris/u/duyi/cs231a/get_a_grip/data/dataset/small_random/final_evaled_grasp_config_dicts/"
checkpoint_path = "/iris/u/duyi/cs231a/evaluator/model_checkpoints/experiment_2048dim_Gaussian_BCE_20250610_015353/checkpoint_epoch_88.pt"

# ---- Grasp Class ----
class Grasp:
    def __init__(self, latent_path, grasp_dict, grasp_idx=None, original_idx=None):
        self.latent_path = latent_path
        self.original_idx = original_idx  # Track original index from dataset

        if grasp_idx is not None:
            # Grasp from dataset: access by index
            self.trans = grasp_dict['trans'][grasp_idx]
            self.rot = grasp_dict['rot'][grasp_idx]
            self.joint_angles = grasp_dict['joint_angles'][grasp_idx]
            self.grasp_orientations = grasp_dict['grasp_orientations'][grasp_idx]

            # Optional labels for evaluation
            self.y_pick = grasp_dict.get('y_pick', [None])[grasp_idx]
            self.y_coll = grasp_dict.get('y_coll', [None])[grasp_idx]
            self.y_pgs = grasp_dict.get('y_PGS', [None])[grasp_idx]
            
            # Set original index if not provided
            if self.original_idx is None:
                self.original_idx = grasp_idx
        else:
            # Noisy/refined grasp: direct values
            self.trans = grasp_dict['trans']
            self.rot = grasp_dict['rot']
            self.joint_angles = grasp_dict['joint_angles']
            self.grasp_orientations = grasp_dict['grasp_orientations']

            # Optional labels may not exist yet
            self.y_pick = grasp_dict.get('y_pick', None)
            self.y_coll = grasp_dict.get('y_coll', None)
            self.y_pgs = grasp_dict.get('y_PGS', None)

    def get_features(self):
        return {
            'trans': self.trans,
            'rot': self.rot,
            'joint_angles': self.joint_angles,
            'grasp_orientations': self.grasp_orientations
        }

    def get_latent(self):
        return np.load(self.latent_path).flatten()

# ---- Rotation Conversion Functions (from evaluator_improved.py) ----
def rotation_matrix_to_6d(rot_matrix):
    """Convert 3x3 rotation matrix to 6D representation (first 2 columns)"""
    # rot_matrix: (batch_size, 3, 3) or (3, 3)
    if rot_matrix.dim() == 2:
        rot_matrix = rot_matrix.unsqueeze(0)
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

def extract_grasp_dirs(grasp_orientations):
    """Extract grasp directions from grasp orientations (first column of each 3x3 matrix)"""
    # grasp_orientations: (4, 3, 3) -> (12,) 
    if isinstance(grasp_orientations, np.ndarray):
        return grasp_orientations[:, :, 0].flatten()  # (4, 3) -> (12,)
    else:
        return grasp_orientations[:, :, 0].flatten()  # Handle torch tensors too

# ---- Grasp Evaluator Model (from evaluator_improved.py) ----
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

# ---- Sampler: Load reasonable grasps (y_pgs >= threshold) per object ----
class FixedDatasetGraspSampler:
    def __init__(self, grasp_dir, latent_dir, threshold=0.5, model=None, device=None, num_sampled_candidates=20, top_k=5):
        self.grasp_dir = grasp_dir
        self.latent_dir = latent_dir
        self.threshold = threshold
        self.num_sampled_candidates = num_sampled_candidates
        self.top_k = top_k
        self.high_conf_grasps = self.load_high_confidence_grasps(model, device)

    def load_high_confidence_grasps(self, model, device):
        grasps = []

        for fname in os.listdir(self.grasp_dir):
            if not fname.endswith(".npy"):
                continue

            path = os.path.join(self.grasp_dir, fname)
            data = np.load(path, allow_pickle=True).item()

            y_pgs = data['y_PGS']
            valid_idxs = np.where(y_pgs >= self.threshold)[0]  # simulate a reasonably good grasp generator

            # Randomly select up to num_sampled_candidates
            np.random.shuffle(valid_idxs)
            if len(valid_idxs) < self.num_sampled_candidates:
                num = len(valid_idxs)
                if num < self.top_k:
                    raise ValueError(f"Not enough grasps for object {fname} with {num} grasps and top_k={self.top_k}")
            else:
                num = self.num_sampled_candidates
            selected_idxs = valid_idxs[:num]

            latent_filename = f"splat_{os.path.splitext(fname)[0]}.npy"
            latent_path = os.path.join(self.latent_dir, latent_filename)
            if not os.path.exists(latent_path):
                raise ValueError(f"Latent file {latent_path} does not exist")

            # Load latent
            latent = torch.tensor(np.load(latent_path), dtype=torch.float32).unsqueeze(0).to(device)

            candidate_grasps = []
            grasp_feats = []

            for idx in selected_idxs:
                grasp = Grasp(latent_path, data, idx, original_idx=idx)
                candidate_grasps.append(grasp)
                
                # Convert to 37D format (matching evaluator training)
                rot_6d = rotation_matrix_to_6d(torch.from_numpy(grasp.rot)).squeeze(0).numpy()
                grasp_dirs = extract_grasp_dirs(grasp.grasp_orientations)
                
                feat = np.concatenate([
                    grasp.trans.flatten(),     # 3D
                    rot_6d,                    # 6D (converted from 3x3 matrix)
                    grasp.joint_angles.flatten(), # 16D  
                    grasp_dirs                 # 12D (first column of each 3x3)
                ])  # Total: 37D
                grasp_feats.append(feat)

            grasp_feats_tensor = torch.tensor(np.stack(grasp_feats), dtype=torch.float32).to(device)
            latent_expanded = latent.expand(grasp_feats_tensor.shape[0], -1)

            with torch.no_grad():
                _, _, y_pgs_pred = model(latent_expanded, grasp_feats_tensor)
                # Model already applies sigmoid, no need to apply again

            topk = torch.topk(y_pgs_pred, k=min(self.top_k, len(candidate_grasps)), largest=True)
            topk_scores = topk.values.cpu().numpy()
            topk_idxs = topk.indices.cpu().numpy()

            # Print the top-k initial y_pgs predictions
            logging.info(f"[Sampler] Initial y_PGS predictions for {fname}: {topk_scores}")

            for i in topk_idxs:
                grasps.append(candidate_grasps[i])

        logging.info(f"[Sampler] Loaded {len(grasps)} grasps (top {self.top_k} per object from initial pool of {self.num_sampled_candidates}).")
        return grasps

    def sample(self):
        return self.high_conf_grasps

# ---- Grasp Optimizer with iterative refinement ----
# Updated based on get_a_grip codebase analysis:
# 1. Use acceptance criterion (only keep if better) - matching their approach
# 2. Apply noise parameters matching their defaults: trans=0.005, rot=0.05, joint=0.01, orient=0.05
# 3. Use PyPose for proper SO(3) manifold-based noise generation (key improvement!)
# 4. Clamp joint angles to valid Allegro hand limits after adding noise
# 5. Enhanced logging to track refinement progress
class GraspOptimizer:
    def __init__(self, model, device, noise_scale=1.0):
        self.model = model
        self.device = device
        self.noise_scale = noise_scale  # Scale factor for all noise levels

    # Removed so3_expmap - now using PyPose for proper SO(3) operations

    def add_noise(self, grasp: Grasp):
        # Noise levels matching get_a_grip defaults, scaled by noise_scale
        # Translation noise: 5mm std dev (matching their trans_noise = 0.005)
        noise_trans = np.random.normal(0, 0.005 * self.noise_scale, size=3)

        # Use PyPose for proper SO(3) noise generation (matching their approach)
        device = torch.device("cpu")  # We'll work on CPU then convert back to numpy
        
        # SO(3) noise for wrist rotation: std 0.05 (matching their rot_noise = 0.05)
        rot_tensor = torch.from_numpy(grasp.rot).float().unsqueeze(0)  # Add batch dim
        rot_pypose = pp.from_matrix(rot_tensor, pp.SO3_type, check=False, atol=1e-3, rtol=1e-3)  # Disable strict validation like get_a_grip
        rot_noise = (pp.randn_so3(1, device=device) * (0.05 * self.noise_scale)).Exp()  # Generate proper SO(3) noise
        noisy_rot_pypose = rot_noise @ rot_pypose  # Apply noise using group operation
        noisy_rot = noisy_rot_pypose.matrix().squeeze(0).numpy()  # Convert back to numpy

        # Joint angle noise: std 0.1 radians (matching their Nerf joint_angle_noise = 0.1)  
        # Note: get_a_grip uses 0.1 for Nerf and 0.01 for BPS - we'll use the larger value
        noise_joint = np.random.normal(0, 0.1 * self.noise_scale, size=grasp.joint_angles.shape)
        noisy_joint_angles = grasp.joint_angles + noise_joint
        
        # Clamp joint angles to valid limits (crucial step from get_a_grip)
        # Using approximate Allegro hand limits
        joint_lower_limits = np.array([-0.47, -0.196, -0.174, -0.227] * 4)  # 4 fingers x 4 joints
        joint_upper_limits = np.array([0.47, 1.61, 1.709, 1.618] * 4)        # 4 fingers x 4 joints
        noisy_joint_angles = np.clip(noisy_joint_angles, joint_lower_limits, joint_upper_limits)

        # Apply noise to each grasp orientation using PyPose: std 0.05 (matching their grasp_orientation_noise = 0.05)
        # Convert to PyPose format
        orientations_tensor = torch.from_numpy(grasp.grasp_orientations).float().unsqueeze(0)  # Add batch dim
        orientations_pypose = pp.from_matrix(orientations_tensor, pp.SO3_type, check=False, atol=1e-3, rtol=1e-3)  # Disable strict validation
        
        # Generate noise for all 4 fingers at once
        orientation_noise = (pp.randn_so3(1, 4, device=device) * (0.05 * self.noise_scale)).Exp()  # (1, 4) for 4 fingers
        noisy_orientations_pypose = orientation_noise @ orientations_pypose
        noisy_orientations = noisy_orientations_pypose.matrix().squeeze(0).numpy()  # Remove batch dim

        # Construct the noisy grasp as a new Grasp instance
        grasp_dict = {
            'trans': grasp.trans + noise_trans,
            'rot': noisy_rot.copy(),
            'joint_angles': noisy_joint_angles,
            'grasp_orientations': noisy_orientations.copy()
        }

        return Grasp(grasp.latent_path, grasp_dict, grasp_idx=None, original_idx=grasp.original_idx)

    def evaluate_grasp(self, grasp: Grasp):
        latent = torch.tensor(grasp.get_latent(), dtype=torch.float32).to(self.device).unsqueeze(0)

        # Convert to 37D format (matching evaluator training)
        rot_6d = rotation_matrix_to_6d(torch.from_numpy(grasp.rot)).squeeze(0).numpy()
        grasp_dirs = extract_grasp_dirs(grasp.grasp_orientations)
        
        grasp_feats_np = np.concatenate([
            grasp.trans.flatten(),     # 3D
            rot_6d,                    # 6D (converted from 3x3 matrix)  
            grasp.joint_angles.flatten(), # 16D
            grasp_dirs                 # 12D (first column of each 3x3)
        ])  # Total: 37D
        grasp_feats = torch.tensor(grasp_feats_np, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            y_pick, y_coll, y_pgs = self.model(latent, grasp_feats)
        # Model already applies sigmoid, no need to apply again
        y_pgs_score = y_pgs.item()

        return y_pgs_score

    def refine_grasps(self, initial_grasps, iterations=50):
        num_grasps = len(initial_grasps)
        score_history = np.zeros((iterations, num_grasps))  # [iteration, grasp_index]

        # Track current best grasps and scores
        current_grasps = initial_grasps.copy()
        current_scores = [self.evaluate_grasp(grasp) for grasp in current_grasps]
        
        # Log initial scores  
        logging.info(f"[Optimizer] Initial scores: {[f'{score:.4f}' for score in current_scores]}")
        
        # Additional debugging info
        score_mean = np.mean(current_scores)
        score_std = np.std(current_scores)  
        logging.info(f"[Optimizer] Score statistics: mean={score_mean:.6f}, std={score_std:.6f}, range=[{min(current_scores):.6f}, {max(current_scores):.6f}]")
        
        # Test noise application with detailed logging for first grasp
        if len(current_grasps) > 0:
            original_grasp = current_grasps[0]
            noisy_grasp = self.add_noise(original_grasp)
            
            # Check actual noise magnitudes applied
            trans_diff = np.linalg.norm(noisy_grasp.trans - original_grasp.trans)
            joint_diff = np.linalg.norm(noisy_grasp.joint_angles - original_grasp.joint_angles)
            
            original_score = current_scores[0]
            noisy_score = self.evaluate_grasp(noisy_grasp)
            score_diff = noisy_score - original_score
            
            logging.info(f"[Optimizer] Noise test - Trans diff: {trans_diff:.6f}, Joint diff: {joint_diff:.6f}")
            logging.info(f"[Optimizer] Noise test - Score: {original_score:.6f} -> {noisy_score:.6f} ({score_diff:+.6f})")
            logging.info(f"[Optimizer] Noise levels: trans={0.005 * self.noise_scale:.4f}, joint={0.1 * self.noise_scale:.4f}")
            
            # Test evaluator determinism
            repeat_score = self.evaluate_grasp(original_grasp)
            determinism_diff = repeat_score - original_score
            logging.info(f"[Optimizer] Evaluator determinism test: {original_score:.6f} vs {repeat_score:.6f} (diff: {determinism_diff:+.8f})")
            if abs(determinism_diff) > 1e-6:
                logging.warning(f"[Optimizer] Non-deterministic evaluator detected! This could affect optimization.")
        
        
        improvements_count = 0
        total_attempts = 0
        
        for it in range(iterations):
            # Try to improve each grasp
            for idx in range(num_grasps):
                # Generate candidate from current best
                candidate = self.add_noise(current_grasps[idx])
                candidate_score = self.evaluate_grasp(candidate)
                total_attempts += 1
                
                # Accept if better (get_a_grip style acceptance criterion)
                if candidate_score > current_scores[idx]:
                    improvement = candidate_score - current_scores[idx]
                    logging.info(f"[Optimizer] Iteration {it+1}, Grasp {idx+1}: Improved from {current_scores[idx]:.6f} to {candidate_score:.6f} (+{improvement:.6f})")
                    current_grasps[idx] = candidate
                    current_scores[idx] = candidate_score
                    improvements_count += 1
                
                # Record current best score for this grasp
                score_history[it, idx] = current_scores[idx]
            
            # Log progress every 10 iterations
            if (it + 1) % 10 == 0:
                mean_score = np.mean(current_scores)
                logging.info(f"[Optimizer] Iteration {it+1}/{iterations}: Mean score = {mean_score:.4f}")

        # Log final scores and statistics
        logging.info(f"[Optimizer] Final scores: {[f'{score:.6f}' for score in current_scores]}")
        initial_eval_scores = [self.evaluate_grasp(grasp) for grasp in initial_grasps]
        improvements = np.array(current_scores) - np.array(initial_eval_scores)
        logging.info(f"[Optimizer] Score improvements: {[f'{imp:+.6f}' for imp in improvements]}")
        logging.info(f"[Optimizer] Refinement statistics: {improvements_count}/{total_attempts} attempts successful ({100*improvements_count/total_attempts:.1f}%)")
        
        # Check if no improvements were made
        if improvements_count == 0:
            actual_trans_noise = 0.005 * self.noise_scale
            actual_rot_noise = 0.05 * self.noise_scale
            actual_joint_noise = 0.1 * self.noise_scale  # Updated to match get_a_grip
            actual_orient_noise = 0.05 * self.noise_scale
            
            logging.warning(f"[Optimizer] No improvements found! This may indicate:")
            logging.warning(f"  - Noise levels inappropriate (current scale={self.noise_scale}: trans={actual_trans_noise:.4f}, rot={actual_rot_noise:.4f}, joint={actual_joint_noise:.4f}, orient={actual_orient_noise:.4f})")
            logging.warning(f"  - Initial grasps already near-optimal") 
            logging.warning(f"  - Evaluator model not sensitive enough")
            logging.warning(f"  - Score range: [{min(current_scores):.6f}, {max(current_scores):.6f}]")
            logging.warning(f"  - Consider trying noise_scale=2.0 or 5.0 for larger noise, or 0.1-0.5 for smaller noise")
        
        # Return refined grasps with their scores
        refined = [(current_grasps[i], current_scores[i]) for i in range(num_grasps)]
        return refined, score_history


def plot_score_trajectories(score_history, obj_id, save_path):
    """
    Plot the PGS score trajectories for each grasp over iterations.

    Args:
        score_history (np.ndarray): Array of shape (iterations, num_grasps)
        obj_id (str): Object identifier
        save_path (str): Path to save the plot
    """
    iterations, num_grasps = score_history.shape
    plt.figure(figsize=(12, 8), dpi=1000)  # Increased figure size and DPI for higher resolution
    for i in range(num_grasps):
        plt.plot(range(iterations), score_history[:, i], label=f'Grasp {i+1}', linewidth=2)
    plt.xlabel('Iteration', fontsize=20)
    plt.ylabel('PGS Score', fontsize=20)
    plt.title(f'PGS Score Trajectories for Object {obj_id}', fontsize=20, fontweight='bold')
    plt.legend(fontsize=20)
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.grid(True, alpha=0.7)
    plt.tight_layout()
    plt.savefig(save_path, dpi=1000, bbox_inches='tight', facecolor='white', edgecolor='none')  # High DPI for crisp output
    plt.close()


# ---- Main ----
def main():
    # Set up the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"[Main] Using device: {device}")

    # Define model dimensions
    latent_dim = 2048
    grasp_dim = 37

    # Initialize the GraspEvaluatorModel (matching training architecture)
    logging.info(f"[Main] Initializing GraspEvaluatorModel with latent_dim={latent_dim}, grasp_dim={grasp_dim}")
    model = GraspEvaluatorModel(latent_dim, grasp_dim).to(device)

    # Load model checkpoint
    logging.info(f"[Main] Loading model checkpoint from {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    # Initialize sampler and optimizer
    logging.info("[Main] Initializing sampler and optimizer...")
    sampler = FixedDatasetGraspSampler(grasp_dir, latent_dir, threshold=0.0, model=model, device=device, num_sampled_candidates=5000, top_k=5)
    # Use much smaller noise - 0.1x get_a_grip levels based on debugging findings  
    optimizer = GraspOptimizer(model, device, noise_scale=0.1)

    # Sample initial reasonable grasps
    logging.info("[Main] Sampling initial reasonable grasps...")
    initial_grasps = sampler.sample()
    if not initial_grasps:
        logging.warning("[Main] No reasonable grasps were found. Exiting.")
        return

    logging.info(f"[Main] {len(initial_grasps)} reasonable grasps sampled for refinement.")

    # Group initial grasps by object ID
    grouped_grasps = defaultdict(list)
    for grasp in initial_grasps:
        obj_id = os.path.basename(grasp.latent_path).replace("splat_", "").replace(".npy", "")
        grouped_grasps[obj_id].append(grasp)

    # Verify feature dimensions match training format (37D)
    example = next(iter(grouped_grasps.values()))[0]  # First grasp object of first group
    rot_6d = rotation_matrix_to_6d(torch.from_numpy(example.rot)).squeeze(0).numpy()
    grasp_dirs = extract_grasp_dirs(example.grasp_orientations)
    
    feature_concat = np.concatenate([
        example.trans.flatten(),     # 3D
        rot_6d,                      # 6D (converted from 3x3 matrix)
        example.joint_angles.flatten(), # 16D
        grasp_dirs                   # 12D (first column of each 3x3)
    ])  # Total: 37D
    
    logging.info(f"[Main] Feature dimensions check: {feature_concat.shape[0]}D (expected 37D)")
    assert feature_concat.shape[0] == 37, f"Feature dimension mismatch! Got {feature_concat.shape[0]}, expected 37"

    # Directory to save refined grasps
    base_dir = "refined_grasps_2048_BCE"
    if os.path.exists(base_dir):
        shutil.rmtree(base_dir)
    os.makedirs(os.path.join(base_dir, "grasps"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "score_history"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "score_plots"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "initial_indices"), exist_ok=True)

    # Set the number of iterations per grasp refinement
    iterations = 50
    top_n = 5  # Number of top grasps to keep per object after refinement

    # Refine grasps per object
    refined_per_object = {}
    start_time = time.time()

    # For each object
    for obj_id, grasps in grouped_grasps.items():
        logging.info(f"[Main] Refining grasps for object {obj_id} (count: {len(grasps)})")
        refined, score_history = optimizer.refine_grasps(grasps, iterations=iterations)
        # Sort and keep top_n
        refined_sorted = sorted(refined, key=lambda x: -x[1])[:top_n]
        refined_per_object[obj_id] = refined_sorted

        # Save score history (iterations x num_grasps) for this object
        score_history_path = os.path.join(base_dir, "score_history", f"y_pgs_scores_{obj_id}.npy")
        np.save(score_history_path, score_history)
        logging.info(f"[Main] Saved y_pgs score history of shape {score_history.shape} for object {obj_id}")

        # Plot and save score trajectory
        plot_path = os.path.join(base_dir, "score_plots", f"score_plot_{obj_id}.png")
        plot_score_trajectories(score_history, obj_id, plot_path)
        logging.info(f"[Main] Saved score trajectory plot for object {obj_id} to {plot_path}")

        # Save top-N refined grasps
        grasp_save_path = os.path.join(base_dir, "grasps", f"refined_grasps_{obj_id}.npy")
        
        # Extract arrays and stack them with batch dimension
        trans_list = []
        rot_list = []
        joint_angles_list = []
        grasp_orientations_list = []
        scores_list = []
        original_indices_list = []
        
        for grasp, score in refined_sorted:
            trans_list.append(grasp.trans)
            rot_list.append(grasp.rot)
            joint_angles_list.append(grasp.joint_angles)
            grasp_orientations_list.append(grasp.grasp_orientations)
            scores_list.append(score)
            original_indices_list.append(grasp.original_idx)
        
        # Create single dictionary with batched arrays
        data = {
            "trans": np.stack(trans_list, axis=0),  # shape (top_n, 3)
            "rot": np.stack(rot_list, axis=0),      # shape (top_n, 3, 3)
            "joint_angles": np.stack(joint_angles_list, axis=0),  # shape (top_n, num_joints)
            "grasp_orientations": np.stack(grasp_orientations_list, axis=0),  # shape (top_n, num_orientations, 3, 3)
            "score": np.array(scores_list),         # shape (top_n,)
            "latent_path": refined_sorted[0][0].latent_path  # Same for all grasps from same object
        }
        
        np.save(grasp_save_path, data)
        logging.info(f"[Main] Saved {len(refined_sorted)} refined grasps for object {obj_id} to {grasp_save_path}")

        # Save original indices of top-N refined grasps
        indices_save_path = os.path.join(base_dir, "initial_indices", f"initial_indices_{obj_id}.npy")
        original_indices_array = np.array(original_indices_list)  # shape (top_n,)
        np.save(indices_save_path, original_indices_array)
        logging.info(f"[Main] Saved {len(original_indices_list)} initial indices for object {obj_id} to {indices_save_path}")

    
    duration = time.time() - start_time

    # Log results
    logging.info(f"[Main] Refinement completed in {duration:.2f} seconds.")
    
    logging.info(f"[Main] Top {top_n} refined grasps per object:")
    for obj_id, grasp_list in refined_per_object.items():
        logging.info(f"  Object {obj_id}:")
        for i, (grasp, score) in enumerate(grasp_list):
            logging.info(f"    Grasp {i+1}: PGS Score = {score:.4f}")

    logging.info("[Main] Grasp optimization pipeline finished successfully.")


if __name__ == "__main__":
    main()
