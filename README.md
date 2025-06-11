# Post Generation Grasp Refinement Pipeline

## Overview
This pipeline implements a complete grasp refinement system that combines 3D object representation learning, grasp quality evaluation, and iterative refinement to improve robotic grasping performance. The system takes 3D Gaussian Splatting representations of objects and learns to generate, evaluate, and refine grasps using learned object encodings.

## Project Structure
```
post_generation_refine
├── encode_gaussian/                    # VQ-VAE and PCA encoding
│   ├── vqvae_encoder.py                # Main VQ-VAE implementation
│   ├── pca.py                          # PCA baseline encoder
│   ├── extract_latents_1d.py           # Extract latents from trained model
│   ├── experiments/                    # VQ-VAE training experiments
│   │   └── experiment_20250607_172942/
│   │       └── latents/                # Pre-extracted 2048D latents (output of the code)
│   └── splats/                         # Raw 3D Gaussian Splatting data (as input)
├── dino_encoding/                      # DINO visual features
│   ├── dino_encoder.py                 # DINO feature extraction
│   └── dino_object_embeddings.pt       # Pre-computed 384D DINO features (output of the code)
├── evaluator/                          # Grasp quality evaluators
│   ├── evaluator.py                    # Standard evaluator (37D features)
│   ├── evaluator_with_dino.py          # DINO-enhanced evaluator (64D features)
│   ├── model_checkpoints/              # Trained evaluator models
│   │   ├── experiment_2048dim_Gaussian_BCE_20250610_015353/
│   │   └── experiment_vqvae_dino_20250603_091956/
│   └── logs/                           # Training logs
├── grasp_refine/                       # Grasp refinement algorithms
│   ├── grasp_refiner_pipeline_new.py   # Standard refinement pipeline
│   ├── grasp_refiner_dino.py           # DINO-enhanced refinement
│   ├── refined_grasps_2048_BCE/        # Refinement outputs
│   └── refined_grasps_dino/
├── final_evaluation/                   # Evaluation infrastructure
│   ├── process_top_grasps.py           # Generate random/original/refined grasps
│   ├── batch_eval_grasps.py            # Automated batch evaluation
│   ├── analyze_results.py              # Statistical analysis and visualization
│   └── README.md                       # Detailed evaluation instructions
└── get_a_grip/                         # Original Get a Grip codebase
    └── get_a_grip/dataset_generation/scripts/
        └── eval_grasp_config_dict.py   # Core simulation evaluator
```

## Dependencies
See `installation.md`.

## 1. Encode Gaussian

### 1.1 VQ-VAE Training
Train a Vector Quantized Variational AutoEncoder to encode 3D Gaussian Splatting representations.

**File:** `encode_gaussian/vqvae_encoder.py`

```bash
cd /iris/u/duyi/cs231a/encode_gaussian
python vqvae_encoder.py
```

**VQ-VAE Architecture:**
- **Encoder:** Maps Gaussian parameters → continuous latent space (2048D)
- **Vector Quantization:** Discretizes latents using learnable codebook (1024 codes)
- **Decoder:** Reconstructs Gaussian parameters from quantized codes
- **Loss Function:** 
  ```python
  loss = reconstruction_loss + commitment_loss + codebook_loss
  # reconstruction_loss: MSE between input and reconstructed Gaussians
  # commitment_loss: ||z_e - sg[z_q]||²  (encoder commits to codebook)
  # codebook_loss: ||sg[z_e] - z_q||²   (codebook updates toward encoder)
  ```

**Extract Latents:**
```bash
python extract_latents_1d.py
# Output: encode_gaussian/experiments/experiment_20250607_172942/latents/
# Format: splat_{object_name}.npy (2048D vectors)
```

### 1.2 PCA Baseline
Simple PCA dimensionality reduction as baseline comparison.

**File:** `encode_gaussian/pca.py`

```bash
cd /iris/u/duyi/cs231a/encode_gaussian
python pca.py
# Output: pca_256/ directory with PCA-reduced latents
```

**PCA Process:**
1. Flatten 3D Gaussian parameters into high-dimensional vectors
2. Apply PCA to reduce to 2048D (matching VQ-VAE latent dimension)
3. Save transformed representations for each object

## 2. Optional DINO Encoder

**File:** `dino_encoding/dino_encoder.py`

```bash
cd /iris/u/duyi/cs231a/dino_encoding
python dino_encoder.py
# Output: dino_object_embeddings.pt
```

**DINO Integration Process:**
1. **Multi-view Rendering:** Render each object from multiple viewpoints (typically 8-12 views)
2. **Feature Extraction:** Extract DINO features using pre-trained ViT-S/16 model
3. **Aggregation:** Average features across all views → 384D representation
4. **Normalization:** L2-normalize the averaged embedding
5. **Storage:** Save as PyTorch tensor dictionary: `{object_name: tensor(384,)}`

**Combined Representation:**
- **Standard:** VQ-VAE latent (2048D) only
- **DINO-Enhanced:** VQ-VAE latent (2048D) + DINO features (384D) = 1024D total (after projection)

## 3. Grasp Evaluator Training

### 3.1 Standard Evaluator Training

**File:** `evaluator/evaluator.py`

```bash
cd /iris/u/duyi/cs231a/evaluator
python evaluator.py
```

**Data Paths:**
- **Latents:** `/iris/u/duyi/cs231a/encode_gaussian/experiments/experiment_20250607_172942/latents`
- **Grasps:** `/iris/u/duyi/cs231a/get_a_grip/data/dataset/small_random/final_evaled_grasp_config_dicts/`

**Input Format (37D):**
```python
# Grasp features: [trans(3) + rot_6d(6) + joints(16) + grasp_dirs(12)] = 37D
trans = grasp.trans.flatten()                    # (3,) position
rot_6d = rotation_matrix_to_6d(grasp.rot)        # (6,) from 3x3 matrix  
joint_angles = grasp.joint_angles.flatten()     # (16,) Allegro configuration
grasp_dirs = grasp_orientations[:,:,0].flatten() # (12,) finger directions
```

**Model Architecture (GraspEvaluatorModel):**
```python
# Input: VQ-VAE(2048D) + grasp(37D) = 2085D total
input_dim = 2048 + 37
model = nn.Sequential(
    nn.BatchNorm1d(input_dim),
    FCResBlock(input_dim, 2048),      # with skip connections
    nn.Dropout(0.2),
    FCResBlock(input_dim + 2048, 2048),
    nn.Dropout(0.2), 
    FCResBlock(input_dim + 2048, 2048),
    nn.Dropout(0.2),
    nn.Linear(2048, 3),               # [y_coll, y_pick, y_pgs]
    nn.Sigmoid()
)
```

**Training Configuration:**
```python
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
criterion = nn.BCELoss()
batch_size = 128
epochs = 100

# Weighted loss emphasizing PGS
loss = 0.5 * loss_pgs + 0.25 * loss_pick + 0.25 * loss_coll
```

**Outputs:**
- **Models:** `evaluator/model_checkpoints/experiment_2048dim_Gaussian_BCE_*/`
- **Logs:** `evaluator/logs/`

### 3.2 DINO-Enhanced Evaluator

**File:** `evaluator/evaluator_with_dino.py`

```bash
cd /iris/u/duyi/cs231a/evaluator  
python evaluator_with_dino.py
```

**Input Format (64D):**
```python
# Grasp features: [trans(3) + rot(9) + joints(16) + orientations(36)] = 64D
trans = grasp.trans.flatten()                      # (3,)
rot = grasp.rot.flatten()                          # (9,) full 3x3 matrix
joint_angles = grasp.joint_angles.flatten()       # (16,)
grasp_orientations = grasp.grasp_orientations.flatten() # (36,) full 4x3x3
```

**Model Architecture (GraspEvaluator):**
```python
# Input: VQ-VAE+DINO(1024D) + grasp(64D) = 1088D total
model = nn.Sequential(
    nn.Linear(1088, 256),
    nn.ReLU(),
    ResidualBlock(256),  # x3
    ResidualBlock(256),
    ResidualBlock(256),
    # Separate heads for pick/coll/pgs
)
```

## 4. Grasp Refinement

### 4.1 Standard Refinement Pipeline

**File:** `grasp_refine/grasp_refiner_pipeline_new.py`

```bash
cd /iris/u/duyi/cs231a/grasp_refine
python grasp_refiner_pipeline_new.py
```

**Configuration (edit file to modify):**
```python
# Paths
latent_dir = "/iris/u/duyi/cs231a/encode_gaussian/experiments/experiment_20250607_172942/latents"
grasp_dir = "/iris/u/duyi/cs231a/get_a_grip/data/dataset/small_random/final_evaled_grasp_config_dicts/"
checkpoint_path = "/iris/u/duyi/cs231a/evaluator/model_checkpoints/experiment_2048dim_Gaussian_BCE_20250610_015353/checkpoint_epoch_88.pt"

# Hyperparameters
num_sampled_candidates = 5000  # Initial candidate pool
top_k = 5                      # Top grasps selected by evaluator
iterations = 50                # Refinement iterations per grasp
noise_scale = 0.1              # Noise magnitude (0.1 = 10% of get_a_grip defaults)
```

**Pipeline Steps:**
1. **Sample Candidates:** 5000 random grasps per object from dataset
2. **Evaluator Selection:** Use trained evaluator to select top-5 highest scoring
3. **Iterative Refinement:** 50 iterations of noise-based hill climbing
4. **Output Storage:** Save refined grasps, score trajectories, and initial indices

**Noise Model (PyPose SO(3)):**
```python
# Translation: Gaussian noise
noise_trans = np.random.normal(0, 0.005 * noise_scale, size=3)  # 5mm std

# Rotation: SO(3) manifold noise  
rot_noise = pp.randn_so3(1) * (0.05 * noise_scale)
noisy_rot = rot_noise.Exp() @ rot_pypose

# Joint angles: Gaussian with clamping
noise_joint = np.random.normal(0, 0.1 * noise_scale, size=16)
noisy_joints = np.clip(joints + noise_joint, joint_limits_low, joint_limits_high)

# Finger orientations: SO(3) per finger
orient_noise = pp.randn_so3(1, 4) * (0.05 * noise_scale)  # 4 fingers
noisy_orients = orient_noise.Exp() @ orientations_pypose
```

**Outputs:**
```
grasp_refine/refined_grasps_2048_BCE/
├── grasps/                    # Top-5 refined grasps per object
│   ├── refined_grasps_{object_id}.npy
│   └── ...
├── score_history/             # Optimization trajectories  
│   ├── y_pgs_scores_{object_id}.npy
│   └── ...
├── score_plots/               # Visualization plots
│   ├── score_plot_{object_id}.png
│   └── ...
└── initial_indices/           # Original dataset indices
    ├── initial_indices_{object_id}.npy
    └── ...
```

### 4.2 DINO-Enhanced Refinement

**File:** `grasp_refine/grasp_refiner_dino.py`

```bash
cd /iris/u/duyi/cs231a/grasp_refine
python grasp_refiner_dino.py
```

**Key Differences:**
- Uses DINO-enhanced evaluator (1024D latents + 64D grasp features)
- Loads latents from `latent_per_object_at_checkpoint_30/`
- Requires DINO embeddings: `dino_encoding/dino_object_embeddings.pt`
- Outputs to `refined_grasps_dino/`

## 5. Grasp Processing for Evaluation

**File:** `final_evaluation/process_top_grasps.py`

```bash
cd /iris/u/duyi/cs231a/final_evaluation
python process_top_grasps.py
```

**Generates three grasp types per object:**
1. **`random_{object_id}.npy`** - 10 randomly sampled grasps (baseline)
2. **`original_grasps_{object_id}.npy`** - Top 10 evaluator-selected grasps
3. **`refined_grasps_{object_id}.npy`** - Top 10 refined grasps

**Output Location:** `final_evaluation/top_grasps_output/`

## 6. Final Evaluation

### 6.1 Single Object Evaluation

**Manual evaluation for individual objects:**

```bash
cd /iris/u/duyi/cs231a/get_a_grip/get_a_grip/dataset_generation/scripts

# Test random grasps
python eval_grasp_config_dict.py \
    --object_code_and_scale_str sem-PersonStanding-1a0710af081df737c50a037462bade42_0_0649 \
    --input_grasp_config_dicts_path /iris/u/duyi/cs231a/final_evaluation/top_grasps_output \
    --gpu 0 \
    --use_gui False \
    --grasp_file_prefix random_

# Test evaluator-selected grasps  
python eval_grasp_config_dict.py \
    --object_code_and_scale_str sem-PersonStanding-1a0710af081df737c50a037462bade42_0_0649 \
    --input_grasp_config_dicts_path /iris/u/duyi/cs231a/final_evaluation/top_grasps_output \
    --gpu 0 \
    --use_gui False \
    --grasp_file_prefix original_grasps_

# Test refined grasps
python eval_grasp_config_dict.py \
    --object_code_and_scale_str sem-PersonStanding-1a0710af081df737c50a037462bade42_0_0649 \
    --input_grasp_config_dicts_path /iris/u/duyi/cs231a/final_evaluation/top_grasps_output \
    --gpu 0 \
    --use_gui False \
    --grasp_file_prefix refined_grasps_
```

### 6.2 Batch Evaluation (Automated)

**File:** `final_evaluation/batch_eval_grasps.py`

```bash
cd /iris/u/duyi/cs231a/final_evaluation

# Basic usage - evaluates all objects automatically
python batch_eval_grasps.py --results_name my_experiment

# Custom grasp directory
python batch_eval_grasps.py \
    --grasp_dir /path/to/your/top_grasps_output \
    --results_name experiment_v2

# Specify GPU
python batch_eval_grasps.py --gpu 1 --results_name gpu1_experiment

# Full parameter control
python batch_eval_grasps.py \
    --grasp_dir /iris/u/duyi/cs231a/final_evaluation/top_grasps_output \
    --results_name refined_vs_baseline \
    --gpu 0
```

**Parallel Evaluation Across Multiple GPUs:**
```bash
# Terminal 1 (GPU 0)
python batch_eval_grasps.py \
    --grasp_dir /path/to/dataset1 \
    --results_name dataset1_results \
    --gpu 0

# Terminal 2 (GPU 1)  
python batch_eval_grasps.py \
    --grasp_dir /path/to/dataset2 \
    --results_name dataset2_results \
    --gpu 1

# Terminal 3 (GPU 2)
python batch_eval_grasps.py \
    --grasp_dir /path/to/dataset3 \
    --results_name dataset3_results \
    --gpu 2
```

**Batch Evaluation Process:**
1. **Auto-discovery:** Finds all objects with all three grasp types (random_, original_grasps_, refined_grasps_)
2. **Sequential Execution:** Runs Isaac Gym simulation for each grasp type per object
3. **Score Extraction:** Parses y_PGS scores from simulation logs
4. **Aggregation:** Collects results into numpy array (num_objects, 3) for [random, original, refined]
5. **Output:** Saves both `.npy` (for analysis) and `.csv` (human-readable) files

**Intermediate Files:**
```
final_evaluation/intermediate_outputs/eval_results_{timestamp}/
├── {object_id}_random_evaled_grasp_config_dict.npy
├── {object_id}_original_grasps_evaled_grasp_config_dict.npy  
├── {object_id}_refined_grasps_evaled_grasp_config_dict.npy
└── ... (for all objects)
```

### 6.3 Results Analysis

**File:** `final_evaluation/analyze_results.py`

```bash
cd /iris/u/duyi/cs231a/final_evaluation

# Analyze results with visualization
python analyze_results.py my_experiment.npy

# Analysis with custom output directory
python analyze_results.py my_experiment.npy --output_dir analysis_results/
```

**Statistical Analysis Performed:**
1. **Descriptive Statistics:** Mean, std, median, quartiles for each method
2. **Pairwise Comparisons:** 
   - Random vs Original (evaluator effectiveness)
   - Original vs Refined (refinement effectiveness)  
   - Random vs Refined (overall improvement)
3. **Statistical Tests:** Paired t-tests with Bonferroni correction
4. **Effect Sizes:** Cohen's d for practical significance
5. **Per-object Analysis:** Performance breakdown by object category

**Visualization Outputs:**
- **Box plots:** Distribution comparison across methods
- **Scatter plots:** Pairwise performance correlation  
- **Trajectory plots:** Per-object improvement patterns
- **Summary tables:** Statistical test results and effect sizes

**Sample Output:**
```
Results Analysis Summary:
======================
Random grasps:        Mean y_PGS = 0.456 ± 0.182
Original grasps:      Mean y_PGS = 0.672 ± 0.145  
Refined grasps:       Mean y_PGS = 0.734 ± 0.128

Improvements:
- Evaluator effect:   +47.4% (p < 0.001, d = 1.34)
- Refinement effect:  +9.2%  (p < 0.001, d = 0.46)
- Overall improvement: +60.9% (p < 0.001, d = 1.72)
```

## 7. Evaluation Infrastructure

### 7.1 Isaac Gym Simulation
- **Physics Engine:** Isaac Gym with Allegro Hand + UR5 arm
- **Objects:** 100 diverse objects (semantic/core/mujoco datasets)
- **Metrics:** 
  - `y_pick`: Pick success rate (grasp acquisition)
  - `y_coll`: Collision avoidance rate  
  - `y_PGS`: Overall success = y_pick × y_coll

### 7.2 Robustness Features
- **Segfault Handling:** Automatic recovery from Isaac Gym crashes using `os._exit(0)`
- **GPU Memory Management:** CUDA cache clearing between evaluations
- **Timestamped Outputs:** Prevents overwriting in parallel execution
- **Progress Tracking:** Real-time logging and progress bars

### 7.3 Expected Performance Validation

**Hypothesis Testing:**
1. **H1:** `original_grasps_` > `random_` (evaluator learns grasp quality)
2. **H2:** `refined_grasps_` > `original_grasps_` (refinement improves grasps)  
3. **H3:** `refined_grasps_` > `random_` (overall pipeline effectiveness)

**Typical Results:**
- **Evaluator Improvement:** 30-50% increase in success rate
- **Refinement Improvement:** 5-15% additional gain
- **Overall Improvement:** 40-65% total improvement over random

This pipeline provides a complete framework for learning-based grasp refinement with rigorous evaluation methodology and comprehensive tooling for reproducible research.