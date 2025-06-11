import os
import shutil
import numpy as np
from plyfile import PlyData
from scipy.spatial.transform import Rotation
from sklearn.decomposition import TruncatedSVD

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
    """
    input a .ply file, output a 1d numpy array for features
    """
    plydata = PlyData.read(ply_path)
    vertex_data = plydata['vertex'].data
    N = len(vertex_data)
    means = np.empty((N, 3), dtype=np.float32)
    covs = np.empty((N, 6), dtype=np.float32)

    for i in range(N):
        means[i] = np.array([vertex_data['x'][i], vertex_data['y'][i], vertex_data['z'][i]], dtype=np.float32)
        covs[i] = extract_covariance_for_vertex(vertex_data, i)
    
    features = np.concatenate([means, covs], axis=1)  # (N, 9)

    return features.flatten()  # (N * 9,)



input_dir = "/iris/u/duyi/cs231a/encode_gaussian/splats"
output_dir = "/iris/u/duyi/cs231a/encode_gaussian/pca_256"

input_files = os.listdir(input_dir)
input_files = [file for file in input_files if file.endswith(".ply")]


all_features = []
max_len = 0
for i, file in enumerate(input_files):
    ply_path = os.path.join(input_dir, file)
    features = load_gaussians_from_ply(ply_path)
    
    all_features.append(features)
    max_len = max(max_len, features.shape[0])

    print(f"Loaded the {i+1}-th object, with {features.shape[0]} features")

print(f"\nThe max length of features is {max_len}")

for i in range(len(all_features)):
    all_features[i] = np.pad(all_features[i], (0, max_len - all_features[i].shape[0]), mode='constant')

all_features = np.array(all_features)
print(f"\nThe shape of all features is {all_features.shape}")

svd = TruncatedSVD(n_components=256)
all_features_reduced = svd.fit_transform(all_features)

print(f"\nThe shape of reduced features is {all_features_reduced.shape}")

# Save each reduced vector as a separate .npy file
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
os.makedirs(output_dir)
for i, file in enumerate(input_files):
    npy_filename = file[:-4] + ".npy"
    npy_path = os.path.join(output_dir, npy_filename)
    np.save(npy_path, all_features_reduced[i])
    print(f"Saved the {i+1}-th object's reduced vector to {npy_path}")
