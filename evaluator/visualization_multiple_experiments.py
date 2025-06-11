import os
import matplotlib.pyplot as plt
import re

experiment_name1 = "experiment_vqvae_v4_ckpt_30_with_validation_20250516_182914"
experiment_name2 = "experiment_provided_object_representation_with_validation_20250516_185812"
experiment_name3 = "experiment_pca_20250516_201629"
experiment_name4 = "experiment_vqvae_dino_20250607_123636"
experiment_name5 = "experiment_4096dim_Gaussian_BCE_20250610_014233"
experiment_name6 = "experiment_4096dim_Gaussian_BCEwithlogits_20250610_014703"
experiment_name7 = "experiment_2048dim_Gaussian_BCE_20250610_015353"

experiment_names = [experiment_name5, experiment_name7]
experiment_names_for_legend = ["4096 dim latent", "2048 dim latent"]
plots_dir = f"/iris/u/duyi/cs231a/evaluator/plots"
os.makedirs(plots_dir, exist_ok=True)
plot_path = os.path.join(plots_dir, f"different_dims_Gaussian_BCE.png")

colors = ['tab:blue', 'tab:pink', 'tab:purple', 'tab:green', 'tab:orange', 'tab:red', 'tab:brown']

plt.figure(figsize=(10, 6))

for idx, experiment_name in enumerate(experiment_names):
    log_path = f"/iris/u/duyi/cs231a/evaluator/logs/{experiment_name}_epoch_avg_loss.txt"
    train_losses = []
    val_losses = []
    with open(log_path, 'r') as f:
        for line in f:
            match = re.match(r"Epoch (\d+): ([0-9.]+) \(val: ([0-9.]+)\)", line)
            if match:
                train_losses.append(float(match.group(2)))
                val_losses.append(float(match.group(3)))
    epochs = range(1, len(train_losses) + 1)
    color = colors[idx % len(colors)]
    plt.plot(epochs, train_losses, label=f"{experiment_names_for_legend[idx]} (train)", color=color, linestyle='-')
    plt.plot(epochs, val_losses, label=f"{experiment_names_for_legend[idx]} (val)", color=color, linestyle='--')

plt.xlabel('Epoch', fontsize=20)
plt.ylabel('Loss', fontsize=20)
plt.title('Training & Validation Loss: VQVAE vs VQVAE+DINO', fontsize=20)
plt.tick_params(axis='both', which='major', labelsize=20)
plt.legend()
plt.grid(True)
plt.ylim(0.4, 0.6)
plt.tight_layout()
plt.savefig(plot_path, dpi=1000)
plt.close()