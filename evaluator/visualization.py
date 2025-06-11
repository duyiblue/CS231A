import os
import matplotlib.pyplot as plt
import re

experiment_name = "experiment_vqvae_dino_20250607_123636"
# experiment_name = "experiment_provided_object_representation_with_validation_20250516_185812"
log_path = f"/iris/u/duyi/cs231a/evaluator/logs/{experiment_name}_epoch_avg_loss.txt"
plots_dir = f"/iris/u/duyi/cs231a/evaluator/plots/{experiment_name}"
os.makedirs(plots_dir, exist_ok=True)

plot_path = os.path.join(plots_dir, f"{experiment_name}_epoch_avg_loss.png")

# Read log file and extract losses
train_losses = []
val_losses = []
with open(log_path, 'r') as f:
    for line in f:
        match = re.match(r"Epoch (\d+): ([0-9.]+) \(val: ([0-9.]+)\)", line)
        if match:
            train_losses.append(float(match.group(2)))
            val_losses.append(float(match.group(3)))

# Plot
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(train_losses)+1), train_losses, label='Train Loss', color='tab:blue', linestyle='-')
plt.plot(range(1, len(val_losses)+1), val_losses, label='Validation Loss', color='tab:blue', linestyle='--')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss per Epoch')
plt.legend()
plt.grid(True)
plt.ylim(1, 2)
plt.tight_layout()
plt.savefig(plot_path, dpi=1000)
plt.close()


