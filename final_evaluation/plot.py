random_results = [30, 50, 30, 30, 40]
selected_results = [90, 80, 90, 100, 70]

object_names = ["bottle 1", "bottle 2", "piano", "bowl", "toy person"]

import matplotlib.pyplot as plt
import numpy as np

# Set up the data
x = np.arange(len(object_names))  # Label locations
width = 0.35  # Width of the bars

# Create the figure with high DPI
plt.figure(figsize=(12, 8), dpi=1000)

# Create the bars
bars1 = plt.bar(x - width/2, random_results, width, label='Random Selection', 
                color='#ff6b6b', alpha=0.8, edgecolor='black', linewidth=0.5)
bars2 = plt.bar(x + width/2, selected_results, width, label='Our Evaluator', 
                color='#4ecdc4', alpha=0.8, edgecolor='black', linewidth=0.5)

# Add value labels on top of bars
for bar in bars1:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 2,
             f'{int(height)}%', ha='center', va='bottom', fontsize=18, fontweight='bold')

for bar in bars2:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 2,
             f'{int(height)}%', ha='center', va='bottom', fontsize=18, fontweight='bold')

# Customize the plot
plt.xlabel('Objects', fontsize=20, fontweight='bold')
plt.ylabel('Success Rate (%)', fontsize=20, fontweight='bold')
plt.title('Grasp Success Rate: Random vs Evaluator Selection', fontsize=22, fontweight='bold', pad=25)
plt.xticks(x, object_names, fontsize=16)
plt.yticks(fontsize=16)
plt.ylim(0, 115)
plt.legend(fontsize=18, loc='upper left')

# Add grid for better readability
plt.grid(True, axis='y', alpha=0.3, linestyle='--')

# Improve layout
plt.tight_layout()

# Save the plot with high resolution
plt.savefig('grasp_comparison.png', dpi=1000, bbox_inches='tight', 
            facecolor='white', edgecolor='none')

print("High-resolution plot saved as 'grasp_comparison.png'")
plt.show()

