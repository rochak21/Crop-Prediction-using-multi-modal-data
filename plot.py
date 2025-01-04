import json
import matplotlib.pyplot as plt
import numpy as np

# Load predictions from JSON file
predictions_path = './output_dir/mmst_vit/predictions.json'

with open(predictions_path, 'r') as f:
    predictions = json.load(f)

# Extract true and predicted labels
true_labels = np.array(predictions['true_labels'])
predicted_labels = np.array(predictions['predicted_labels'])

# Plotting
# Scatter Plot for each dimension
plt.figure(figsize=(10, 5))
for i in range(true_labels.shape[1]):
    plt.subplot(1, 2, i + 1)
    plt.scatter(true_labels[:, i], predicted_labels[:, i], alpha=0.7, label='Predictions')
    plt.plot([true_labels[:, i].min(), true_labels[:, i].max()],
             [true_labels[:, i].min(), true_labels[:, i].max()],
             color='red', linestyle='--', label='Ideal')
    plt.xlabel(f'True Labels (Dimension {i})')
    plt.ylabel(f'Predicted Labels (Dimension {i})')
    plt.title(f'Dimension {i}: True vs Predicted')
    plt.legend()

plt.tight_layout()
plt.show()

# Line Plot for each dimension
plt.figure(figsize=(10, 5))
for i in range(true_labels.shape[1]):
    plt.subplot(2, 1, i + 1)
    plt.plot(true_labels[:, i], label=f'True Labels (Dimension {i})', marker='o')
    plt.plot(predicted_labels[:, i], label=f'Predicted Labels (Dimension {i})', marker='x')
    plt.xlabel('Sample Index')
    plt.ylabel('Value')
    plt.title(f'Dimension {i}: True vs Predicted')
    plt.legend()

plt.tight_layout()
plt.show()
