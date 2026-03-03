import torch
import matplotlib.pyplot as plt

# Load the saved histories
history = torch.load("history.pt")

fig, axs = plt.subplots(1, 3, figsize=(18, 5))

phases = ['baseline', 'overfit', 'dropout']
titles = ['Phase 2: Underfitting', 'Phase 3: Overfitting (Gap!)', 'Phase 4: Regularization']

for i, phase in enumerate(phases):
    train_loss = history[phase]['train']
    test_loss = history[phase]['test']
    
    axs[i].plot(train_loss, label='Train Loss', color='blue', linewidth=2)
    axs[i].plot(test_loss, label='Test Loss', color='red', linestyle='--', linewidth=2)
    axs[i].set_title(titles[i], fontsize=14)
    axs[i].set_xlabel('Epochs', fontsize=12)
    axs[i].set_ylabel('Loss', fontsize=12)
    axs[i].legend()
    axs[i].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("loss_curves.png")
print("Plot saved as loss_curves.png successfully!")
