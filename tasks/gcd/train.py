"""
PREREQUISITE: Run tasks/gcd/create_dataset.py to generate gcd_train.csv and gcd_test.csv
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm

from neural_verification import RNN
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Convert binary string to tensor
def binary_str_to_tensor(s):
    return torch.tensor([int(b) for b in str(s)], dtype=torch.float32).to(device)

# Applies sigmoid to output, then computes weighted BCE
# Rationale is that we want to penalize errors in the more significant bits more
def weighted_bitwise_crossentropy(output, target, weights):
    loss = nn.BCEWithLogitsLoss(weight=weights, reduction='none')(output, target)
    return loss.mean()

# Load dataset
df_train = pd.read_csv("gcd_train.csv", dtype={'first': str, 'second': str, 'gcd': str})
df_test = pd.read_csv("gcd_test.csv", dtype={'first': str, 'second': str, 'gcd': str})


print(df_train.head())  # Check a few rows to ensure the data looks as expected

# Convert to tensors
train_data = [(binary_str_to_tensor(row['first']), binary_str_to_tensor(row['second']), binary_str_to_tensor(row['gcd'])) for _, row in df_train.iterrows()]
test_data = [(binary_str_to_tensor(row['first']), binary_str_to_tensor(row['second']), binary_str_to_tensor(row['gcd'])) for _, row in df_test.iterrows()]

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

model = RNN(input_dim=2, hidden_dim=64, output_dim=16, device=device).to(device)
optimizer = optim.Adam(model.parameters(), lr=3e-4)

# Bit importance weights
weights = torch.pow(2, torch.arange(15, -1, -1).float())

# Train loop
steps = 150
pbar = tqdm(total=steps)
train_losses = []
test_losses = []
train_accuracies = []
test_accuracies = []


for step in range(steps):
    model.train()
    for first, second, gcd in train_loader:
        optimizer.zero_grad()
        # Shape: [batch_size, sequence_length, 2]
        input_pair = torch.stack([first, second], dim=-1)
        # Shape: [batch_size, sequence_length]
        y_pred = model(input_pair)[:, -1, :]

        loss = weighted_bitwise_crossentropy(y_pred, gcd, weights)
        
        loss.backward()
        optimizer.step()
        
        train_losses.append(loss.item())

        y_pred_binary = (torch.sigmoid(y_pred) > 0.5).float()
        correct = (y_pred_binary == gcd).float().sum()
        accuracy = correct / (y_pred.shape[0] * y_pred.shape[1])
        
        train_accuracies.append(accuracy.item())
        
    model.eval()
    with torch.no_grad():
        for first, second, gcd in test_loader:
            input_pair = torch.stack([first, second], dim=-1)
            y_pred = model(input_pair)[:, -1, :]

            loss = weighted_bitwise_crossentropy(y_pred, gcd, weights)
            test_losses.append(loss.item())

            y_pred_binary = (torch.sigmoid(y_pred) > 0.5).float()
            correct = (y_pred_binary == gcd).float().sum()
            accuracy = correct / (y_pred.shape[0] * y_pred.shape[1])

            test_accuracies.append(accuracy.item())
    
    pbar.set_description(f"Train Loss: {np.mean(train_losses[-len(train_loader):]):.4f}, Test Loss: {np.mean(test_losses[-len(test_loader):]):.4f}")
    pbar.update(1)

pbar.close()

metrics = {
    "train_losses": train_losses,
    "test_losses": test_losses,
    "train_accuracies": train_accuracies,
    "test_accuracies": test_accuracies
}

# Plotting
plt.figure()
plt.title("Train and Test Loss")
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.legend()
plt.show()

plt.figure()
plt.title("Train and Test Accuracy")
plt.plot(train_accuracies, label='Train Accuracy')
plt.plot(test_accuracies, label='Test Accuracy')
plt.legend()
plt.show()

# Save metrics and model
torch.save(metrics, "metrics.pt")
torch.save(model.state_dict(), "model.pt")






