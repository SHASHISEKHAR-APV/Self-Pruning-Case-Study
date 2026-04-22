# =============================================================================
# train.py — Self-Pruning Neural Network with Learnable Gates
# =============================================================================
# This script trains a neural network that PRUNES ITSELF during training.
# Instead of manually removing weights after training, we attach learnable
# "gate" values to every weight. During training, gates that are not useful
# get pushed toward 0 (effectively removing that weight from the network).
#
# Key idea:
#   pruned_weight = original_weight × sigmoid(gate_score)
#   If gate_score → -∞, sigmoid → 0, so that weight is zeroed out (pruned).
#   If gate_score is large positive, sigmoid → 1, so weight is kept fully.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[INFO] Using device: {device}")



class PrunableLinear(nn.Module):
    """
    A drop-in replacement for nn.Linear that adds a learnable gate for every
    weight. During the forward pass, each weight is multiplied by the sigmoid
    of its corresponding gate score before the linear transformation.

    Args:
        in_features  (int): Number of input features.
        out_features (int): Number of output neurons.
    """

    def __init__(self, in_features, out_features):
        super(PrunableLinear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features

        # ── Standard weight matrix ──────────────────────────────────────────
        
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features)
        )
        nn.init.kaiming_uniform_(self.weight, a=0)   # PyTorch default init

        # ── Bias vector ─────────────────────────────────────────────────────
        # One bias per output neuron.
        self.bias = nn.Parameter(torch.zeros(out_features))

        # ── Gate scores ──────────────────────────────────────────────────────
       
        self.gate_scores = nn.Parameter(
            torch.ones(out_features, in_features)
        )

    def forward(self, x):
        """
        Forward pass with gated weights.

        Steps:
          1. Apply sigmoid to gate_scores  → values in (0, 1)
          2. Element-wise multiply with weight → pruned_weights
          3. Apply standard linear transformation: x @ pruned_weights.T + bias
        """
        # sigmoid squashes gate_scores into [0, 1]
        gates = torch.sigmoid(self.gate_scores)          # shape: (out, in)

        # Hadamard product: each weight is scaled by its gate value
        pruned_weights = self.weight * gates             # shape: (out, in)

        # Standard linear op. F.linear computes: x @ weight.T + bias
        return F.linear(x, pruned_weights, self.bias)


# =============================================================================
# 2. Feed-Forward Network using PrunableLinear
# =============================================================================
class PrunableNet(nn.Module):
    """
    A simple feed-forward classifier for CIFAR-10 built entirely from
    PrunableLinear layers (no nn.Linear used).

    Architecture:
        Flatten → 3072 → 1024 → 512 → 256 → 10
    """

    def __init__(self):
        super(PrunableNet, self).__init__()

        # Layer 1: 3072 inputs (32×32×3 flattened) → 1024 hidden units
        self.fc1 = PrunableLinear(3072, 1024)

        # Layer 2: 1024 → 512 hidden units
        self.fc2 = PrunableLinear(1024, 512)

        # Layer 3: 512 → 256 hidden units
        self.fc3 = PrunableLinear(512, 256)

        # Output layer: 256 → 10 class logits (no activation here;
        # CrossEntropyLoss applies softmax internally)
        self.fc4 = PrunableLinear(256, 10)

    def forward(self, x):
        # Flatten the image: (batch, 3, 32, 32) → (batch, 3072)
        x = x.view(x.size(0), -1)

        # Hidden layers with ReLU activations
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        # Output logits (no softmax — CrossEntropyLoss handles it)
        x = self.fc4(x)
        return x


# =============================================================================
# 3. Sparsity Loss
# =============================================================================
def compute_sparsity_loss(model):
    """
    Computes the sparsity regularisation term.

    For every PrunableLinear layer, we compute sigmoid(gate_scores) and sum
    all of those values. The optimizer will try to minimise this sum, which
    means it will try to push gate values toward 0 (pruning weights).

    This is analogous to L1 regularisation on the gates.

    Args:
        model: The PrunableNet instance.

    Returns:
        A scalar tensor representing the total sparsity loss.
    """
    total_gate_sum = 0.0

    # Iterate over every module in the network
    for module in model.modules():
        if isinstance(module, PrunableLinear):
            # sigmoid maps gate_scores to (0, 1) — same as in forward()
            gates = torch.sigmoid(module.gate_scores)
            # Sum all gate values; optimizer minimises this → gates → 0
            total_gate_sum = total_gate_sum + gates.sum()

    return total_gate_sum


# =============================================================================
# 4. Sparsity Measurement
# =============================================================================
def compute_sparsity_level(model, threshold=0.01):
    """
    Measures what fraction of gates are effectively pruned (gate < threshold).

    A gate below 0.01 contributes < 1% of its weight to the computation —
    we consider it "pruned".

    Args:
        model     : The PrunableNet instance.
        threshold : Gate value below which a weight is considered pruned.

    Returns:
        sparsity (float): Percentage of pruned gates (0–100).
    """
    total_gates = 0
    pruned_gates = 0

    with torch.no_grad():   # no gradients needed for evaluation
        for module in model.modules():
            if isinstance(module, PrunableLinear):
                gates = torch.sigmoid(module.gate_scores)
                total_gates  += gates.numel()              # total number of gates
                pruned_gates += (gates < threshold).sum().item()

    if total_gates == 0:
        return 0.0
    return 100.0 * pruned_gates / total_gates


# =============================================================================
# 5. Data Loading — CIFAR-10
# =============================================================================
def get_dataloaders(batch_size=128):
    """
    Downloads CIFAR-10 and returns DataLoader objects for train and test sets.

    Normalisation: pixel values are shifted from [0,1] to [-1,1] using
    mean=0.5 and std=0.5 for all three colour channels.

    Args:
        batch_size (int): Number of images per mini-batch.

    Returns:
        train_loader, test_loader
    """
    # Define the image preprocessing pipeline
    transform = transforms.Compose([
        transforms.ToTensor(),                                  # [0,255] → [0,1]
        transforms.Normalize(mean=(0.5, 0.5, 0.5),             # [0,1]  → [-1,1]
                             std=(0.5, 0.5, 0.5))
    ])

    # Download and load training set (50,000 images)
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform
    )

    # Download and load test set (10,000 images)
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform
    )

    # DataLoader wraps the dataset and handles batching, shuffling, etc.
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,           # shuffle training data every epoch
        num_workers=2,          # parallel data loading workers
        pin_memory=True         # speeds up CPU→GPU transfer
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,          # no need to shuffle test data
        num_workers=2,
        pin_memory=True
    )

    return train_loader, test_loader


# =============================================================================
# 6. Training Function
# =============================================================================
def train_one_run(lambda_sparse, train_loader, test_loader,
                  num_epochs=20, lr=1e-3):
    """
    Trains PrunableNet from scratch for `num_epochs` epochs.

    Total loss = CrossEntropyLoss + lambda_sparse × SparsityLoss

    Args:
        lambda_sparse  (float): Weight of the sparsity penalty.
        train_loader         : DataLoader for training data.
        test_loader          : DataLoader for test data.
        num_epochs     (int)  : Number of training epochs.
        lr             (float): Adam learning rate.

    Returns:
        test_accuracy  (float): Final test accuracy (%).
        sparsity_level (float): Percentage of pruned gates (%).
        all_gate_values(tensor): All gate values (for histogram plotting).
    """
    print(f"\n{'='*60}")
    print(f"  Training with lambda_sparse = {lambda_sparse}")
    print(f"{'='*60}")

    # ── Instantiate a fresh model and move it to GPU/CPU ───────────────────
    model = PrunableNet().to(device)

    # ── Optimiser: Adam updates BOTH weights and gate_scores ───────────────
    # All nn.Parameters (weight, bias, gate_scores) are included automatically.
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # ── Loss function for classification ───────────────────────────────────
    criterion = nn.CrossEntropyLoss()

    # ── Training loop ──────────────────────────────────────────────────────
    for epoch in range(1, num_epochs + 1):

        model.train()   # enable training mode (activates dropout etc. if any)
        running_ce_loss  = 0.0   # cumulative cross-entropy loss for this epoch
        running_sp_loss  = 0.0   # cumulative sparsity loss
        correct          = 0
        total            = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            # Move data to the correct device (GPU or CPU)
            images = images.to(device)
            labels = labels.to(device)

            # Zero gradients from the previous step
            # (PyTorch accumulates gradients by default)
            optimizer.zero_grad()

            # ── Forward pass ──────────────────────────────────────────────
            logits = model(images)                         # raw class scores

            # ── Classification loss ────────────────────────────────────────
            ce_loss = criterion(logits, labels)

            # ── Sparsity loss ──────────────────────────────────────────────
            sp_loss = compute_sparsity_loss(model)

            # ── Combined loss ──────────────────────────────────────────────
            # lambda_sparse controls the strength of the pruning pressure
            total_loss = ce_loss + lambda_sparse * sp_loss

            # ── Backward pass: compute gradients ───────────────────────────
            total_loss.backward()

            # ── Parameter update ───────────────────────────────────────────
            optimizer.step()

            # ── Accumulate stats ───────────────────────────────────────────
            running_ce_loss += ce_loss.item()
            running_sp_loss += sp_loss.item()

            # Count correct predictions for training accuracy
            _, predicted = torch.max(logits, dim=1)
            total   += labels.size(0)
            correct += (predicted == labels).sum().item()

        # ── End of epoch: print summary ────────────────────────────────────
        train_acc = 100.0 * correct / total
        avg_ce    = running_ce_loss / len(train_loader)
        avg_sp    = running_sp_loss / len(train_loader)
        print(f"  Epoch [{epoch:2d}/{num_epochs}] "
              f"CE Loss: {avg_ce:.4f}  "
              f"Sparsity Loss: {avg_sp:.2f}  "
              f"Train Acc: {train_acc:.2f}%")

    # ── Evaluation on test set ─────────────────────────────────────────────
    model.eval()        # switch to evaluation mode (disables dropout etc.)
    correct = 0
    total   = 0

    with torch.no_grad():   # no gradient computation needed during evaluation
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            _, predicted = torch.max(logits, dim=1)
            total   += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_accuracy  = 100.0 * correct / total
    sparsity_level = compute_sparsity_level(model)

    print(f"\n  ✔ Final Test Accuracy : {test_accuracy:.2f}%")
    print(f"  ✔ Sparsity Level      : {sparsity_level:.2f}%")

    # ── Collect all gate values for the histogram ──────────────────────────
    all_gate_values = []
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, PrunableLinear):
                gates = torch.sigmoid(module.gate_scores)
                # Move to CPU and flatten to a 1-D list
                all_gate_values.append(gates.cpu().numpy().flatten())

    # Concatenate gates from all layers into one array
    all_gate_values = np.concatenate(all_gate_values)

    return test_accuracy, sparsity_level, all_gate_values


# =============================================================================
# 7. Plot Gate Distribution
# =============================================================================
def plot_gate_distribution(gate_values, best_lambda, filename='gate_distribution.png'):
    """
    Plots a histogram of gate values for the best lambda run and saves it.

    A well-trained model should show:
      - A large spike near 0 (many pruned weights)
      - A smaller cluster near 1 (surviving weights)

    Args:
        gate_values  : 1-D numpy array of all sigmoid gate values.
        best_lambda  : The lambda value that produced these gates (for title).
        filename     : Output file name for the plot.
    """
    plt.figure(figsize=(9, 5))

    # 50 bins across [0, 1]; rwidth adds small gaps between bars for clarity
    plt.hist(gate_values, bins=50, range=(0, 1), color='steelblue',
             edgecolor='white', rwidth=0.9)

    plt.title(f'Distribution of Gate Values (Best Lambda = {best_lambda})',
              fontsize=14, fontweight='bold')
    plt.xlabel('Gate Value (0 = pruned, 1 = fully retained)', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.xlim(0, 1)
    plt.grid(axis='y', alpha=0.4)
    plt.tight_layout()

    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"\n[INFO] Gate distribution plot saved → {filename}")


# =============================================================================
# 8. Main — Run All Experiments
# =============================================================================
def main():
    # ── Hyperparameters ────────────────────────────────────────────────────
    BATCH_SIZE  = 128
    NUM_EPOCHS  = 20
    LAMBDAS     = [0.0001, 0.001, 0.01]   # three sparsity strengths

    # ── Load data once; reuse for all three runs ───────────────────────────
    print("[INFO] Loading CIFAR-10 dataset …")
    train_loader, test_loader = get_dataloaders(batch_size=BATCH_SIZE)

    # ── Storage for results ────────────────────────────────────────────────
    results = {}          # { lambda_value: (accuracy, sparsity) }
    gate_values_per_run = {}   # gate arrays, keyed by lambda

    # ── Run training three times ───────────────────────────────────────────
    for lam in LAMBDAS:
        acc, sparsity, gates = train_one_run(
            lambda_sparse=lam,
            train_loader=train_loader,
            test_loader=test_loader,
            num_epochs=NUM_EPOCHS
        )
        results[lam] = (acc, sparsity)
        gate_values_per_run[lam] = gates

    # ── Print summary table ────────────────────────────────────────────────
    print("\n" + "="*55)
    print("  RESULTS SUMMARY")
    print("="*55)
    print(f"  {'Lambda':<10} {'Test Accuracy':>15} {'Sparsity':>12}")
    print(f"  {'-'*10} {'-'*15} {'-'*12}")
    for lam in LAMBDAS:
        acc, sp = results[lam]
        print(f"  {lam:<10} {acc:>14.2f}% {sp:>11.2f}%")

    # ── Determine best lambda (highest test accuracy) ──────────────────────
    best_lambda = max(results, key=lambda lam: results[lam][0])
    print(f"\n  Best lambda (by accuracy): {best_lambda}")

    # ── Plot gate distribution for best lambda ─────────────────────────────
    plot_gate_distribution(
        gate_values=gate_values_per_run[best_lambda],
        best_lambda=best_lambda,
        filename='gate_distribution.png'
    )

    # ── Save results to results.txt ────────────────────────────────────────
    results_path = 'results.txt'
    with open(results_path, 'w') as f:
        f.write("Lambda | Test Accuracy | Sparsity Level\n")
        f.write("-" * 42 + "\n")
        for lam in LAMBDAS:
            acc, sp = results[lam]
            f.write(f"{lam:<7} | {acc:>10.2f}%    | {sp:>10.2f}%\n")

    print(f"[INFO] Results saved → {results_path}")
    print("\n[DONE] All experiments complete.")


# ── Entry point ────────────────────────────────────────────────────────────
if __name__ == '__main__':
    main()
