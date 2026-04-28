import torch
import torch.nn as nn
import numpy as np
from model import PrunableNet, compute_sparsity_loss, compute_sparsity_level

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_one_run(lambda_sparse, train_loader, test_loader, num_epochs=5, lr=1e-3):
    """
    Trains PrunableNet for num_epochs epochs.
    """
    print(f"\n[INFO] Training with lambda_sparse = {lambda_sparse}")
    
    model = PrunableNet().to(device)

    # gate_params = []
    #base_params = []
    
    gate_params = []
    base_params = []
    for name, param in model.named_parameters():
        if 'gate_scores' in name:
            gate_params.append(param)
        elif:
            base_params.append(param)
            
    optimizer = torch.optim.Adam([
        {'params': base_params},
        {'params': gate_params, 'lr': lr * 100}
    ], lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, num_epochs + 1):
        model.train()
        running_ce_loss = 0.0
        running_sp_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            logits = model(images)
            ce_loss = criterion(logits, labels)
            sp_loss = compute_sparsity_loss(model)
            total_loss = ce_loss + lambda_sparse * sp_loss

            total_loss.backward()
            optimizer.step()

            running_ce_loss += ce_loss.item()
            running_sp_loss += sp_loss.item()
            _, predicted = torch.max(logits, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_acc = 100.0 * correct / total
        avg_ce = running_ce_loss / len(train_loader)
        print(f"  Epoch [{epoch:2d}/{num_epochs}] CE Loss: {avg_ce:.4f} Train Acc: {train_acc:.2f}%")

    # Evaluation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            _, predicted = torch.max(logits, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_accuracy = 100.0 * correct / total
    sparsity_level = compute_sparsity_level(model)
    print(f"  [OK] Test Accuracy: {test_accuracy:.2f}% | Sparsity: {sparsity_level:.2f}%")

    # Collect gates
    all_gate_values = []
    with torch.no_grad():
        for module in model.modules():
            from model import PrunableLinear
            if isinstance(module, PrunableLinear):
                gates = torch.sigmoid(module.gate_scores)
                all_gate_values.append(gates.cpu().numpy().flatten())
    
    return test_accuracy, sparsity_level, np.concatenate(all_gate_values)
