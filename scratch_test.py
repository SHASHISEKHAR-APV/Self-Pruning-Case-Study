import torch
from dataset import get_dataloaders
from model import PrunableNet
import torch.nn as nn
from model import compute_sparsity_loss, compute_sparsity_level

def test_mult(mult, lam=0.01):
    train_loader, test_loader = get_dataloaders(batch_size=256)
    model = PrunableNet().to(torch.device('cpu'))
    
    gate_params = []
    base_params = []
    for name, param in model.named_parameters():
        if 'gate_scores' in name:
            gate_params.append(param)
        else:
            base_params.append(param)
            
    optimizer = torch.optim.Adam([
        {'params': base_params},
        {'params': gate_params, 'lr': 1e-3 * mult}
    ], lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(1): # Just 1 epoch to see if it kills the model
        model.train()
        for images, labels in train_loader:
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels) + lam * compute_sparsity_loss(model)
            loss.backward()
            optimizer.step()
    
    # Eval
    model.eval()
    corr = 0
    tot = 0
    with torch.no_grad():
        for images, labels in test_loader:
            out = model(images)
            _, pred = out.max(1)
            tot += labels.size(0)
            corr += (pred == labels).sum().item()
    print(f"Mult {mult}: Acc={100*corr/tot:.2f}%, Sparsity={compute_sparsity_level(model):.2f}%")

if __name__ == '__main__':
    for m in [1, 5, 10, 20]:
        test_mult(m, 0.01)
