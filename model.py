import torch
import torch.nn as nn
import torch.nn.functional as F

class PrunableLinear(nn.Module):
    """
    A drop-in replacement for nn.Linear that adds a learnable gate for every
    weight. During the forward pass, each weight is multiplied by the sigmoid
    of its corresponding gate score before the linear transformation.
    """
    def __init__(self, in_features, out_features):
        super(PrunableLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=0)
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.gate_scores = nn.Parameter(torch.ones(out_features, in_features))

    def forward(self, x):
        gates = torch.sigmoid(self.gate_scores)
        pruned_weights = self.weight * gates
        return F.linear(x, pruned_weights, self.bias)

class PrunableNet(nn.Module):
    """
    A simple feed-forward classifier built entirely from PrunableLinear layers.
    Architecture: Flatten -> 3072 -> 1024 -> 512 -> 256 -> 10
    """
    def __init__(self):
        super(PrunableNet, self).__init__()
        self.fc1 = PrunableLinear(3072, 1024)
        self.fc2 = PrunableLinear(1024, 512)
        self.fc3 = PrunableLinear(512, 256)
        self.fc4 = PrunableLinear(256, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

def compute_sparsity_loss(model):
    """
    Computes the sparsity regularisation term (sum of all gate values).
    """
    total_gate_sum = 0.0
    for module in model.modules():
        if isinstance(module, PrunableLinear):
            gates = torch.sigmoid(module.gate_scores)
            total_gate_sum = total_gate_sum + gates.sum()
    return total_gate_sum

def compute_sparsity_level(model, threshold=0.01):
    """
    Measures what fraction of gates are effectively pruned (gate < threshold).
    """
    total_gates = 0
    pruned_gates = 0
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, PrunableLinear):
                gates = torch.sigmoid(module.gate_scores)
                total_gates += gates.numel()
                pruned_gates += (gates < threshold).sum().item()
    if total_gates == 0:
        return 0.0
    return 100.0 * pruned_gates / total_gates
