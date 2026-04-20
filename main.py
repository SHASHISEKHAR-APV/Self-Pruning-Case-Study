import matplotlib.pyplot as plt
from dataset import get_dataloaders
from trainer import train_one_run
import numpy as np

def plot_gate_distribution(gate_values, lambda_val, filename=None):
    if filename is None:
        filename = f'gate_distribution_lam_{lambda_val}.png'
        
    plt.figure(figsize=(10, 6), facecolor="#f8f9fa")
    ax = plt.gca()
    ax.set_facecolor("#ffffff")
    
    # Vibrant histogram styling with Logarithmic Scale
    plt.hist(gate_values, bins=50, range=(0, 1), 
             color='#4361ee', edgecolor='white', linewidth=1.2, alpha=0.9, log=True)
    
    # Premium Typography
    plt.title(f'Learned Gate Scores Distribution (λ = {lambda_val})', 
              fontsize=16, fontweight='800', color='#2b2d42', pad=20)
    plt.xlabel('Gate Value (0 = Pruned, 1 = Retained)', fontsize=13, fontweight='600', color='#4a4e69', labelpad=10)
    plt.ylabel('Number of Weights (Log Scale)', fontsize=13, fontweight='600', color='#4a4e69', labelpad=10)
    
    # Clean grid and borders
    plt.grid(axis='y', linestyle='--', alpha=0.7, color='#ced4da')
    plt.grid(axis='x', visible=False)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('#adb5bd')
    ax.spines['left'].set_color('#adb5bd')
    
    plt.xticks(fontsize=11, color='#6c757d')
    plt.yticks(fontsize=11, color='#6c757d')
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor="#f8f9fa")
    plt.close()

def main():
    BATCH_SIZE = 128
    NUM_EPOCHS = 5
    LAMBDAS = [0.0001, 0.001, 0.01]

    print("[INFO] Loading CIFAR-10...")
    train_loader, test_loader = get_dataloaders(batch_size=BATCH_SIZE)

    results = {}
    gate_values_per_run = {}

    for lam in LAMBDAS:
        acc, sparsity, gates = train_one_run(
            lambda_sparse=lam,
            train_loader=train_loader,
            test_loader=test_loader,
            num_epochs=NUM_EPOCHS
        )
        results[lam] = (acc, sparsity)
        gate_values_per_run[lam] = gates

    # Save results
    with open('results.txt', 'w') as f:
        f.write("Lambda | Test Accuracy | Sparsity Level\n")
        f.write("-" * 42 + "\n")
        for lam in LAMBDAS:
            acc, sp = results[lam]
            f.write(f"{lam:<7} | {acc:>14.2f}%    | {sp:>10.2f}%\n")

    best_lambda = max(results, key=lambda lam: results[lam][0])
    
    images_saved = []
    for lam in LAMBDAS:
        filename = f'gate_distribution_lam_{lam}.png'
        plot_gate_distribution(gate_values_per_run[lam], lam, filename=filename)
        images_saved.append(filename)

    print(f"\n[DONE] Results saved to results.txt")
    print(f"[INFO] Generated {len(images_saved)} high-quality plots:")
    for img in images_saved:
        print(f"       - {img}")

if __name__ == '__main__':
    main()
