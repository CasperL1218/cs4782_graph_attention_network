import copy
import argparse

import numpy as np
import torch
import torch.nn.functional as F

from model import GAT
from data_utils import load_dataset


def train(dataset_name="Cora", num_runs=5, num_epochs=100000,
          lr=0.005, weight_decay=5e-4, patience=100, dropout=0.6,
          data_root="/tmp/pyg_data", checkpoint=None):

    data = load_dataset(name=dataset_name, root=data_root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = data.to(device)

    in_features = data.num_node_features
    num_classes = int(data.y.max().item()) + 1

    test_accs = []

    for run in range(num_runs):
        torch.manual_seed(run)

        model = GAT(in_features, num_classes, dropout=dropout).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        best_val_acc = 0.0
        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0

        for epoch in range(1, num_epochs + 1):
            # Training step
            model.train()
            optimizer.zero_grad()
            out = model(data.x, data.edge_index)
            train_loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
            train_loss.backward()
            optimizer.step()

            # Validation step
            model.eval()
            with torch.no_grad():
                out = model(data.x, data.edge_index)
                val_loss = F.nll_loss(out[data.val_mask], data.y[data.val_mask]).item()
                val_pred = out[data.val_mask].argmax(dim=1)
                val_acc = (val_pred == data.y[data.val_mask]).float().mean().item()

            # Early stopping: reset if either val loss or val acc improves
            improved = False
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                improved = True
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = copy.deepcopy(model.state_dict())
                improved = True

            if improved:
                patience_counter = 0
            else:
                patience_counter += 1

            if epoch % 50 == 0:
                print(f"Run {run+1}/{num_runs} | Epoch {epoch:5d} | "
                      f"Train Loss: {train_loss.item():.4f} | "
                      f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.1f}%")

            if patience_counter >= patience:
                print(f"Run {run+1}/{num_runs} | Early stop at epoch {epoch}")
                break

        # Test on best checkpoint
        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            out = model(data.x, data.edge_index)
            test_pred = out[data.test_mask].argmax(dim=1)
            test_acc = (test_pred == data.y[data.test_mask]).float().mean().item()

        print(f"Run {run+1}/{num_runs} | Test Accuracy: {test_acc*100:.1f}%")
        test_accs.append(test_acc)

        # Save best model from the last run if a path is given
        if checkpoint and run == num_runs - 1:
            import os
            torch.save({"state_dict": best_state, "in_features": in_features,
                        "num_classes": num_classes}, checkpoint)
            print(f"Checkpoint saved → {checkpoint}")

    mean = np.mean(test_accs) * 100
    std = np.std(test_accs) * 100
    print(f"\nTest Accuracy over {num_runs} runs: {mean:.1f} ± {std:.1f}%")
    return test_accs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="Cora",
                        choices=["Cora", "Citeseer"])
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=100000)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to save the best model (e.g. cora_best.pt)")
    args = parser.parse_args()

    train(
        dataset_name=args.dataset,
        num_runs=args.runs,
        num_epochs=args.epochs,
        patience=args.patience,
        checkpoint=args.checkpoint,
    )
