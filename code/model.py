import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import add_self_loops
from gat_layer import GATLayer


class GAT(nn.Module):
    def __init__(self, in_features, num_classes, dropout=0.6):
        super().__init__()
        self.layer1 = GATLayer(in_features, 8, num_heads=8, dropout=dropout, concat=True)
        self.layer2 = GATLayer(64, num_classes, num_heads=1, dropout=dropout, concat=False)

    def forward(self, x, edge_index):
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        x = F.elu(self.layer1(x, edge_index))
        x = self.layer2(x, edge_index)
        return F.log_softmax(x, dim=1)


if __name__ == "__main__":
    from torch_geometric.datasets import Planetoid

    def sanity_check():
        dataset = Planetoid(root="/tmp/Cora", name="Cora")
        data = dataset[0]
        model = GAT(in_features=dataset.num_node_features, num_classes=dataset.num_classes)
        model.train()

        # Check 1: output shape
        out = model(data.x, data.edge_index)
        expected_shape = (data.num_nodes, dataset.num_classes)
        if out.shape == expected_shape:
            print(f"PASSED shape: {tuple(out.shape)}")
        else:
            print(f"FAILED shape: got {tuple(out.shape)}, expected {expected_shape}")

        # Check 2: loss near ln(num_classes)
        import math
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        expected_loss = math.log(dataset.num_classes)
        if abs(loss.item() - expected_loss) < 1.0:
            print(f"PASSED loss: {loss.item():.4f} (expected ~{expected_loss:.4f})")
        else:
            print(f"FAILED loss: {loss.item():.4f} far from expected ~{expected_loss:.4f}")

        # Check 3: gradients
        loss.backward()
        bad = [
            name for name, p in model.named_parameters()
            if p.grad is None or not torch.isfinite(p.grad).all()
        ]
        if not bad:
            print("PASSED gradients: all parameters have finite gradients")
        else:
            print(f"FAILED gradients: bad params: {bad}")

    sanity_check()
