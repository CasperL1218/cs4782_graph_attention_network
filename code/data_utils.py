import argparse
import os

import torch
from torch_geometric.datasets import Planetoid


EXPECTED = {
    "cora": {
        "num_nodes": 2708,
        "num_edges": 10556,   # edge_index stores both directions → 5429 * 2
        "num_classes": 7,
        "num_features": 1433,
        "train_size": 140,
        "val_size": 500,
        "test_size": 1000,
    },
    "citeseer": {
        "num_nodes": 3327,
        "num_edges": 9104,
        "num_classes": 6,
        "num_features": 3703,
        "train_size": 120,
        "val_size": 500,
        "test_size": 1000,
    },
}


def load_dataset(name: str = "Cora", root: str = "/tmp/pyg_data") -> object:
    """Download (if needed) and return the single-graph Planetoid data object."""
    dataset = Planetoid(root=root, name=name)
    data = dataset[0]

    key = name.lower()
    exp = EXPECTED.get(key)

    print(f"\n=== {name} dataset summary ===")
    print(f"  Nodes      : {data.num_nodes:>6}  (expected {exp['num_nodes']})")
    print(f"  Edges (dir): {data.edge_index.shape[1]:>6}  (expected {exp['num_edges']}, i.e. {exp['num_edges']//2} undirected)")
    print(f"  Classes    : {dataset.num_classes:>6}  (expected {exp['num_classes']})")
    print(f"  Features   : {data.num_node_features:>6}  (expected {exp['num_features']})")

    train_n = int(data.train_mask.sum())
    val_n   = int(data.val_mask.sum())
    test_n  = int(data.test_mask.sum())
    print(f"  Train/Val/Test split: {train_n}/{val_n}/{test_n}  "
          f"(expected {exp['train_size']}/{exp['val_size']}/{exp['test_size']})")

    ok = (
        data.num_nodes == exp["num_nodes"]
        and data.edge_index.shape[1] == exp["num_edges"]
        and dataset.num_classes == exp["num_classes"]
        and data.num_node_features == exp["num_features"]
        and train_n == exp["train_size"]
        and val_n == exp["val_size"]
        and test_n == exp["test_size"]
    )
    print(f"\n  Sanity check: {'PASSED' if ok else 'FAILED'}")
    if not ok:
        raise RuntimeError(f"{name} dataset does not match expected statistics.")

    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="Cora",
                        choices=["Cora", "Citeseer"],
                        help="Dataset to load and verify")
    parser.add_argument("--root", type=str, default="/tmp/pyg_data",
                        help="Directory for caching downloaded datasets")
    args = parser.parse_args()

    data = load_dataset(name=args.dataset, root=args.root)
    print(f"\nData object: {data}")
