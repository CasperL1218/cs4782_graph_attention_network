# Data

Both datasets used in this project (Cora and Citeseer) are downloaded automatically by PyTorch Geometric's `Planetoid` loader on first run — no manual download needed.

To trigger and verify the download manually:

```bash
# Cora (2708 nodes, 7 classes, 1433 features)
python code/data_utils.py --dataset Cora --root /tmp/pyg_data

# Citeseer (3327 nodes, 6 classes, 3703 features)
python code/data_utils.py --dataset Citeseer --root /tmp/pyg_data
```

The default cache path is `/tmp/pyg_data`. Pass `--root` to override the download location. The script validates node/edge counts against expected values and raises an error if the dataset does not match.
