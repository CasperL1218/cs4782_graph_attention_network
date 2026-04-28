#!/usr/bin/env python3
"""
Attention weight visualization for the Cora GAT.

Produces three figures saved under --output-dir (default: ../figures/):
  fig1_ego_graphs.png  — 2×3 grid of ego-graphs; edge thickness ∝ mean α across heads
  fig2_heads.png       — 2×4 grid showing each of the 8 attention heads for one node
  fig3_stats.png       — attention weight distributions and self-loop vs. neighbor comparison

Usage:
  # Quick training then visualize:
  python visualize_attention.py

  # Load a saved checkpoint:
  python visualize_attention.py --checkpoint cora_best.pt

  # Custom output directory:
  python visualize_attention.py --checkpoint cora_best.pt --output-dir my_figures/
"""

import os
import sys
import copy
import argparse

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec

sys.path.insert(0, os.path.dirname(__file__))
from model import GAT
from data_utils import load_dataset


CORA_CLASSES = [
    "Theory",
    "Reinforcement Learning",
    "Genetic Algorithms",
    "Neural Networks",
    "Probabilistic Methods",
    "Case-Based",
    "Rule Learning",
]
# Distinct colors for 7 classes; use tab10 (first 7 entries)
PALETTE = np.array(plt.cm.tab10(np.arange(7)))


# ---------------------------------------------------------------------------
# Training helper
# ---------------------------------------------------------------------------

def quick_train(data, epochs=600, lr=0.005, weight_decay=5e-4, dropout=0.6, seed=42):
    device = data.x.device
    torch.manual_seed(seed)
    num_classes = int(data.y.max().item()) + 1

    model = GAT(data.num_node_features, num_classes, dropout=dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_acc, best_state = 0.0, None
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        F.nll_loss(out[data.train_mask], data.y[data.train_mask]).backward()
        optimizer.step()

        if epoch % 50 == 0 or epoch == epochs:
            model.eval()
            with torch.no_grad():
                out = model(data.x, data.edge_index)
            val_pred = out[data.val_mask].argmax(dim=1)
            val_acc = (val_pred == data.y[data.val_mask]).float().mean().item()
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = copy.deepcopy(model.state_dict())
            print(f"  Epoch {epoch:4d} | Val Acc: {val_acc*100:.1f}%")

    model.load_state_dict(best_state)
    print(f"  Best val acc: {best_val_acc*100:.1f}%")
    return model


# ---------------------------------------------------------------------------
# Attention extraction
# ---------------------------------------------------------------------------

def extract_attention(model, data):
    """Return (edge_index_with_selfloops, alpha_layer1 [E,8], alpha_layer2 [E,1])."""
    model.eval()
    with torch.no_grad():
        _, (attn1, ei), (attn2, _) = model(data.x, data.edge_index, return_attn=True)
    return ei.cpu(), attn1.cpu(), attn2.cpu()


# ---------------------------------------------------------------------------
# Node selection
# ---------------------------------------------------------------------------

def pick_nodes(data, model, ei, alpha1, n=6, seed=0):
    """
    Return n correctly-classified test nodes, one per class where possible.
    Prefer nodes whose degree (after self-loops) is between 5 and 20 so the
    ego-graph is legible.
    """
    model.eval()
    with torch.no_grad():
        logits, _, _ = model(data.x, data.edge_index, return_attn=True)
    pred = logits.argmax(dim=1).cpu()
    labels = data.y.cpu()

    correct_test = ((data.test_mask.cpu()) & (pred == labels)).nonzero(as_tuple=True)[0].numpy()
    src_np = ei[0].numpy()
    rng = np.random.default_rng(seed)

    chosen = []
    for cls in range(7):
        cands = correct_test[labels[correct_test].numpy() == cls]
        legible = [v for v in cands if 5 <= int((src_np == v).sum()) <= 20]
        pool = legible if legible else list(cands)
        if pool:
            chosen.append(int(rng.choice(pool)))
        if len(chosen) == n:
            break

    # fallback: fill remaining slots from any correct test node
    while len(chosen) < n:
        chosen.append(int(rng.choice(correct_test)))

    return chosen[:n]


# ---------------------------------------------------------------------------
# Single ego-graph drawing utility
# ---------------------------------------------------------------------------

def _draw_ego(ax, node, ei, alpha, labels, head=None, show_title=True, uniform_line=True):
    """
    Draw the 1-hop ego-graph of `node` on `ax`.

    edge_index[0] = receiver, edge_index[1] = sender.
    alpha shape: [E, H].  head=None → average over H.
    """
    src, dst = ei[0].numpy(), ei[1].numpy()
    a = alpha.numpy()

    w = a.mean(axis=1) if head is None else a[:, head]

    mask = src == node
    nbrs = dst[mask]
    ws = w[mask]

    if len(nbrs) == 0:
        ax.text(0.5, 0.5, "isolated", ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
        return

    # Circular layout: center at origin, neighbors on unit circle
    n_nbrs = len(nbrs)
    angles = np.linspace(0, 2 * np.pi, n_nbrs, endpoint=False)
    pos = {node: np.array([0.0, 0.0])}
    for i, nbr in enumerate(nbrs):
        pos[int(nbr)] = np.array([np.cos(angles[i]), np.sin(angles[i])])

    w_max = ws.max() if ws.max() > 0 else 1.0
    w_norm = ws / w_max  # normalize so the strongest edge has width 1

    # Edges
    for nbr, wn in zip(nbrs, w_norm):
        p0, p1 = pos[node], pos[int(nbr)]
        is_self = int(nbr) == node
        ax.plot(
            [p0[0], p1[0]], [p0[1], p1[1]],
            color="#555555" if not is_self else "#cc4444",
            linewidth=0.4 + wn * 4.5,
            alpha=max(0.15, wn),
            zorder=1,
            solid_capstyle="round",
        )

    # Uniform-attention reference (dashed circle border annotation)
    if uniform_line and n_nbrs > 0:
        uniform_wn = (1.0 / n_nbrs) / w_max
        ax.text(
            -1.35, -1.35,
            f"uniform={uniform_wn:.2f}",
            fontsize=5.5, color="#888888", va="bottom",
        )

    # Nodes
    all_nodes = [node] + [int(nb) for nb in nbrs if int(nb) != node]
    for nd in all_nodes:
        c = PALETTE[int(labels[nd])]
        is_center = nd == node
        ax.scatter(
            pos[nd][0], pos[nd][1],
            c=[c], s=160 if is_center else 70,
            zorder=3,
            linewidths=2.0 if is_center else 0.5,
            edgecolors="black" if is_center else "#666666",
        )

    cls_name = CORA_CLASSES[int(labels[node])]
    if show_title:
        ax.set_title(f"Node {node}  [{cls_name}]", fontsize=8, pad=3)
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect("equal")
    ax.axis("off")


# ---------------------------------------------------------------------------
# Figure 1: Ego-graph grid
# ---------------------------------------------------------------------------

def fig_ego_grid(nodes, ei, alpha1, labels, out_path):
    n = len(nodes)
    ncols = 3
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 5 * nrows))
    axes = np.array(axes).flatten()

    for ax, node in zip(axes, nodes):
        _draw_ego(ax, node, ei, alpha1, labels)

    for ax in axes[n:]:
        ax.axis("off")

    # Class legend
    patches = [
        mpatches.Patch(color=PALETTE[i], label=CORA_CLASSES[i])
        for i in range(7)
    ]
    fig.legend(
        handles=patches,
        loc="lower center",
        ncol=4,
        fontsize=9,
        frameon=False,
        bbox_to_anchor=(0.5, -0.04),
    )

    # Shared caption
    fig.text(
        0.5, 1.01,
        "GAT Attention Weights — Cora Ego-Graphs\n"
        "Edge thickness ∝ mean α across 8 heads  |  center node outlined in black  |  red edge = self-loop",
        ha="center", va="bottom", fontsize=10,
    )

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Figure 2: Per-head breakdown for one node
# ---------------------------------------------------------------------------

def fig_heads(node, ei, alpha1, labels, out_path):
    num_heads = alpha1.shape[1]
    ncols = 4
    nrows = 2

    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 7))
    axes = axes.flatten()

    for h in range(num_heads):
        _draw_ego(axes[h], node, ei, alpha1, labels, head=h,
                  show_title=False, uniform_line=False)
        axes[h].set_title(f"Head {h+1}", fontsize=9, pad=3)

    cls_name = CORA_CLASSES[int(labels[node])]
    fig.suptitle(
        f"Per-Head Attention — Node {node}  [{cls_name}]\n"
        "Each subplot shows one attention head's α distribution over the same neighborhood",
        fontsize=11, y=1.02,
    )

    # Class legend
    patches = [
        mpatches.Patch(color=PALETTE[i], label=CORA_CLASSES[i])
        for i in range(7)
    ]
    fig.legend(
        handles=patches,
        loc="lower center",
        ncol=4,
        fontsize=8,
        frameon=False,
        bbox_to_anchor=(0.5, -0.05),
    )

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Figure 3: Attention statistics
# ---------------------------------------------------------------------------

def fig_stats(ei, alpha1, labels, out_path):
    src = ei[0].numpy()
    dst = ei[1].numpy()
    a = alpha1.numpy()  # [E, 8]
    mean_alpha = a.mean(axis=1)  # [E] — mean across heads

    self_loop_mask = src == dst
    nbr_mask = ~self_loop_mask

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # --- Panel A: overall distribution ---
    ax = axes[0]
    ax.hist(mean_alpha[nbr_mask], bins=60, color="#4c78a8", alpha=0.8,
            label="neighbor edges", density=True)
    ax.hist(mean_alpha[self_loop_mask], bins=30, color="#f58518", alpha=0.8,
            label="self-loops", density=True)
    ax.set_xlabel("Mean attention weight α", fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.set_title("A  Distribution of α (layer 1)", fontsize=11)
    ax.legend(fontsize=9)

    # --- Panel B: self-loop vs. neighbor mean per class ---
    ax = axes[1]
    classes = list(range(7))
    self_means, nbr_means = [], []
    for cls in classes:
        node_mask = labels[src] == cls
        sl = mean_alpha[node_mask & self_loop_mask]
        nb = mean_alpha[node_mask & nbr_mask]
        self_means.append(float(sl.mean()) if len(sl) else 0.0)
        nbr_means.append(float(nb.mean()) if len(nb) else 0.0)

    x = np.arange(7)
    w = 0.35
    ax.bar(x - w / 2, self_means, width=w, color="#f58518", label="self-loop α")
    ax.bar(x + w / 2, nbr_means, width=w, color="#4c78a8", label="neighbor α")
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace(" ", "\n") for c in CORA_CLASSES], fontsize=7)
    ax.set_ylabel("Mean α", fontsize=10)
    ax.set_title("B  Self-loop vs. neighbor α by class", fontsize=11)
    ax.legend(fontsize=9)

    # --- Panel C: per-head entropy (specialization) ---
    ax = axes[2]
    entropies = []
    for h in range(alpha1.shape[1]):
        ah = alpha1[:, h].numpy()
        # per-node entropy: sum over node's neighborhood
        node_entropies = []
        for nd in np.unique(src):
            m = src == nd
            p = ah[m]
            if len(p) > 1:
                ent = -float((p * np.log(p + 1e-12)).sum())
                node_entropies.append(ent)
        entropies.append(np.mean(node_entropies))

    ax.bar(np.arange(1, 9), entropies, color=plt.cm.viridis(np.linspace(0.2, 0.8, 8)))
    ax.set_xlabel("Attention head", fontsize=10)
    ax.set_ylabel("Mean entropy H(α)", fontsize=10)
    ax.set_title("C  Attention entropy per head\n(lower = more concentrated)", fontsize=11)
    ax.set_xticks(np.arange(1, 9))

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Visualize GAT attention weights on Cora")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to .pt checkpoint saved by train.py --checkpoint")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory for output figures (default: ../figures/ relative to this file)")
    parser.add_argument("--dataset", type=str, default="Cora", choices=["Cora", "Citeseer"])
    parser.add_argument("--data-root", type=str, default="/tmp/pyg_data")
    parser.add_argument("--train-epochs", type=int, default=600,
                        help="Epochs for quick training when no checkpoint is given")
    parser.add_argument("--head-node", type=int, default=None,
                        help="Node index to use for the per-head figure (auto-selected if omitted)")
    args = parser.parse_args()

    # Output directory
    if args.output_dir is None:
        args.output_dir = os.path.join(os.path.dirname(__file__), "..", "figures")
    os.makedirs(args.output_dir, exist_ok=True)

    # Dataset
    print("Loading dataset...")
    data = load_dataset(name=args.dataset, root=args.data_root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = data.to(device)
    labels = data.y.cpu()
    num_classes = int(data.y.max().item()) + 1

    # Model
    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"Loading checkpoint: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location=device)
        model = GAT(ckpt["in_features"], ckpt["num_classes"]).to(device)
        model.load_state_dict(ckpt["state_dict"])
    else:
        if args.checkpoint:
            print(f"Checkpoint not found at {args.checkpoint}, training from scratch.")
        else:
            print("No checkpoint given — running quick training.")
        model = quick_train(data, epochs=args.train_epochs)

    # Extract attention
    print("Extracting attention weights...")
    ei, alpha1, alpha2 = extract_attention(model, data)

    # Pick nodes for ego-graph grid
    nodes = pick_nodes(data, model, ei, alpha1, n=6)
    print(f"Selected nodes: {nodes}")
    print(f"  classes: {[CORA_CLASSES[int(labels[n])] for n in nodes]}")

    # Figure 1: Ego-graph grid
    fig_ego_grid(
        nodes, ei, alpha1, labels,
        out_path=os.path.join(args.output_dir, "fig1_ego_graphs.png"),
    )

    # Figure 2: Per-head breakdown
    head_node = args.head_node if args.head_node is not None else nodes[0]
    fig_heads(
        head_node, ei, alpha1, labels,
        out_path=os.path.join(args.output_dir, "fig2_heads.png"),
    )

    # Figure 3: Statistics
    fig_stats(
        ei, alpha1, labels,
        out_path=os.path.join(args.output_dir, "fig3_stats.png"),
    )

    print(f"\nAll figures written to: {os.path.abspath(args.output_dir)}/")


if __name__ == "__main__":
    main()
