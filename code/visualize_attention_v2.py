#!/usr/bin/env python3
"""
visualize_attention_v2.py — Extended attention visualization for CS 4782 GAT project.
Extends visualize_attention.py with:
  - Mixed-label node selection for per-head figure (Fig 2)
  - 2-hop directed subgraph visualization (Fig 4)
Original: visualize_attention.py
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
# Mixed-label node selection for per-head figure
# ---------------------------------------------------------------------------

def find_mixed_label_node(dadel, model, ei, alpha1, labels,
                          min_degree=5, max_degree=20):
    """
    Find a test node whose neighbors span multiple classes (high label entropy).
    Returns (best_node, best_entropy, neighbor_label_counts).
    """
    model.eval()
    with torch.no_grad():
        preds = model(dadel.x, dadel.edge_index, return_attn=True)[0].argmax(dim=1).cpu()

    test_mask = dadel.test_mask.cpu()
    correct_test = (test_mask & (preds == labels)).nonzero(as_tuple=True)[0]

    src = ei[0]
    dst = ei[1]

    def _best_among(candidates, enforce_max):
        best_node = None
        best_entropy = -1.0
        best_counts = {}

        for node in candidates:
            node = int(node)
            # edges where this node is the source (receiver in GAT convention)
            edge_mask = src == node
            neighbors = dst[edge_mask]
            # exclude self-loops
            neighbors = neighbors[neighbors != node]
            unique_nbrs = torch.unique(neighbors)
            degree = int(unique_nbrs.shape[0])

            if degree < min_degree:
                continue
            if enforce_max and degree > max_degree:
                continue

            nbr_labels = labels[unique_nbrs.long()].numpy()
            counts = {}
            for lbl in nbr_labels:
                lbl = int(lbl)
                counts[lbl] = counts.get(lbl, 0) + 1

            total = sum(counts.values())
            entropy = 0.0
            for cnt in counts.values():
                p = cnt / total
                entropy += -p * np.log(p + 1e-12)

            if entropy > best_entropy:
                best_entropy = entropy
                best_node = node
                best_counts = {CORA_CLASSES[lbl]: cnt for lbl, cnt in counts.items()}

        return best_node, best_entropy, best_counts

    best_node, best_entropy, best_counts = _best_among(correct_test, enforce_max=True)

    if best_node is None:
        # relax max_degree cap and retry
        best_node, best_entropy, best_counts = _best_among(correct_test, enforce_max=False)

    if best_node is None:
        return (None, 0.0, {})

    return (best_node, best_entropy, best_counts)


# ---------------------------------------------------------------------------
# Single ego-graph drawing utility
# ---------------------------------------------------------------------------

def _draw_ego(ax, node, ei, alpha, labels, pred_labels=None,
              head=None, show_title=True, uniform_line=True,
              show_legend_note=False):
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

    # Edges + attention weight text labels
    for idx, (nbr, wn) in enumerate(zip(nbrs, w_norm)):
        p0, p1 = pos[node], pos[int(nbr)]
        is_self = int(nbr) == node
        if is_self:
            arc_cx = pos[node][0] + 0.18
            arc_cy = pos[node][1] - 0.18
            loop = mpatches.Arc(
                (arc_cx, arc_cy),
                width=0.32,
                height=0.32,
                angle=0,
                theta1=180,
                theta2=90,
                color="#cc4444",
                linewidth=0.5 + wn * 3.5,
                alpha=max(0.3, wn),
                zorder=1,
            )
            ax.add_patch(loop)
            ax.text(pos[node][0] + 0.38, pos[node][1] - 0.28, f"{ws[idx]:.3f}",
                    fontsize=5, ha="center", va="center",
                    color="white",
                    bbox=dict(boxstyle="round,pad=0.1", fc="#aa2222",
                              ec="none", alpha=0.75),
                    zorder=5)
        else:
            ax.plot(
                [p0[0], p1[0]], [p0[1], p1[1]],
                color="#555555",
                linewidth=0.4 + wn * 4.5,
                alpha=max(0.15, wn),
                zorder=1,
                solid_capstyle="round",
            )
            mid_x = (p0[0] + p1[0]) / 2
            mid_y = (p0[1] + p1[1]) / 2
            ax.text(mid_x, mid_y, f"{ws[idx]:.3f}",
                    fontsize=5, ha="center", va="center",
                    color="white",
                    bbox=dict(boxstyle="round,pad=0.1", fc="#333333", ec="none", alpha=0.7),
                    zorder=5)

    # Uniform-attention reference (dashed circle border annotation)
    if uniform_line and n_nbrs > 0:
        uniform_wn = (1.0 / n_nbrs) / w_max
        ax.text(
            -1.35, -1.35,
            f"uniform={uniform_wn:.2f}",
            fontsize=5.5, color="#888888", va="bottom",
        )

    # Nodes — double-circle: outer=predicted label, inner=true label
    all_nodes = [node] + [int(nb) for nb in nbrs if int(nb) != node]
    for nd in all_nodes:
        true_cls = int(labels[nd])
        pred_cls = int(pred_labels[nd]) if pred_labels is not None else true_cls
        is_center = nd == node
        outer_size = 280 if is_center else 130
        inner_size = 160 if is_center else 70

        ax.scatter(pos[nd][0], pos[nd][1],
                   c=[PALETTE[pred_cls]],
                   s=outer_size, zorder=3, linewidths=0)

        ax.scatter(pos[nd][0], pos[nd][1],
                   c=[PALETTE[true_cls]],
                   s=inner_size, zorder=4,
                   linewidths=2 if is_center else 0.8,
                   edgecolors="black")

    if pred_labels is not None and show_legend_note:
        ax.text(0.5, -0.01, "inner=true  outer=predicted  (match=correct)",
                ha="center", va="top", fontsize=5, color="#666666",
                transform=ax.transAxes)

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

def fig_ego_grid(nodes, ei, alpha1, labels, pred_labels, out_path):
    n = len(nodes)
    ncols = 4
    nrows = 2

    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 5 * nrows))
    axes = np.array(axes).flatten()

    for i, (ax, node) in enumerate(zip(axes, nodes)):
        _draw_ego(ax, node, ei, alpha1, labels, pred_labels=pred_labels,
                  show_legend_note=False)

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

    fig.suptitle(
        "GAT Attention Weights — Cora Ego-Graphs\n"
        "Edge thickness ∝ mean α across 8 heads  |  "
        "center node outlined in black  |  red edge = self-loop\n"
        "Node fill = true label  |  outer ring = predicted label  |  "
        "matching colors = correctly classified",
        fontsize=11, y=1.01
    )

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Figure 2: Per-head breakdown for one node
# ---------------------------------------------------------------------------

def fig_heads(node, ei, alpha1, labels, pred_labels, out_path):
    num_heads = alpha1.shape[1]
    ncols = 4
    nrows = 2

    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 7))
    axes = axes.flatten()

    for h in range(num_heads):
        _draw_ego(axes[h], node, ei, alpha1, labels, pred_labels=pred_labels,
                  head=h, show_title=False, uniform_line=False,
                  show_legend_note=False)
        axes[h].set_title(f"Head {h+1}", fontsize=9, pad=3)

    cls_name = CORA_CLASSES[int(labels[node])]
    fig.suptitle(
        f"Per-Head Attention — Node {node}  [{cls_name}]\n"
        "Each subplot shows one attention head's α distribution "
        "over the same neighborhood\n"
        "Node fill = true label  |  outer ring = predicted label  |  "
        "matching colors = correctly classified",
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
# Figure 4 helper: build 2-hop directed subgraph
# ---------------------------------------------------------------------------

def build_2hop_subgraph(center_node, ei, alpha1):
    src = ei[0]
    dst = ei[1]
    src_np = src.numpy()
    dst_np = dst.numpy()

    # 1-hop neighbors: center_node attends over these
    hop1_nodes = set(
        int(dst_np[i]) for i in range(len(src_np))
        if int(src_np[i]) == center_node and int(dst_np[i]) != center_node
    )

    # 2-hop neighbors
    hop2_nodes = set()
    for nbr in hop1_nodes:
        for i in range(len(src_np)):
            if int(src_np[i]) == nbr and int(dst_np[i]) != nbr:
                hop2_nodes.add(int(dst_np[i]))
    hop2_nodes -= hop1_nodes
    hop2_nodes.discard(center_node)

    all_nodes = {center_node} | hop1_nodes | hop2_nodes

    if len(all_nodes) > 80:
        print(
            f"Warning: 2-hop subgraph around node {center_node} has {len(all_nodes)} nodes "
            f"(> 80); capping to center + 1-hop only."
        )
        hop2_nodes = set()
        all_nodes = {center_node} | hop1_nodes

    subgraph_nodes = sorted(all_nodes)

    subgraph_edges = []
    for i in range(len(src_np)):
        u, v = int(src_np[i]), int(dst_np[i])
        if u in all_nodes and v in all_nodes and u != v:
            mean_alpha = float(alpha1[i].mean())
            subgraph_edges.append((u, v, mean_alpha))

    hop_label = {center_node: 0}
    for n in hop1_nodes:
        hop_label[n] = 1
    for n in hop2_nodes:
        hop_label[n] = 2
    for n in subgraph_nodes:
        if n not in hop_label:
            hop_label[n] = 2

    return subgraph_nodes, subgraph_edges, hop_label


# ---------------------------------------------------------------------------
# Figure 4: 2-hop directed subgraph
# ---------------------------------------------------------------------------

def fig_directed_subgraph(center_node, ei, alpha1, labels, pred_labels, out_path):
    subgraph_nodes, subgraph_edges, hop_label = build_2hop_subgraph(center_node, ei, alpha1)

    try:
        import networkx as nx
    except ImportError:
        print("networkx required for fig4: pip install networkx")
        return

    G = nx.DiGraph()
    G.add_nodes_from(subgraph_nodes)
    for u, v, _ in subgraph_edges:
        G.add_edge(u, v)

    pos = nx.spring_layout(G, seed=42)

    fig, ax = plt.subplots(figsize=(14, 12))

    # Double-circle nodes: outer=predicted label, inner=true label
    outer_sizes = {0: 500, 1: 250, 2: 100}
    inner_sizes = {0: 300, 1: 140, 2: 55}
    edge_lws = {0: 2.5, 1: 1.5, 2: 0.5}

    for nd in subgraph_nodes:
        true_cls = int(labels[nd])
        pred_cls = int(pred_labels[nd])
        hop = hop_label[nd]

        ax.scatter(pos[nd][0], pos[nd][1],
                   c=[PALETTE[pred_cls]], s=outer_sizes[hop],
                   zorder=3, linewidths=0)
        ax.scatter(pos[nd][0], pos[nd][1],
                   c=[PALETTE[true_cls]], s=inner_sizes[hop],
                   zorder=4, linewidths=edge_lws[hop],
                   edgecolors="black")

    max_alpha = max((w for _, _, w in subgraph_edges), default=1.0)
    if max_alpha == 0:
        max_alpha = 1.0

    cmap = plt.cm.RdYlBu
    norm = plt.Normalize(vmin=0, vmax=max_alpha)

    edge_set = {(u, v) for u, v, _ in subgraph_edges if u != v}

    for u, v, mean_alpha in subgraph_edges:
        if u == v:
            continue
        lw = 0.3 + (mean_alpha / max_alpha) * 2.7
        edge_color = cmap(norm(mean_alpha))
        opacity = max(0.2, mean_alpha / max_alpha)
        has_reverse = (v, u) in edge_set
        rad = 0.25 if has_reverse else 0.1
        ax.annotate(
            "",
            xy=pos[v], xytext=pos[u],
            arrowprops=dict(
                arrowstyle="->, head_width=0.3, head_length=0.3",
                connectionstyle=f"arc3,rad={rad}",
                color=edge_color,
                lw=lw,
                alpha=opacity,
                shrinkA=12,
                shrinkB=12,
            ),
            zorder=2,
        )

        mid_x = (pos[u][0] + pos[v][0]) / 2
        mid_y = (pos[u][1] + pos[v][1]) / 2
        dx = pos[v][0] - pos[u][0]
        dy = pos[v][1] - pos[u][1]
        length = max((dx**2 + dy**2)**0.5, 1e-6)
        perp_x = -dy / length
        perp_y = dx / length
        offset = 0.04
        label_x = mid_x + perp_x * offset
        label_y = mid_y + perp_y * offset
        ax.text(label_x, label_y, f"{mean_alpha:.3f}",
                fontsize=6, ha="center", va="center",
                color="white",
                bbox=dict(boxstyle="round,pad=0.1", fc="#222222", ec="none", alpha=0.75),
                zorder=6)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label="Mean attention weight α", shrink=0.6)

    legend_handles = [
        ax.scatter([], [], s=120, c=[[0.5, 0.5, 0.5, 1.0]],
                   linewidths=0, label="Center node"),
        ax.scatter([], [], s=60, c=[[0.5, 0.5, 0.5, 1.0]],
                   linewidths=0, label="1-hop neighbor"),
        ax.scatter([], [], s=25, c=[[0.5, 0.5, 0.5, 1.0]],
                   linewidths=0, label="2-hop neighbor"),
        mpatches.Patch(color="none", label="inner fill = true label"),
        mpatches.Patch(color="none", label="outer ring = predicted label"),
    ]
    for i in range(7):
        legend_handles.append(mpatches.Patch(color=PALETTE[i], label=CORA_CLASSES[i]))
    ax.legend(handles=legend_handles, loc="lower left",
              fontsize=8, frameon=True,
              bbox_to_anchor=(0.0, 0.0),
              markerscale=1.0,
              borderpad=1.0,
              labelspacing=0.6)

    ax.set_title(
        f"2-Hop Directed Attention Subgraph — Node {center_node} "
        f"[{CORA_CLASSES[int(labels[center_node])]}]\n"
        "Arrow width & color ∝ mean α (layer 1, averaged over 8 heads)  |  "
        "arrows show direction of attention\n"
        "Center node selected as correctly-classified test node with most "
        "diverse neighbor labels (max neighborhood entropy)",
        fontsize=11,
    )
    ax.axis("off")

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
    parser.add_argument("--subgraph-node", type=int, default=None,
                        help="Center node for the 2-hop directed subgraph figure (Fig 4). "
                             "Defaults to the same node used for Fig 2 (per-head figure).")
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

    # Compute predictions
    model.eval()
    with torch.no_grad():
        logits, _, _ = model(data.x, data.edge_index, return_attn=True)
    pred = logits.argmax(dim=1).cpu()

    # Pick nodes for ego-graph grid
    nodes = pick_nodes(data, model, ei, alpha1, n=7)
    print(f"Selected nodes: {nodes}")
    print(f"  classes: {[CORA_CLASSES[int(labels[n])] for n in nodes]}")

    # Find the highest-entropy mixed-label node for the per-head figure
    mixed_node, mixed_entropy, mixed_counts = find_mixed_label_node(
        data, model, ei, alpha1, labels
    )
    if mixed_node is not None:
        print(f"\nMixed-label node selected: Node {mixed_node} [{CORA_CLASSES[int(labels[mixed_node])]}]")
        print(f"  Neighborhood entropy: {mixed_entropy:.2f} nats")
        print(f"  Neighbor label counts: {mixed_counts}")
        print(f"  (override with --head-node to use a specific node instead)")

    # Figure 1: Ego-graph grid
    fig_ego_grid(
        nodes, ei, alpha1, labels, pred,
        out_path=os.path.join(args.output_dir, "fig1_ego_graphs.png"),
    )

    # Figure 2: Per-head breakdown
    # Use CLI override if given; otherwise use the mixed-label node; fall back to nodes[0]
    if args.head_node is not None:
        head_node = args.head_node
    elif mixed_node is not None:
        head_node = mixed_node
    else:
        head_node = nodes[0]
    fig_heads(
        head_node, ei, alpha1, labels, pred,
        out_path=os.path.join(args.output_dir, "fig2_heads.png"),
    )

    # Figure 3: Statistics
    fig_stats(
        ei, alpha1, labels,
        out_path=os.path.join(args.output_dir, "fig3_stats.png"),
    )

    # Figure 4: 2-hop directed subgraph
    if args.subgraph_node is not None:
        subgraph_node = args.subgraph_node
    else:
        subgraph_node = head_node
    print(f"Fig 4: 2-hop subgraph centered on Node {subgraph_node} "
          f"[{CORA_CLASSES[int(labels[subgraph_node])]}]")
    fig_directed_subgraph(
        subgraph_node, ei, alpha1, labels, pred,
        out_path=os.path.join(args.output_dir, "fig4_directed_subgraph.png"),
    )

    print(f"\nAll figures written to: {os.path.abspath(args.output_dir)}/")


if __name__ == "__main__":
    main()
