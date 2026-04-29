import os
import sys

# Allow the data cache path to be overridden via environment variable.
# Defaults to /tmp/pyg_data, which works on Mac and Linux without any config.
DATA_ROOT = os.environ.get("CORA_DATA_ROOT", "/tmp/pyg_data")

sys.path.insert(0, "code")

import streamlit as st
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch_geometric.utils import add_self_loops

from model import GAT
from data_utils import load_dataset
from visualize_attention_v2 import (
    extract_attention, _draw_ego, find_mixed_label_node,
    CORA_CLASSES, PALETTE
)

CHECKPOINT = "cora_best.pt"

st.set_page_config(layout="wide", page_title="GAT Visualizer")
st.title("Graph Attention Network — Cora Interactive Visualizer")

# --- Session state initialization (runs once per session) ---
if "model" not in st.session_state:
    if not os.path.exists(CHECKPOINT):
        st.error(
            f"Checkpoint not found: '{CHECKPOINT}'. "
            "Train the model and place cora_best.pt at the repo root before running the app."
        )
        st.stop()

    data = load_dataset(name="Cora", root=DATA_ROOT)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = data.to(device)

    ckpt = torch.load(CHECKPOINT, map_location=device)
    model = GAT(ckpt["in_features"], ckpt["num_classes"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    # extract_attention requires eval mode (dropout must be off)
    ei, alpha1, alpha2 = extract_attention(model, data)
    # ei: [2, E] CPU, alpha1: [E, 8] CPU, alpha2: [E, 1] CPU

    with torch.no_grad():
        logits = model(data.x, data.edge_index)   # [N, 7] log_softmax
    probs      = logits.exp().cpu()                # [N, 7] softmax probs
    pred       = probs.argmax(dim=1)               # [N] predicted class
    labels     = data.y.cpu()                      # [N] true class
    confidence = probs.max(dim=1).values           # [N] max prob

    st.session_state.update({
        "data": data, "model": model, "device": device,
        "ei": ei, "alpha1": alpha1, "alpha2": alpha2,
        "probs": probs, "pred": pred, "labels": labels,
        "confidence": confidence,
        "num_nodes": data.num_nodes,
    })

tab1, tab2 = st.tabs(["🔍 Node Explorer", "🌐 Graph View"])

# =========================================================================
# Tab 1 — Node Explorer
# =========================================================================
with tab1:
    st.sidebar.header("Node Explorer")

    node_input = st.sidebar.number_input(
        "Node index", min_value=0,
        max_value=st.session_state.num_nodes - 1,
        value=0, step=1
    )

    if st.sidebar.button("🎲 Random Node"):
        node_input = int(np.random.randint(0, st.session_state.num_nodes))

    if st.sidebar.button("❌ Random Misclassified"):
        s = st.session_state
        wrong = (s.pred != s.labels).nonzero(as_tuple=True)[0].numpy()
        node_input = int(np.random.choice(wrong))

    if st.sidebar.button("🔀 Random Mixed-Label"):
        s = st.session_state
        best_node, entropy, counts = find_mixed_label_node(
            s.data, s.model, s.ei, s.alpha1, s.labels
        )
        if best_node is not None:
            node_input = best_node

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Ego-Graph (mean α across 8 heads)")
        fig, ax = plt.subplots(figsize=(5, 5))
        _draw_ego(ax, node_input, st.session_state.ei,
                  st.session_state.alpha1, st.session_state.labels)
        st.pyplot(fig)
        plt.close(fig)

    with col2:
        st.subheader("Per-Head Attention Breakdown")
        fig, axes = plt.subplots(2, 4, figsize=(10, 5))
        for h, ax in enumerate(axes.flatten()):
            _draw_ego(ax, node_input, st.session_state.ei,
                      st.session_state.alpha1, st.session_state.labels,
                      head=h, show_title=False, uniform_line=False)
            ax.set_title(f"Head {h+1}", fontsize=8)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # --- Info table ---
    s = st.session_state
    node = node_input
    true_label = CORA_CLASSES[int(s.labels[node])]
    pred_label = CORA_CLASSES[int(s.pred[node])]
    conf       = float(s.confidence[node])
    correct    = "✅" if s.labels[node] == s.pred[node] else "❌"

    st.markdown(f"""
| Field | Value |
|---|---|
| Node index | {node} |
| True label | {true_label} |
| Predicted label | {pred_label} {correct} |
| Confidence | {conf:.1%} |
""")

    # Top neighbors by mean attention weight across heads
    src_np         = s.ei[0].numpy()
    dst_np         = s.ei[1].numpy()
    mask           = src_np == node
    nbrs           = dst_np[mask]
    weights        = s.alpha1.numpy()[mask].mean(axis=1)
    self_loop_mask = nbrs == node
    order          = np.argsort(weights)[::-1]

    rows = []
    for idx in order:
        nbr   = int(nbrs[idx])
        w     = float(weights[idx])
        is_sl = bool(self_loop_mask[idx])
        label = CORA_CLASSES[int(s.labels[nbr])]
        rows.append({"Neighbor": f"{nbr} {'(self)' if is_sl else ''}",
                     "Class": label, "Mean α": f"{w:.4f}"})

    st.subheader("Neighbor Attention Weights (ranked)")
    st.table(rows)

# =========================================================================
# Tab 2 — Graph View
# =========================================================================
with tab2:
    try:
        from pyvis.network import Network
    except ImportError:
        st.error("pyvis required: pip install pyvis")
        st.stop()
    import streamlit.components.v1 as components

    st.sidebar.header("Graph View")

    mode = st.sidebar.radio(
        "Visualization mode",
        ["Class Filter", "Confidence Overlay", "Attention Concentration"]
    )

    selected_classes = []
    if mode == "Class Filter":
        selected_classes = st.sidebar.multiselect(
            "Show classes", CORA_CLASSES, default=["Theory"]
        )

    # Precompute attention entropy per node (cached after first render)
    s = st.session_state
    if "node_entropy" not in st.session_state:
        src_np    = s.ei[0].numpy()
        alpha_np  = s.alpha1.numpy()        # [E, 8]
        mean_alpha = alpha_np.mean(axis=1)  # [E]
        entropies = np.zeros(s.num_nodes)
        for nd in range(s.num_nodes):
            mask = src_np == nd
            p = mean_alpha[mask]
            if len(p) > 1:
                p = p / (p.sum() + 1e-12)
                entropies[nd] = -float((p * np.log(p + 1e-12)).sum())
        e_min, e_max = entropies.min(), entropies.max()
        st.session_state.node_entropy = (entropies - e_min) / (e_max - e_min + 1e-12)

    def rgba_to_hex(rgba):
        r, g, b = [int(x * 255) for x in rgba[:3]]
        return f"#{r:02x}{g:02x}{b:02x}"

    def build_pyvis_graph(mode, selected_classes, s):
        net = Network(height="700px", width="100%", bgcolor="#1a1a2e",
                      font_color="white")
        net.barnes_hut(spring_length=80, spring_strength=0.01,
                       damping=0.09, overlap=0)

        src_np     = s.ei[0].numpy()
        dst_np     = s.ei[1].numpy()
        alpha_np   = s.alpha1.numpy().mean(axis=1)  # [E] mean across heads
        labels_np  = s.labels.numpy()
        pred_np    = s.pred.numpy()
        conf_np    = s.confidence.numpy()
        entropy_np = s.node_entropy                  # [N] normalized 0-1

        for nd in range(s.num_nodes):
            cls_idx   = int(labels_np[nd])
            base_rgba = PALETTE[cls_idx]
            base_hex  = rgba_to_hex(base_rgba)

            if mode == "Class Filter":
                if CORA_CLASSES[cls_idx] in selected_classes:
                    color, size, opacity = base_hex, 12, 1.0
                else:
                    color, size, opacity = "#444444", 4, 0.3

            elif mode == "Confidence Overlay":
                c = float(conf_np[nd])
                r = int((1 - c) * 0x88 + c * base_rgba[0] * 255)
                g = int((1 - c) * 0x88 + c * base_rgba[1] * 255)
                b = int((1 - c) * 0x88 + c * base_rgba[2] * 255)
                color, size, opacity = f"#{r:02x}{g:02x}{b:02x}", 8, 0.9

            elif mode == "Attention Concentration":
                conc = 1.0 - float(entropy_np[nd])
                r = int((1 - conc) * 0xaa + conc * base_rgba[0] * 255)
                g = int((1 - conc) * 0xaa + conc * base_rgba[1] * 255)
                b = int((1 - conc) * 0xaa + conc * base_rgba[2] * 255)
                color, size, opacity = f"#{r:02x}{g:02x}{b:02x}", 8, 0.9

            title = (f"Node {nd}<br>"
                     f"True: {CORA_CLASSES[cls_idx]}<br>"
                     f"Pred: {CORA_CLASSES[int(pred_np[nd])]}<br>"
                     f"Conf: {conf_np[nd]:.1%}")
            net.add_node(nd, label="", color=color, size=size,
                         title=title, opacity=opacity)

        max_alpha = float(alpha_np.max()) if alpha_np.max() > 0 else 1.0
        # Compute threshold once outside the edge loop (gotcha #8)
        threshold = np.percentile(alpha_np, 90)

        for k in range(len(src_np)):
            u, v = int(src_np[k]), int(dst_np[k])
            if u == v:   # skip self-loops in graph view
                continue
            w = float(alpha_np[k])

            if mode == "Class Filter":
                if CORA_CLASSES[int(labels_np[u])] not in selected_classes:
                    continue
                width = 0.5 + (w / max_alpha) * 4.0
                net.add_edge(u, v, width=width, color="#ffffff",
                             arrows="to", smooth={"type": "curvedCW"})
            else:
                if w < threshold:
                    continue
                width = 0.3 + (w / max_alpha) * 3.0
                net.add_edge(u, v, width=width, color="#ffffff44",
                             arrows="to", smooth={"type": "curvedCW"})

        return net

    # Cache rendered HTML per mode/selection to avoid re-generating on every
    # interaction (2708-node graphs are slow to build from scratch each time).
    if mode == "Class Filter":
        cache_key = "pyvis_html_Class Filter_" + "_".join(sorted(selected_classes))
    else:
        cache_key = f"pyvis_html_{mode}"

    if cache_key not in st.session_state:
        net = build_pyvis_graph(mode, selected_classes, st.session_state)
        st.session_state[cache_key] = net.generate_html()

    components.html(st.session_state[cache_key], height=720, scrolling=False)

    # --- Class color legend ---
    st.markdown("**Class legend:**")
    cols = st.columns(7)
    for i, (cls_name, col) in enumerate(zip(CORA_CLASSES, cols)):
        hex_color = rgba_to_hex(PALETTE[i])
        col.markdown(
            f"<div style='background:{hex_color};padding:4px 8px;"
            f"border-radius:4px;color:white;font-size:12px'>{cls_name}</div>",
            unsafe_allow_html=True
        )
