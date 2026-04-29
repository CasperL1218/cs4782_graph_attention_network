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

def annotate_ego(ax, node, ei, alpha, labels, head=None):
    """
    Add neighbor index labels to an ego-graph axis drawn by _draw_ego().
    Recomputes the same circular layout _draw_ego uses so positions match.
    Edge attention scores are already rendered by _draw_ego; this adds only
    the neighbor index labels outside each node dot.
    """
    src_np = ei[0].numpy()
    dst_np = ei[1].numpy()
    a = alpha.numpy()
    w = a.mean(axis=1) if head is None else a[:, head]

    mask = src_np == node
    nbrs = dst_np[mask]
    ws = w[mask]

    if len(nbrs) == 0:
        return

    n_nbrs = len(nbrs)
    angles = np.linspace(0, 2 * np.pi, n_nbrs, endpoint=False)
    pos = {}
    for i, nbr in enumerate(nbrs):
        pos[int(nbr)] = np.array([np.cos(angles[i]), np.sin(angles[i])])

    # Neighbor index labels (outside the node dot)
    for nbr in nbrs:
        nbr = int(nbr)
        if nbr == node:
            continue
        p = pos[nbr]
        offset = p * 0.22  # push label slightly outward from dot
        ax.text(
            p[0] + offset[0], p[1] + offset[1],
            str(nbr),
            fontsize=6,
            ha="center", va="center",
            color="white",
            bbox=dict(boxstyle="round,pad=0.1", fc="#111111", ec="none", alpha=0.6),
            zorder=6,
        )


tab1, tab2 = st.tabs(["🔍 Node Explorer", "🌐 Graph View"])

# =========================================================================
# Tab 1 — Node Explorer
# =========================================================================
with tab1:
    if "selected_node" not in st.session_state:
        st.session_state.selected_node = 0

    st.sidebar.header("Node Explorer")

    src_np_tab1 = st.session_state.ei[0].numpy()
    all_degrees = np.array([int((src_np_tab1 == nd).sum()) for nd in range(st.session_state.num_nodes)])
    max_deg = int(all_degrees.max())

    # Slider renders before buttons so its value is available to button handlers
    deg_min, deg_max = st.sidebar.slider(
        "Neighbor degree filter (for random buttons)",
        min_value=1, max_value=max_deg,
        value=(1, max_deg), step=1
    )

    if st.sidebar.button("🎲 Random Node"):
        candidates = np.where((all_degrees >= deg_min) & (all_degrees <= deg_max))[0]
        st.session_state.selected_node = int(np.random.choice(candidates))

    if st.sidebar.button("❌ Random Misclassified"):
        s = st.session_state
        wrong = (s.pred != s.labels).nonzero(as_tuple=True)[0].numpy()
        deg_candidates = np.where((all_degrees >= deg_min) & (all_degrees <= deg_max))[0]
        filtered_wrong = np.intersect1d(wrong, deg_candidates)
        pool = filtered_wrong if len(filtered_wrong) > 0 else wrong
        st.session_state.selected_node = int(np.random.choice(pool))

    if st.sidebar.button("🔀 Random Mixed-Label"):
        s = st.session_state
        best_node, entropy, counts = find_mixed_label_node(
            s.data, s.model, s.ei, s.alpha1, s.labels
        )
        if best_node is not None and deg_min <= all_degrees[best_node] <= deg_max:
            st.session_state.selected_node = best_node

    node_input = st.sidebar.number_input(
        "Node index", min_value=0,
        max_value=st.session_state.num_nodes - 1,
        value=st.session_state.selected_node, step=1,
        key="node_input_widget"
    )
    st.session_state.selected_node = int(node_input)

    node = st.session_state.selected_node

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Ego-Graph (mean α across 8 heads)")
        fig, ax = plt.subplots(figsize=(5, 5))
        _draw_ego(ax, node, st.session_state.ei, st.session_state.alpha1,
                  st.session_state.labels)
        annotate_ego(ax, node, st.session_state.ei, st.session_state.alpha1,
                     st.session_state.labels)
        st.pyplot(fig)
        plt.close(fig)

    with col2:
        st.subheader("Per-Head Attention Breakdown")
        fig, axes = plt.subplots(2, 4, figsize=(10, 5))
        for h, ax in enumerate(axes.flatten()):
            _draw_ego(ax, node, st.session_state.ei,
                      st.session_state.alpha1, st.session_state.labels,
                      head=h, show_title=False, uniform_line=False)
            ax.set_title(f"Head {h+1}", fontsize=8)
            annotate_ego(ax, node, st.session_state.ei,
                         st.session_state.alpha1, st.session_state.labels,
                         head=h)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # --- Info table ---
    s = st.session_state
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

    st.metric("Node degree (incl. self-loop)", int(all_degrees[node]))

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
                       damping=0.4, overlap=0)

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
                    color, size, opacity = base_hex, 35, 1.0
                else:
                    color, size, opacity = "#444444", 6, 0.3
                title = (f"Node {nd}<br>"
                         f"True: {CORA_CLASSES[cls_idx]}<br>"
                         f"Pred: {CORA_CLASSES[int(pred_np[nd])]}<br>"
                         f"Conf: {conf_np[nd]:.1%}"
                         f"<br>Color: {color}")

            elif mode == "Confidence Overlay":
                c = float(conf_np[nd])
                c_boosted = c ** 0.5
                r = int((1 - c_boosted) * 0x88 + c_boosted * base_rgba[0] * 255)
                g = int((1 - c_boosted) * 0x88 + c_boosted * base_rgba[1] * 255)
                b = int((1 - c_boosted) * 0x88 + c_boosted * base_rgba[2] * 255)
                color, size, opacity = f"#{r:02x}{g:02x}{b:02x}", 20, 0.95
                title = (f"Node {nd}<br>"
                         f"True: {CORA_CLASSES[cls_idx]}<br>"
                         f"Pred: {CORA_CLASSES[int(pred_np[nd])]}<br>"
                         f"Conf: {conf_np[nd]:.1%}<br>"
                         f"Conf (boosted): {float(conf_np[nd])**0.5:.3f}<br>"
                         f"Color: {color}")

            elif mode == "Attention Concentration":
                conc = (1.0 - float(entropy_np[nd])) ** 0.5
                r = int((1 - conc) * 0xaa + conc * base_rgba[0] * 255)
                g = int((1 - conc) * 0xaa + conc * base_rgba[1] * 255)
                b = int((1 - conc) * 0xaa + conc * base_rgba[2] * 255)
                color, size, opacity = f"#{r:02x}{g:02x}{b:02x}", 20, 0.9
                title = (f"Node {nd}<br>"
                         f"True: {CORA_CLASSES[cls_idx]}<br>"
                         f"Pred: {CORA_CLASSES[int(pred_np[nd])]}<br>"
                         f"Raw entropy (norm): {float(entropy_np[nd]):.3f}<br>"
                         f"Conc (boosted): {(1-float(entropy_np[nd]))**0.5:.3f}<br>"
                         f"Color: {color}")

            net.add_node(nd, label="", color=color, size=size,
                         title=title, opacity=opacity)

        max_alpha = float(alpha_np.max()) if alpha_np.max() > 0 else 1.0

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
                             arrows="to", smooth={"type": "dynamic"})
            else:
                pass  # Modes 2 and 3: color is the signal, no edges drawn

        html = net.generate_html()
        freeze_js = """
<script type="text/javascript">
  document.addEventListener("DOMContentLoaded", function() {
    var checkNetwork = setInterval(function() {
      if (window.network) {
        clearInterval(checkNetwork);
        network.once("stabilized", function() {
          network.fit({ animation: { duration: 500, easingFunction: "easeInOutQuad" } });
          network.setOptions({ physics: { enabled: false } });
        });
      }
    }, 100);
  });
</script>
"""
        html = html.replace("</body>", freeze_js + "</body>")
        return html

    # Cache rendered HTML per mode/selection to avoid re-generating on every
    # interaction (2708-node graphs are slow to build from scratch each time).
    if mode == "Class Filter":
        cache_key = "v3_pyvis_html_Class Filter_" + "_".join(sorted(selected_classes))
    else:
        cache_key = f"v3_pyvis_html_{mode}"

    if cache_key not in st.session_state:
        st.session_state[cache_key] = build_pyvis_graph(mode, selected_classes, st.session_state)

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
