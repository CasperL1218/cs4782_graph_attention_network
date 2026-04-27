import torch
import pytest
from gat_layer import GATLayer

N, F_in, F_out, H = 5, 8, 4, 3

# triangle: 0→1→2→0, plus self-loops on all nodes
EDGES = torch.tensor([
    [0, 1, 2, 0, 1, 2, 3, 4],
    [1, 2, 0, 0, 1, 2, 3, 4],
])


def make_layer(concat=True, **kw):
    return GATLayer(F_in, F_out, H, dropout=0.0, concat=concat, **kw)


def test_output_shape_concat():
    layer = make_layer(concat=True)
    x = torch.randn(N, F_in)
    out = layer(x, EDGES)
    assert out.shape == (N, H * F_out)


def test_output_shape_mean():
    layer = make_layer(concat=False)
    x = torch.randn(N, F_in)
    out = layer(x, EDGES)
    assert out.shape == (N, F_out)


def test_attention_sums_to_one():
    """Alpha coefficients for each node's neighbourhood must sum to 1."""
    from torch_geometric.utils import softmax
    layer = make_layer(concat=True)
    layer.eval()
    x = torch.randn(N, F_in)

    with torch.no_grad():
        src, dst = EDGES[0], EDGES[1]
        h = torch.einsum('ni,hio->nho', x, layer.W)
        e_src = (h * layer.a_src).sum(-1)
        e_dst = (h * layer.a_dst).sum(-1)
        e = layer.leaky_relu(e_src[src] + e_dst[dst])
        alpha = softmax(e, index=EDGES[0], num_nodes=N)  # [E, H]

    # sum alpha over edges per source node, per head
    sums = torch.zeros(N, H).scatter_add(0, EDGES[0].unsqueeze(1).expand(-1, H), alpha)
    # nodes with at least one edge should sum to ~1
    assert torch.allclose(sums[sums > 0], torch.ones_like(sums[sums > 0]), atol=1e-5)


def test_no_nan_or_inf():
    layer = make_layer(concat=True)
    x = torch.randn(N, F_in)
    out = layer(x, EDGES)
    assert torch.isfinite(out).all()


def test_isolated_node_is_zero():
    """A node with no incoming edges should produce an all-zero output."""
    layer = make_layer(concat=True)
    layer.eval()
    # edges only among nodes 0,1,2 — node 3 and 4 are isolated
    edges = torch.tensor([[0, 1, 2], [1, 2, 0]])
    x = torch.randn(N, F_in)
    with torch.no_grad():
        out = layer(x, edges)
    assert torch.all(out[3] == 0) and torch.all(out[4] == 0)


def test_gradients_flow():
    layer = make_layer(concat=True)
    x = torch.randn(N, F_in, requires_grad=True)
    out = layer(x, EDGES)
    out.sum().backward()
    assert x.grad is not None
    assert all(p.grad is not None for p in layer.parameters())


def test_single_head_equiv_no_concat():
    """With 1 head, concat and mean should give the same result."""
    layer_c = GATLayer(F_in, F_out, num_heads=1, dropout=0.0, concat=True)
    layer_m = GATLayer(F_in, F_out, num_heads=1, dropout=0.0, concat=False)
    # share weights
    layer_m.load_state_dict(layer_c.state_dict())
    x = torch.randn(N, F_in)
    with torch.no_grad():
        assert torch.allclose(layer_c(x, EDGES), layer_m(x, EDGES), atol=1e-6)
