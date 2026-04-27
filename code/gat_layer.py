import torch
import torch.nn as nn
from torch_geometric.utils import softmax


class GATLayer(nn.Module):
    def __init__(self, in_features, out_features, num_heads, dropout=0.6, concat=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        self.dropout = dropout
        self.concat = concat

        self.W = nn.Parameter(torch.empty(num_heads, in_features, out_features))
        # Split attention vector a into src and dst halves to avoid explicit concat
        self.a_src = nn.Parameter(torch.empty(num_heads, out_features))
        self.a_dst = nn.Parameter(torch.empty(num_heads, out_features))

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.feat_dropout = nn.Dropout(p=dropout)
        self.attn_dropout = nn.Dropout(p=dropout)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.a_src.unsqueeze(0))
        nn.init.xavier_uniform_(self.a_dst.unsqueeze(0))

    def forward(self, x, edge_index):
        # x: [N, in_features]
        # edge_index: [2, E]
        src, dst = edge_index[0], edge_index[1]

        x = self.feat_dropout(x)

        # Linear transform: [N, H, out_features]
        h = torch.einsum('ni,hio->nho', x, self.W)

        # e_ij = LeakyReLU(a_src^T h_i + a_dst^T h_j) — equivalent to a^T [h_i || h_j]
        # [N, H, 1] → [N, H]
        e_src = (h * self.a_src).sum(dim=-1)  # [N, H]
        e_dst = (h * self.a_dst).sum(dim=-1)  # [N, H]

        # Raw scores per edge: [E, H]
        e = self.leaky_relu(e_src[src] + e_dst[dst])

        # Masked softmax over each node's neighborhood (indexed by source node)
        # softmax from PyG: normalises over index groups
        alpha = softmax(e, index=edge_index[0], num_nodes=x.size(0))  # [E, H]
        alpha = self.attn_dropout(alpha)

        # Aggregate: weighted sum of neighbour features
        # h[dst]: [E, H, out_features]; alpha: [E, H, 1]
        out = torch.zeros(x.size(0), self.num_heads, self.out_features, device=x.device)
        out.scatter_add_(
            0,
            src.view(-1, 1, 1).expand(-1, self.num_heads, self.out_features),
            alpha.unsqueeze(-1) * h[dst],
        )

        if self.concat:
            return out.view(x.size(0), self.num_heads * self.out_features)
        else:
            return out.mean(dim=1)
