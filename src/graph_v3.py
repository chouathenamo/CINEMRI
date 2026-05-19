"""
graph.py — Myocardium Graph Construction and Graph Attention Network  (v3)
==========================================================================
Changes from v2:
  - Dropout raised from 0.1 → 0.3 in GATLayer and MyocardiumGraphEncoder
    (key regularisation fix to combat overfitting on 100-patient dataset)

Everything else is identical to v2.
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


# ── Constants ────────────────────────────────────────────────────────────────

N_CONTOUR_VERTS = 64
K_NEIGHBORS     = 6


# ── Contour extraction ───────────────────────────────────────────────────────

def extract_contour_vertices(
    mask: np.ndarray,
    n_verts: int = N_CONTOUR_VERTS,
) -> Optional[np.ndarray]:
    mask_u8  = (mask * 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    if len(contour) < 3:
        return None
    pts = contour.reshape(-1, 2).astype(np.float32)
    return _resample_contour(pts, n_verts)


def _resample_contour(pts: np.ndarray, n_verts: int) -> np.ndarray:
    diffs      = np.diff(pts, axis=0)
    seg_lens   = np.sqrt((diffs ** 2).sum(axis=1))
    cum_len    = np.concatenate([[0], np.cumsum(seg_lens)])
    total_len  = cum_len[-1]
    if total_len < 1e-6:
        return np.tile(pts[0:1], (n_verts, 1))
    sample_lens = np.linspace(0, total_len, n_verts, endpoint=False)
    new_x = np.interp(sample_lens, cum_len, pts[:, 0])
    new_y = np.interp(sample_lens, cum_len, pts[:, 1])
    return np.stack([new_x, new_y], axis=1)


# ── Node features ────────────────────────────────────────────────────────────

def compute_node_features(
    vertices: np.ndarray,
    phi:      torch.Tensor,
    H:        int,
    W:        int,
) -> torch.Tensor:
    if phi.dim() == 3:
        phi = phi.unsqueeze(0)

    N      = len(vertices)
    device = phi.device

    verts_norm = vertices.copy()
    verts_norm[:, 0] = (verts_norm[:, 0] / (W - 1)) * 2 - 1
    verts_norm[:, 1] = (verts_norm[:, 1] / (H - 1)) * 2 - 1

    grid = torch.from_numpy(verts_norm).float().to(device)
    grid = grid.unsqueeze(0).unsqueeze(0)

    disp = F.grid_sample(phi, grid, mode="bilinear",
                         padding_mode="border", align_corners=True)
    disp = disp.squeeze(0).squeeze(1).permute(1, 0)  # (N, 2)

    verts_t = torch.from_numpy(vertices).float().to(device)
    x_norm  = (verts_t[:, 0] / (W - 1)) * 2 - 1
    y_norm  = (verts_t[:, 1] / (H - 1)) * 2 - 1

    cx    = x_norm.mean()
    cy    = y_norm.mean()
    rel_x = x_norm - cx
    rel_y = y_norm - cy
    r     = torch.sqrt(rel_x ** 2 + rel_y ** 2)
    r     = r / (r.max() + 1e-8)
    theta = torch.atan2(rel_y, rel_x) / np.pi

    return torch.stack([disp[:, 0], disp[:, 1], x_norm, y_norm, r, theta], dim=1)


# ── k-NN adjacency ───────────────────────────────────────────────────────────

def build_knn_adjacency(vertices: np.ndarray, k: int = K_NEIGHBORS) -> torch.Tensor:
    N       = len(vertices)
    verts_t = torch.from_numpy(vertices).float()
    diff    = verts_t.unsqueeze(0) - verts_t.unsqueeze(1)
    dist2   = (diff ** 2).sum(-1)
    d2_ns   = dist2.clone()
    d2_ns.fill_diagonal_(float("inf"))
    _, knn_idx = d2_ns.topk(k, dim=1, largest=False)
    adj  = torch.zeros(N, N)
    rows = torch.arange(N).unsqueeze(1).expand_as(knn_idx)
    adj[rows.reshape(-1), knn_idx.reshape(-1)] = 1.0
    adj  = torch.maximum(adj, adj.T)
    return adj


# ── Graph Attention Layer ─────────────────────────────────────────────────────

class GATLayer(nn.Module):
    """
    Single Graph Attention Layer.
    v3 change: default dropout raised 0.1 → 0.3.
    """

    def __init__(
        self,
        in_features:  int,
        out_features: int,
        n_heads:      int   = 4,
        dropout:      float = 0.3,   # v3: was 0.1
        concat:       bool  = True,
    ):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.n_heads      = n_heads
        self.concat       = concat
        self.dropout_p    = dropout

        self.W = nn.Linear(in_features, n_heads * out_features, bias=False)
        self.a = nn.Parameter(torch.empty(n_heads, 2 * out_features))
        nn.init.xavier_uniform_(self.a.unsqueeze(0))

        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout    = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        B, N, _ = h.shape
        n_h     = self.n_heads
        d_f     = self.out_features

        Wh     = self.W(h).view(B, N, n_h, d_f)
        Wh_src = Wh.unsqueeze(2).expand(B, N, N, n_h, d_f)
        Wh_dst = Wh.unsqueeze(1).expand(B, N, N, n_h, d_f)

        attn_input = torch.cat([Wh_src, Wh_dst], dim=-1)
        a  = self.a
        e  = (attn_input * a.unsqueeze(0).unsqueeze(0).unsqueeze(0)).sum(-1)
        e  = self.leaky_relu(e)

        adj_mask = adj.to(h.device).unsqueeze(0).unsqueeze(-1)
        e        = e.masked_fill(adj_mask == 0, float("-inf"))
        alpha    = torch.softmax(e, dim=2)
        alpha    = self.dropout(alpha)
        alpha    = torch.nan_to_num(alpha, nan=0.0)

        alpha_t = alpha.permute(0, 3, 1, 2)
        Wh_t    = Wh.permute(0, 2, 1, 3)
        h_out   = torch.bmm(
            alpha_t.reshape(B * n_h, N, N),
            Wh_t.reshape(B * n_h, N, d_f)
        ).reshape(B, n_h, N, d_f).permute(0, 2, 1, 3)

        if self.concat:
            return F.elu(h_out.reshape(B, N, n_h * d_f))
        else:
            return h_out.mean(dim=2)


# ── Full Graph Encoder ────────────────────────────────────────────────────────

class MyocardiumGraphEncoder(nn.Module):
    """
    Two-layer GAT encoder.
    v3 change: dropout 0.1 → 0.3.
    """

    def __init__(
        self,
        in_features: int   = 6,
        hidden_dim:  int   = 16,
        out_dim:     int   = 64,
        n_heads_l1:  int   = 4,
        dropout:     float = 0.3,   # v3: was 0.1
    ):
        super().__init__()
        self.out_dim = out_dim
        mid_dim      = n_heads_l1 * hidden_dim

        self.gat1  = GATLayer(in_features, hidden_dim, n_heads=n_heads_l1,
                               dropout=dropout, concat=True)
        self.gat2  = GATLayer(mid_dim,     out_dim,    n_heads=1,
                               dropout=dropout, concat=False)
        self.norm1 = nn.LayerNorm(mid_dim)
        self.norm2 = nn.LayerNorm(out_dim)

    def forward(self, node_features: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        h = self.norm1(self.gat1(node_features, adj))
        h = self.norm2(self.gat2(h, adj))
        return h.mean(dim=1)  # global mean pool → (B, out_dim)


# ── End-to-end graph pipeline ────────────────────────────────────────────────

class GraphMotionEncoder(nn.Module):
    """mask + phi → graph embedding (B, out_dim)."""

    def __init__(self, n_verts: int = N_CONTOUR_VERTS, k: int = K_NEIGHBORS, out_dim: int = 64):
        super().__init__()
        self.n_verts = n_verts
        self.k       = k
        self.encoder = MyocardiumGraphEncoder(in_features=6, out_dim=out_dim)
        self._adj_cache = {}

    def _get_or_build_adj(self, vertices: np.ndarray, device: torch.device) -> torch.Tensor:
        key = vertices.tobytes()
        if key not in self._adj_cache:
            adj = build_knn_adjacency(vertices, self.k).to(device)
            if len(self._adj_cache) > 256:
                self._adj_cache.clear()
            self._adj_cache[key] = adj
        return self._adj_cache[key]

    def forward(self, masks: torch.Tensor, phis: torch.Tensor) -> torch.Tensor:
        B, _, H, W = masks.shape
        device     = masks.device
        out_dim    = self.encoder.out_dim

        all_features, all_adjs = [], []

        for b in range(B):
            mask_np = masks[b, 0].detach().cpu().numpy()
            phi_b   = phis[b]
            verts   = extract_contour_vertices(mask_np, self.n_verts)

            if verts is None:
                all_features.append(torch.zeros(self.n_verts, 6, device=device))
                all_adjs.append(torch.zeros(self.n_verts, self.n_verts, device=device))
                continue

            all_features.append(compute_node_features(verts, phi_b, H, W))
            all_adjs.append(self._get_or_build_adj(verts, device))

        embeddings = []
        for feats, adj in zip(all_features, all_adjs):
            emb = self.encoder(feats.unsqueeze(0), adj)
            embeddings.append(emb.squeeze(0))

        return torch.stack(embeddings, dim=0)  # (B, out_dim)


# ── Sanity check ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    B, H, W = 4, 128, 128

    mask_np = np.zeros((H, W), dtype=np.float32)
    cx, cy  = W // 2, H // 2
    for i in range(H):
        for j in range(W):
            r = np.sqrt((i - cy)**2 + (j - cx)**2)
            if 20 < r < 35:
                mask_np[i, j] = 1.0

    gme    = GraphMotionEncoder(n_verts=N_CONTOUR_VERTS, k=K_NEIGHBORS, out_dim=64).to(device)
    masks  = torch.from_numpy(np.stack([mask_np[None]] * B)).float().to(device)
    phis   = torch.randn(B, 2, H, W, device=device) * 2.0
    embeds = gme(masks, phis)
    print(f"Output: {embeds.shape}  (should be ({B}, 64))")

    # Backward
    embeds.sum().backward()
    gn = sum(p.grad.norm().item() for p in gme.parameters() if p.grad is not None)
    print(f"Grad norm: {gn:.4f}  (> 0 = OK)")
    print("[graph_v3.py] All checks passed.")
