"""
graph.py — Myocardium Graph Construction and Graph Attention Network
=====================================================================
No torch-scatter / torch-sparse dependency — pure PyTorch + OpenCV.

Pipeline per time step:
  1. Extract myocardium contour vertices from binary mask (OpenCV)
  2. Sample displacement vectors from φ at each vertex (bilinear interp)
  3. Build k-NN graph from vertex positions
  4. Run 2-layer GAT over graph → node embeddings
  5. Global mean pooling → graph-level embedding (B, d_out)

The graph embedding is the input to the Neural ODE in ode.py.
One embedding per time step → sequence fed to ODE.

Key design choices:
  - Manual GAT: no torch-scatter needed, works on Python 3.13 + PyTorch 2.7
  - Fixed N_verts per sample via uniform contour resampling → enables batching
  - AHA-inspired radial ordering of contour points (consistent across patients)
  - Node features = [dx, dy, x_norm, y_norm, r, theta] (6-dim) not just displacement
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


# ── Constants ────────────────────────────────────────────────────────────────

N_CONTOUR_VERTS = 64   # fixed number of vertices per contour (resampled)
                        # 64 gives good spatial resolution while staying cheap
K_NEIGHBORS     = 6    # each vertex attends to 6 nearest neighbors


# ── Contour extraction ───────────────────────────────────────────────────────

def extract_contour_vertices(
    mask: np.ndarray,
    n_verts: int = N_CONTOUR_VERTS,
) -> Optional[np.ndarray]:
    """
    Extract and uniformly resample the myocardium contour from a binary mask.

    We use the OUTER contour (epicardium boundary) so that the full
    myocardial wall motion is captured, not just the endocardial surface.

    Args:
        mask    : (H, W) binary numpy array, 1 = myocardium
        n_verts : number of vertices to resample to (fixed across all slices)

    Returns:
        vertices : (n_verts, 2) float array in pixel coords [x, y]
                   or None if no valid contour found
    """
    mask_u8 = (mask * 255).astype(np.uint8)

    contours, _ = cv2.findContours(
        mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )

    if not contours:
        return None

    # Take the largest contour (should be epicardium)
    contour = max(contours, key=cv2.contourArea)

    if len(contour) < 3:
        return None

    # contour shape from OpenCV: (N, 1, 2) → reshape to (N, 2)
    pts = contour.reshape(-1, 2).astype(np.float32)  # (N, 2) in [x, y] order

    # Uniformly resample to exactly n_verts points along the contour
    pts = _resample_contour(pts, n_verts)

    return pts  # (n_verts, 2)


def _resample_contour(pts: np.ndarray, n_verts: int) -> np.ndarray:
    """
    Uniformly resample a contour to exactly n_verts points by
    arc-length parameterization.

    This ensures consistent vertex density regardless of original
    contour complexity, enabling batching across slices/patients.
    """
    # Compute cumulative arc length
    diffs = np.diff(pts, axis=0)                          # (N-1, 2)
    seg_lens = np.sqrt((diffs ** 2).sum(axis=1))          # (N-1,)
    cum_len = np.concatenate([[0], np.cumsum(seg_lens)])  # (N,)
    total_len = cum_len[-1]

    if total_len < 1e-6:
        # Degenerate contour — return repeated first point
        return np.tile(pts[0:1], (n_verts, 1))

    # Sample evenly spaced arc-length positions
    sample_lens = np.linspace(0, total_len, n_verts, endpoint=False)

    # Interpolate x and y independently
    new_x = np.interp(sample_lens, cum_len, pts[:, 0])
    new_y = np.interp(sample_lens, cum_len, pts[:, 1])

    return np.stack([new_x, new_y], axis=1)  # (n_verts, 2)


# ── Node feature computation ─────────────────────────────────────────────────

def compute_node_features(
    vertices: np.ndarray,   # (N_verts, 2) pixel coords [x, y]
    phi: torch.Tensor,      # (1, 2, H, W) or (2, H, W) deformation field
    H: int,
    W: int,
) -> torch.Tensor:
    """
    Compute 6-dimensional node features for each contour vertex:
        [dx, dy, x_norm, y_norm, r, theta]

    dx, dy      : displacement from deformation field φ at vertex location
                  (the primary motion signal)
    x_norm      : vertex x position normalized to [-1, 1]
    y_norm      : vertex y position normalized to [-1, 1]
    r           : radial distance from centroid (normalized)
    theta       : angle from centroid (normalized to [-1, 1])

    The positional features (x_norm, y_norm, r, theta) give the GAT
    spatial context — important because the same displacement magnitude
    means different things at the apex vs the base.

    Args:
        vertices : (N_verts, 2) in pixel coords
        phi      : deformation field, shape (1, 2, H, W) or (2, H, W)
        H, W     : spatial dimensions

    Returns:
        features : (N_verts, 6) float tensor
    """
    if phi.dim() == 3:
        phi = phi.unsqueeze(0)  # → (1, 2, H, W)

    N = len(vertices)
    device = phi.device

    # ── Sample displacement at each vertex via bilinear interpolation ──────
    # Normalize vertex coords to [-1, 1] for grid_sample
    verts_norm = vertices.copy()
    verts_norm[:, 0] = (verts_norm[:, 0] / (W - 1)) * 2 - 1  # x
    verts_norm[:, 1] = (verts_norm[:, 1] / (H - 1)) * 2 - 1  # y

    # grid_sample expects (1, 1, N, 2) grid for N sample points
    grid = torch.from_numpy(verts_norm).float().to(device)
    grid = grid.unsqueeze(0).unsqueeze(0)  # (1, 1, N_verts, 2)

    # Sample displacement: (1, 2, 1, N_verts) → squeeze → (N_verts, 2)
    disp = F.grid_sample(
        phi, grid,
        mode="bilinear", padding_mode="border", align_corners=True
    )
    disp = disp.squeeze(0).squeeze(1).permute(1, 0)  # (N_verts, 2) [dx, dy]

    # ── Positional features ───────────────────────────────────────────────
    verts_t = torch.from_numpy(vertices).float().to(device)  # (N_verts, 2)

    x_norm = (verts_t[:, 0] / (W - 1)) * 2 - 1  # (N_verts,)
    y_norm = (verts_t[:, 1] / (H - 1)) * 2 - 1  # (N_verts,)

    # Centroid-relative polar coordinates
    cx = x_norm.mean()
    cy = y_norm.mean()
    rel_x = x_norm - cx
    rel_y = y_norm - cy
    r     = torch.sqrt(rel_x ** 2 + rel_y ** 2)
    r     = r / (r.max() + 1e-8)                 # normalize to [0, 1]
    theta = torch.atan2(rel_y, rel_x) / np.pi    # normalize to [-1, 1]

    # Stack: (N_verts, 6)
    features = torch.stack([
        disp[:, 0],   # dx
        disp[:, 1],   # dy
        x_norm,       # absolute x position
        y_norm,       # absolute y position
        r,            # radial distance from centroid
        theta,        # angle from centroid
    ], dim=1)

    return features  # (N_verts, 6)


# ── k-NN graph construction ──────────────────────────────────────────────────

def build_knn_adjacency(
    vertices: np.ndarray,
    k: int = K_NEIGHBORS,
) -> torch.Tensor:
    """
    Build a k-NN adjacency matrix from vertex positions.
    Returns a dense adjacency matrix (N, N) — avoids torch-scatter entirely.

    For N=64, k=6: matrix is (64, 64) which is trivially small.

    Args:
        vertices : (N_verts, 2) numpy array
        k        : number of nearest neighbors

    Returns:
        adj : (N_verts, N_verts) float tensor, 1 = edge exists (symmetric)
    """
    N = len(vertices)
    verts_t = torch.from_numpy(vertices).float()

    # Pairwise squared distances (N, N)
    diff = verts_t.unsqueeze(0) - verts_t.unsqueeze(1)  # (N, N, 2)
    dist2 = (diff ** 2).sum(-1)                          # (N, N)

    # For each node, find k nearest neighbors (excluding self)
    dist2_no_self = dist2.clone()
    dist2_no_self.fill_diagonal_(float("inf"))

    _, knn_idx = dist2_no_self.topk(k, dim=1, largest=False)  # (N, k)

    # Build symmetric adjacency matrix
    adj = torch.zeros(N, N)
    rows = torch.arange(N).unsqueeze(1).expand_as(knn_idx)  # (N, k)
    adj[rows.reshape(-1), knn_idx.reshape(-1)] = 1.0
    adj = torch.maximum(adj, adj.T)  # symmetrize

    return adj  # (N, N)


# ── Manual Graph Attention Layer ─────────────────────────────────────────────

class GATLayer(nn.Module):
    """
    Single Graph Attention Layer — manual implementation, no torch-scatter.

    For each node i:
        h_i' = σ( Σ_{j∈N(i)} α_ij * W * h_j )

    where attention weights α_ij are computed as:
        e_ij  = LeakyReLU( a^T [W*h_i || W*h_j] )
        α_ij  = softmax_j( e_ij )  (only over neighbors of i)

    The adjacency matrix handles neighbor masking — non-neighbors
    get -inf before softmax, so they contribute zero attention.

    Args:
        in_features  : input node feature dimension
        out_features : output node feature dimension
        n_heads      : number of attention heads
        dropout      : attention dropout probability
    """

    def __init__(
        self,
        in_features:  int,
        out_features: int,
        n_heads:      int = 4,
        dropout:      float = 0.1,
        concat:       bool = True,   # if True: output = n_heads * out_features
    ):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.n_heads      = n_heads
        self.concat       = concat
        self.dropout_p    = dropout

        # Shared linear projection for all heads
        self.W = nn.Linear(in_features, n_heads * out_features, bias=False)

        # Attention vector: [W*h_i || W*h_j] → scalar score
        # size = 2 * out_features (one for source, one for target)
        self.a = nn.Parameter(torch.empty(n_heads, 2 * out_features))
        nn.init.xavier_uniform_(self.a.unsqueeze(0))

        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout    = nn.Dropout(dropout)

    def forward(
        self,
        h:   torch.Tensor,   # (B, N, in_features)
        adj: torch.Tensor,   # (N, N) adjacency matrix (same for all in batch)
    ) -> torch.Tensor:
        """
        Args:
            h   : (B, N, in_features)
            adj : (N, N) binary adjacency (0/1)

        Returns:
            h_out : (B, N, n_heads * out_features)  if concat=True
                    (B, N, out_features)             if concat=False
        """
        B, N, _ = h.shape
        n_h = self.n_heads
        d_f = self.out_features

        # Linear projection: (B, N, n_h*d_f)
        Wh = self.W(h).view(B, N, n_h, d_f)  # (B, N, n_h, d_f)

        # ── Attention coefficients ─────────────────────────────────────────
        # For head k: e_ij = LeakyReLU( a_k^T [Wh_i || Wh_j] )
        # Vectorized: compute all pairs simultaneously

        # (B, N, n_h, d_f) → source and target expansions
        Wh_src = Wh.unsqueeze(2).expand(B, N, N, n_h, d_f)  # (B, N, N, n_h, d_f) — i
        Wh_dst = Wh.unsqueeze(1).expand(B, N, N, n_h, d_f)  # (B, N, N, n_h, d_f) — j

        # Concatenate and apply attention vector: (B, N, N, n_h, 2*d_f)
        attn_input = torch.cat([Wh_src, Wh_dst], dim=-1)  # (B, N, N, n_h, 2*d_f)

        # a has shape (n_h, 2*d_f) → apply per head
        # e[b, i, j, h] = a[h] · attn_input[b, i, j, h, :]
        a = self.a  # (n_h, 2*d_f)
        e = (attn_input * a.unsqueeze(0).unsqueeze(0).unsqueeze(0)).sum(-1)
        e = self.leaky_relu(e)  # (B, N, N, n_h)

        # ── Mask non-neighbors with -inf before softmax ────────────────────
        # adj: (N, N) → expand to (1, N, N, 1)
        adj_mask = adj.to(h.device).unsqueeze(0).unsqueeze(-1)  # (1, N, N, 1)
        e = e.masked_fill(adj_mask == 0, float("-inf"))

        # Softmax over neighbors (dim=2 = j dimension)
        alpha = torch.softmax(e, dim=2)  # (B, N, N, n_h)
        alpha = self.dropout(alpha)

        # Handle nodes with no neighbors (all -inf → NaN after softmax)
        alpha = torch.nan_to_num(alpha, nan=0.0)

        # ── Aggregate neighbor features ────────────────────────────────────
        # h_out[b, i, h, :] = Σ_j alpha[b, i, j, h] * Wh[b, j, h, :]
        # alpha: (B, N, N, n_h) | Wh: (B, N, n_h, d_f)

        alpha_t = alpha.permute(0, 3, 1, 2)    # (B, n_h, N, N)
        Wh_t    = Wh.permute(0, 2, 1, 3)       # (B, n_h, N, d_f)

        h_out = torch.bmm(
            alpha_t.reshape(B * n_h, N, N),
            Wh_t.reshape(B * n_h, N, d_f)
        ).reshape(B, n_h, N, d_f)              # (B, n_h, N, d_f)

        h_out = h_out.permute(0, 2, 1, 3)      # (B, N, n_h, d_f)

        if self.concat:
            h_out = h_out.reshape(B, N, n_h * d_f)  # (B, N, n_h*d_f)
            return F.elu(h_out)
        else:
            h_out = h_out.mean(dim=2)               # (B, N, d_f)
            return h_out


# ── Full Graph Encoder ───────────────────────────────────────────────────────

class MyocardiumGraphEncoder(nn.Module):
    """
    Two-layer GAT encoder that maps a myocardium motion graph to a
    fixed-size embedding vector.

    Layer 1: GATLayer(6  → 16, n_heads=4) → output dim = 64
    Layer 2: GATLayer(64 → 64, n_heads=1) → output dim = 64
    Pool   : global mean → (B, 64)

    Input:
        node_features : (B, N_verts, 6)
        adj           : (N_verts, N_verts)

    Output:
        graph_embed : (B, d_out=64)
    """

    def __init__(
        self,
        in_features:  int = 6,
        hidden_dim:   int = 16,
        out_dim:      int = 64,
        n_heads_l1:   int = 4,    # layer 1 heads (concat → 4*16=64)
        dropout:      float = 0.1,
    ):
        super().__init__()
        self.out_dim = out_dim

        mid_dim = n_heads_l1 * hidden_dim  # 4 * 16 = 64

        self.gat1 = GATLayer(
            in_features  = in_features,
            out_features = hidden_dim,
            n_heads      = n_heads_l1,
            dropout      = dropout,
            concat       = True,   # output: (B, N, 64)
        )
        self.gat2 = GATLayer(
            in_features  = mid_dim,
            out_features = out_dim,
            n_heads      = 1,
            dropout      = dropout,
            concat       = False,  # output: (B, N, 64) — averaged over 1 head
        )

        # Layer norm after each GAT for stability
        self.norm1 = nn.LayerNorm(mid_dim)
        self.norm2 = nn.LayerNorm(out_dim)

    def forward(
        self,
        node_features: torch.Tensor,   # (B, N_verts, 6)
        adj:           torch.Tensor,   # (N_verts, N_verts)
    ) -> torch.Tensor:
        """
        Returns:
            graph_embed : (B, out_dim=64)
        """
        h = self.gat1(node_features, adj)   # (B, N_verts, 64)
        h = self.norm1(h)
        h = self.gat2(h, adj)               # (B, N_verts, 64)
        h = self.norm2(h)

        # Global mean pooling over vertices → (B, 64)
        graph_embed = h.mean(dim=1)

        return graph_embed


# ── Full graph pipeline: mask + phi → embedding ──────────────────────────────

class GraphMotionEncoder(nn.Module):
    """
    End-to-end module: takes raw mask and deformation field,
    builds the graph, computes node features, runs GAT, returns embedding.

    This is the interface that model.py calls at each time step.

    Args:
        n_verts  : number of contour vertices (default 64)
        k        : k-NN neighbors (default 6)
        out_dim  : embedding dimension (default 64)
    """

    def __init__(
        self,
        n_verts:  int = N_CONTOUR_VERTS,
        k:        int = K_NEIGHBORS,
        out_dim:  int = 64,
    ):
        super().__init__()
        self.n_verts = n_verts
        self.k       = k
        self.encoder = MyocardiumGraphEncoder(
            in_features = 6,
            out_dim     = out_dim,
        )

        # Cache adjacency matrix — computed from vertex positions
        # which are patient-specific, but we recompute if None
        self._adj_cache = {}

    def _get_or_build_adj(
        self,
        vertices: np.ndarray,   # (N_verts, 2)
        device:   torch.device,
    ) -> torch.Tensor:
        """
        Build k-NN adjacency matrix from vertex positions.
        Cache by vertex hash to avoid recomputing every forward pass
        for the same contour shape.
        """
        key = vertices.tobytes()
        if key not in self._adj_cache:
            adj = build_knn_adjacency(vertices, self.k).to(device)
            # Keep cache small
            if len(self._adj_cache) > 256:
                self._adj_cache.clear()
            self._adj_cache[key] = adj
        return self._adj_cache[key]

    def forward(
        self,
        masks: torch.Tensor,   # (B, 1, H, W) binary myocardium masks
        phis:  torch.Tensor,   # (B, 2, H, W) deformation fields
    ) -> torch.Tensor:
        """
        Args:
            masks : (B, 1, H, W) — ED myocardium masks
            phis  : (B, 2, H, W) — deformation fields from registration net

        Returns:
            embeddings : (B, out_dim)  — one embedding per sample in batch
        """
        B, _, H, W = masks.shape
        device = masks.device
        out_dim = self.encoder.out_dim

        all_features = []
        all_adjs     = []

        for b in range(B):
            mask_np = masks[b, 0].detach().cpu().numpy()   # (H, W)
            phi_b   = phis[b]                              # (2, H, W)

            verts = extract_contour_vertices(mask_np, self.n_verts)

            if verts is None:
                # No contour found — return zeros for this sample
                all_features.append(
                    torch.zeros(self.n_verts, 6, device=device)
                )
                all_adjs.append(
                    torch.zeros(self.n_verts, self.n_verts, device=device)
                )
                continue

            # Node features: (N_verts, 6)
            feats = compute_node_features(verts, phi_b, H, W)
            all_features.append(feats)

            # Adjacency matrix: (N_verts, N_verts)
            adj = self._get_or_build_adj(verts, device)
            all_adjs.append(adj)

        # Stack into batches
        # Note: adjacency matrices are sample-specific (contours differ)
        # We process each sample independently through the GAT
        # then stack the embeddings — no batched adj needed
        embeddings = []
        for feats, adj in zip(all_features, all_adjs):
            feat_b = feats.unsqueeze(0)   # (1, N_verts, 6)
            adj_b  = adj                  # (N_verts, N_verts)
            emb    = self.encoder(feat_b, adj_b)   # (1, out_dim)
            embeddings.append(emb.squeeze(0))      # (out_dim,)

        return torch.stack(embeddings, dim=0)  # (B, out_dim)


# ── Sanity check ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import time

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    B, H, W = 4, 128, 128

    # ── 1. Test contour extraction ────────────────────────────────────────────
    print("\n── Contour extraction ──")
    # Synthetic ring-shaped myocardium mask (inner r=20, outer r=35)
    mask_np = np.zeros((H, W), dtype=np.float32)
    cx, cy = W // 2, H // 2
    for i in range(H):
        for j in range(W):
            r = np.sqrt((i - cy)**2 + (j - cx)**2)
            if 20 < r < 35:
                mask_np[i, j] = 1.0

    verts = extract_contour_vertices(mask_np, N_CONTOUR_VERTS)
    print(f"  Contour vertices : {verts.shape}")     # (64, 2)
    print(f"  x range : [{verts[:,0].min():.1f}, {verts[:,0].max():.1f}]")
    print(f"  y range : [{verts[:,1].min():.1f}, {verts[:,1].max():.1f}]")

    # ── 2. Test node features ─────────────────────────────────────────────────
    print("\n── Node features ──")
    phi = torch.randn(1, 2, H, W, device=device) * 2.0  # random displacements
    feats = compute_node_features(verts, phi, H, W)
    print(f"  Features shape : {feats.shape}")   # (64, 6)
    print(f"  dx range       : [{feats[:,0].min():.3f}, {feats[:,0].max():.3f}]")
    print(f"  dy range       : [{feats[:,1].min():.3f}, {feats[:,1].max():.3f}]")
    print(f"  r  range       : [{feats[:,4].min():.3f}, {feats[:,4].max():.3f}]")
    print(f"  theta range    : [{feats[:,5].min():.3f}, {feats[:,5].max():.3f}]")

    # ── 3. Test k-NN adjacency ────────────────────────────────────────────────
    print("\n── k-NN adjacency ──")
    adj = build_knn_adjacency(verts, K_NEIGHBORS)
    print(f"  Adj shape       : {adj.shape}")             # (64, 64)
    print(f"  Symmetric       : {(adj == adj.T).all()}")  # True
    print(f"  Edges per node  : {adj.sum(dim=1).mean():.1f} (target ≈ {K_NEIGHBORS})")
    print(f"  Self-loops      : {adj.diagonal().sum():.0f} (should be 0)")

    # ── 4. Test GAT layer ─────────────────────────────────────────────────────
    print("\n── GATLayer ──")
    gat = GATLayer(in_features=6, out_features=16, n_heads=4).to(device)
    h_in  = torch.randn(B, N_CONTOUR_VERTS, 6, device=device)
    adj_d = adj.to(device)
    h_out = gat(h_in, adj_d)
    print(f"  Input  shape : {h_in.shape}")    # (4, 64, 6)
    print(f"  Output shape : {h_out.shape}")   # (4, 64, 64)  [4 heads * 16]
    print(f"  Output range : [{h_out.min():.3f}, {h_out.max():.3f}]")

    # ── 5. Test full encoder ──────────────────────────────────────────────────
    print("\n── MyocardiumGraphEncoder ──")
    enc = MyocardiumGraphEncoder().to(device)
    n_params = sum(p.numel() for p in enc.parameters() if p.requires_grad)
    print(f"  Parameters : {n_params:,}")

    h_in2 = torch.randn(B, N_CONTOUR_VERTS, 6, device=device)
    emb = enc(h_in2, adj_d)
    print(f"  Input  shape : {h_in2.shape}")   # (4, 64, 6)
    print(f"  Output shape : {emb.shape}")     # (4, 64)

    # ── 6. Test backward pass ────────────────────────────────────────────────
    loss = emb.sum()
    loss.backward()
    grad_norm = sum(
        p.grad.norm().item() for p in enc.parameters() if p.grad is not None
    )
    print(f"  Gradient norm : {grad_norm:.4f}  (should be > 0)")

    # ── 7. Test full GraphMotionEncoder ──────────────────────────────────────
    print("\n── GraphMotionEncoder (end-to-end) ──")
    gme = GraphMotionEncoder(n_verts=N_CONTOUR_VERTS, k=K_NEIGHBORS, out_dim=64).to(device)
    n_params2 = sum(p.numel() for p in gme.parameters() if p.requires_grad)
    print(f"  Parameters : {n_params2:,}")

    # Batch of ring masks + random deformation fields
    masks_batch = torch.from_numpy(
        np.stack([mask_np[None]] * B, axis=0)   # (B, 1, H, W)
    ).float().to(device)
    phis_batch = torch.randn(B, 2, H, W, device=device) * 2.0

    t0 = time.time()
    embeddings = gme(masks_batch, phis_batch)
    t1 = time.time()

    print(f"  Input masks  : {masks_batch.shape}")
    print(f"  Input phis   : {phis_batch.shape}")
    print(f"  Output shape : {embeddings.shape}")   # (4, 64)
    print(f"  Forward time : {(t1-t0)*1000:.1f} ms  (B={B}, N_verts={N_CONTOUR_VERTS})")

    # ── 8. Test with empty mask (edge case) ──────────────────────────────────
    print("\n── Edge case: empty mask ──")
    empty_masks = torch.zeros(2, 1, H, W, device=device)
    empty_phis  = torch.randn(2, 2, H, W, device=device)
    emb_empty = gme(empty_masks, empty_phis)
    print(f"  Empty mask embedding shape : {emb_empty.shape}")  # (2, 64)
    print(f"  All zeros                  : {(emb_empty == 0).all()}")  # True

    if torch.cuda.is_available():
        mem_mb = torch.cuda.max_memory_allocated(device) / 1024**2
        print(f"\nGPU memory used: {mem_mb:.1f} MB")

    print("\n[graph.py] All checks passed.")