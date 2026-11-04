"""
Dynamic EdgeConv GNN Model (ParticleNet-lite) for Jet Tagging
============================================================
References:
-----------
1. H. Qu, L. Gouskos, "Jet Tagging via Particle Clouds", Phys. Rev. D 101, 056019 (2020) [arXiv:1902.08570].
2. Y. Wang et al., "Dynamic Graph CNN for Learning on Point Clouds", ACM TOG 2019 [arXiv:1801.07829].
3. J. Shlomi, P. Battaglia, J.-R. Vlimant, arXiv:2007.13681 (Sec. 3.1 & 4.3).

Model Concept:
--------------
Rather than treating particles independently (Deep Sets) or assuming a fixed static graph,
Dynamic EdgeConv constructs a k-Nearest Neighbors (k-NN) graph dynamically.

For each particle i and neighbor j in N_k(i):
  1. Edge feature: e_{ij} = Theta([x_j - x_i, x_i])
  2. Aggregation:  x_i' = max_{j in N_k(i)} e_{ij}

This captures local relational substructures, collinear clusterings, and multi-particle correlations.
The implementation uses pure, vectorized PyTorch (no brittle external C++ dependencies).
"""

from typing import Optional, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_knn_indices(
    coords: torch.Tensor,
    k: int,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Computes indices of k-nearest neighbors in coordinate or latent feature space.

    Args:
        coords: Tensor of shape (B, N, D_coord)
        k: Number of nearest neighbors to select
        mask: Tensor of shape (B, N) with 1.0 for valid particles, 0.0 for padding.

    Returns:
        knn_indices: Tensor of shape (B, N, k) with neighbor indices.
    """
    B, N, _ = coords.shape
    k = min(k, N)

    # Pairwise squared Euclidean distance: ||p_i - p_j||^2 = ||p_i||^2 + ||p_j||^2 - 2 <p_i, p_j>
    coords_norm = (coords ** 2).sum(dim=-1, keepdim=True)  # (B, N, 1)
    dist = coords_norm + coords_norm.transpose(1, 2) - 2.0 * torch.bmm(coords, coords.transpose(1, 2))

    if mask is not None:
        # Padded dummy particles should never be picked as neighbors by real particles:
        # Add a large penalty to columns corresponding to masked particles
        penalty = (1.0 - mask.unsqueeze(1)) * 1e9  # (B, 1, N)
        dist = dist + penalty

    # Select k smallest distances
    knn_indices = torch.topk(dist, k=k, dim=-1, largest=False)[1]  # (B, N, k)
    return knn_indices


def gather_neighbor_features(x: torch.Tensor, knn_indices: torch.Tensor) -> torch.Tensor:
    """
    Gathers neighbor features given k-NN indices.

    Args:
        x: Tensor of shape (B, C, N)
        knn_indices: Tensor of shape (B, N, k)

    Returns:
        neighbor_x: Tensor of shape (B, C, N, k)
    """
    B, C, N = x.shape
    k = knn_indices.shape[-1]

    # Expand indices to match channel dimensions
    indices_expanded = knn_indices.unsqueeze(1).expand(-1, C, -1, -1)  # (B, C, N, k)
    x_expanded = x.unsqueeze(-1).expand(-1, -1, -1, k)                  # (B, C, N, k)

    neighbor_x = torch.gather(x_expanded, dim=2, index=indices_expanded)
    return neighbor_x


class EdgeConvBlock(nn.Module):
    """
    A single EdgeConv Graph Network block with dynamic or static k-NN graph construction.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        k: int = 16,
        dynamic: bool = True,
        dropout: float = 0.05,
    ):
        """
        Args:
            in_channels: Feature dimension of input nodes.
            out_channels: Feature dimension of output nodes.
            k: Number of nearest neighbors.
            dynamic: If True, recomputes k-NN in latent feature space. If False, uses input coords.
            dropout: Dropout probability.
        """
        super().__init__()
        self.k = k
        self.dynamic = dynamic

        # Edge MLP Theta takes [x_j - x_i, x_i] -> 2 * in_channels
        self.edge_mlp = nn.Sequential(
            nn.Conv2d(2 * in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
        )

        # Shortcut projection for residual connection
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm1d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

        self.relu = nn.ReLU(inplace=True)

    def forward(
        self,
        x: torch.Tensor,
        coords: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Node features of shape (B, C, N)
            coords: Spatial / latent coordinates of shape (B, N, D_coord)
            mask: Binary mask of shape (B, N)

        Returns:
            x_out: Updated node features of shape (B, out_channels, N)
            new_coords: Latent coordinates for downstream blocks
        """
        B, C, N = x.shape

        # Determine metric space for k-NN
        if self.dynamic:
            # Latent coordinate space (B, N, C)
            metric_space = x.transpose(1, 2).contiguous()
        else:
            # Fixed coordinate space (B, N, 2)
            metric_space = coords

        knn_idx = compute_knn_indices(metric_space, k=self.k, mask=mask)  # (B, N, k)
        neighbor_x = gather_neighbor_features(x, knn_idx)                  # (B, C, N, k)
        center_x = x.unsqueeze(-1).expand(-1, -1, -1, self.k)              # (B, C, N, k)

        # Construct edge features: [x_j - x_i, x_i]
        edge_feat = torch.cat([neighbor_x - center_x, center_x], dim=1)    # (B, 2*C, N, k)

        # Message transformation
        edge_msgs = self.edge_mlp(edge_feat)                              # (B, out_channels, N, k)

        # Max aggregation across neighbors
        aggregated = torch.max(edge_msgs, dim=-1)[0]                       # (B, out_channels, N)

        # Residual update
        x_out = self.relu(aggregated + self.shortcut(x))

        if mask is not None:
            x_out = x_out * mask.unsqueeze(1)

        return x_out, x_out.transpose(1, 2)


class ParticleNetLite(nn.Module):
    """
    ParticleNet-lite Graph Neural Network for jet tagging.

    Architecture:
      1. Linear projection of input features
      2. Stacked EdgeConv message-passing blocks (with k-NN graph updates)
      3. Global masked pooling (concatenated mean + max)
      4. Fully connected classification head
    """

    def __init__(
        self,
        input_dim: int = 7,
        num_classes: int = 2,
        conv_params: Optional[List[Tuple[int, int]]] = None,
        fc_params: Optional[List[int]] = None,
        k: int = 16,
        dynamic: bool = True,
        dropout: float = 0.1,
    ):
        """
        Args:
            input_dim: Number of input particle features (default 7).
            num_classes: Number of target classes (2 for Quark vs Gluon).
            conv_params: List of (out_channels, k) for each EdgeConv block.
            fc_params: Hidden dimensions for classification MLP.
            k: Default neighborhood size for k-NN.
            dynamic: If True, dynamically updates k-NN in latent space.
            dropout: Dropout probability.
        """
        super().__init__()

        if conv_params is None:
            # 3 stacked EdgeConv blocks
            conv_params = [(64, k), (128, k), (128, k)]
        if fc_params is None:
            fc_params = [128, 64]

        self.input_dim = input_dim
        self.num_classes = num_classes

        # Initial node embedding projection
        first_channels = conv_params[0][0]
        self.input_proj = nn.Sequential(
            nn.Conv1d(input_dim, first_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(first_channels),
            nn.ReLU(inplace=True),
        )

        # Build EdgeConv blocks
        self.edge_blocks = nn.ModuleList()
        in_c = first_channels
        for i, (out_c, block_k) in enumerate(conv_params):
            # First block uses spatial coordinates (delta_eta, delta_phi)
            # Subsequent blocks use dynamic latent space if dynamic=True
            is_dynamic = dynamic if i > 0 else False
            self.edge_blocks.append(
                EdgeConvBlock(
                    in_channels=in_c,
                    out_channels=out_c,
                    k=block_k,
                    dynamic=is_dynamic,
                    dropout=dropout,
                )
            )
            in_c = out_c

        # Global pooling dimension (mean + max)
        pooled_dim = in_c * 2

        # Classification Head
        fc_layers = []
        curr_dim = pooled_dim
        for fc_dim in fc_params:
            fc_layers.extend([
                nn.Linear(curr_dim, fc_dim),
                nn.BatchNorm1d(fc_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            ])
            curr_dim = fc_dim
        fc_layers.append(nn.Linear(curr_dim, num_classes))
        self.classifier = nn.Sequential(*fc_layers)

    def forward(
        self,
        features: torch.Tensor,
        coordinates: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            features: Tensor of shape (B, N, input_dim)
            coordinates: Spatial detector coordinates of shape (B, N, 2) [delta_eta, delta_phi]
            mask: Binary tensor of shape (B, N) with 1.0 for valid particles.

        Returns:
            logits: Tensor of shape (B, num_classes)
        """
        B, N, D = features.shape

        if mask is None:
            mask = torch.ones((B, N), device=features.device, dtype=features.dtype)

        # Transpose to (B, D, N) for Conv1d
        x = features.transpose(1, 2)
        x = self.input_proj(x)
        x = x * mask.unsqueeze(1)

        # Pass through EdgeConv blocks
        curr_coords = coordinates
        for block in self.edge_blocks:
            x, curr_coords = block(x, curr_coords, mask=mask)

        # -------------------------------------------------------------
        # Masked Global Pooling (Concatenated Mean + Max)
        # -------------------------------------------------------------
        valid_counts = mask.sum(dim=-1, keepdim=True).clamp(min=1.0)  # (B, 1)
        mean_pooled = (x * mask.unsqueeze(1)).sum(dim=-1) / valid_counts

        # Padded elements masked with -1e9 for max pooling
        x_for_max = x + (1.0 - mask.unsqueeze(1)) * -1e9
        max_pooled = torch.max(x_for_max, dim=-1)[0]

        pooled = torch.cat([mean_pooled, max_pooled], dim=-1)  # (B, 2 * out_channels)

        # -------------------------------------------------------------
        # Jet Classification Head
        # -------------------------------------------------------------
        logits = self.classifier(pooled)
        return logits
