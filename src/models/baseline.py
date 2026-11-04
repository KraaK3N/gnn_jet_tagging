"""
Deep Sets Baseline Model for Particle Jet Tagging
=================================================
References:
-----------
1. M. Zaheer et al., "Deep Sets", NeurIPS 2017.
2. P. T. Komiske, E. M. Metodiev, J. Thaler, "Energy Flow Networks: Deep Sets for Particle Jets", JHEP 2019.
3. J. Shlomi, P. Battaglia, J.-R. Vlimant, arXiv:2007.13681 (Sec. 3.1 & 4.3).

Model Concept:
--------------
Deep Sets treats a particle jet as an unordered set of constituents with NO explicit edges (E = empty).
It operates in two stages:
  1. Per-particle feature encoding: h_i = phi(x_i)
  2. Permutation-invariant global pooling: z = Aggregation_{i}(h_i)
  3. Jet classification: y = rho(z)

Because there are no pairwise edges, Deep Sets has O(N) computational and memory complexity,
making it an exceptionally fast and interpretable baseline against which GNNs are compared.
"""

from typing import Optional, List
import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepSetsJetTagger(nn.Module):
    """
    Permutation-invariant Deep Sets classifier for particle jet point clouds.
    """

    def __init__(
        self,
        input_dim: int = 7,
        hidden_dims: Optional[List[int]] = None,
        classifier_dims: Optional[List[int]] = None,
        num_classes: int = 2,
        pooling: str = "mean_max",
        dropout: float = 0.1,
    ):
        """
        Args:
            input_dim: Number of input features per particle (default 7: delta_eta, delta_phi,
                       log_pt, log_E, log_pt_rel, log_E_rel, delta_R).
            hidden_dims: Hidden channel dimensions for the per-particle encoder phi.
            classifier_dims: Hidden dimensions for the jet-level classifier rho.
            num_classes: Number of output classes (2 for Quark vs Gluon).
            pooling: Permutation-invariant pooling type ('mean', 'sum', 'max', or 'mean_max').
            dropout: Dropout probability.
        """
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [64, 128, 128]
        if classifier_dims is None:
            classifier_dims = [128, 64]

        self.input_dim = input_dim
        self.pooling = pooling
        self.num_classes = num_classes

        # -------------------------------------------------------------
        # 1. Per-Particle Encoder: phi(x_i)
        # Implemented using 1D convolutions (kernel_size=1) over (B, C, N)
        # -------------------------------------------------------------
        phi_layers = []
        in_c = input_dim
        for out_c in hidden_dims:
            phi_layers.extend([
                nn.Conv1d(in_c, out_c, kernel_size=1, bias=False),
                nn.BatchNorm1d(out_c),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            ])
            in_c = out_c
        self.phi = nn.Sequential(*phi_layers)

        # Dimension of the pooled jet representation
        latent_dim = hidden_dims[-1]
        pooled_dim = latent_dim * 2 if pooling == "mean_max" else latent_dim

        # -------------------------------------------------------------
        # 2. Jet Classifier: rho(z)
        # -------------------------------------------------------------
        classifier_layers = []
        curr_dim = pooled_dim
        for hidden_c in classifier_dims:
            classifier_layers.extend([
                nn.Linear(curr_dim, hidden_c),
                nn.BatchNorm1d(hidden_c),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            ])
            curr_dim = hidden_c
        classifier_layers.append(nn.Linear(curr_dim, num_classes))

        self.classifier = nn.Sequential(*classifier_layers)

    def forward(
        self,
        features: torch.Tensor,
        coordinates: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            features: Tensor of shape (B, N, input_dim)
            coordinates: Optional tensor of shape (B, N, 2) (unused in Deep Sets)
            mask: Binary tensor of shape (B, N) where 1 indicates real constituent.

        Returns:
            logits: Tensor of shape (B, num_classes)
        """
        B, N, D = features.shape

        if mask is None:
            mask = torch.ones((B, N), device=features.device, dtype=features.dtype)

        # Transpose to (B, D, N) for Conv1d
        x = features.transpose(1, 2)
        
        # Apply per-particle mapping: (B, hidden_dim, N)
        h = self.phi(x)

        # Zero out features of padded particles using mask
        mask_expanded = mask.unsqueeze(1)  # (B, 1, N)
        h = h * mask_expanded

        # -------------------------------------------------------------
        # Masked Permutation-Invariant Pooling
        # -------------------------------------------------------------
        eps = 1e-8
        valid_counts = mask.sum(dim=-1, keepdim=True).clamp(min=1.0)  # (B, 1)

        if self.pooling == "sum":
            z = h.sum(dim=-1)  # (B, latent_dim)
        elif self.pooling == "mean":
            z = h.sum(dim=-1) / valid_counts
        elif self.pooling == "max":
            # Set padded positions to large negative value before max
            h_masked = h + (1.0 - mask_expanded) * -1e9
            z = h_masked.max(dim=-1)[0]
        elif self.pooling == "mean_max":
            mean_z = h.sum(dim=-1) / valid_counts
            h_masked = h + (1.0 - mask_expanded) * -1e9
            max_z = h_masked.max(dim=-1)[0]
            z = torch.cat([mean_z, max_z], dim=-1)  # (B, 2 * latent_dim)
        else:
            raise ValueError(f"Unknown pooling mechanism: {self.pooling}")

        # -------------------------------------------------------------
        # Classify pooled representation
        # -------------------------------------------------------------
        logits = self.classifier(z)
        return logits
