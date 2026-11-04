"""
Model architectures for jet tagging:
  - DeepSetsJetTagger: Permutation-invariant baseline (no edges)
  - ParticleNetLite: Dynamic EdgeConv GNN
"""

from src.models.baseline import DeepSetsJetTagger
from src.models.gnn import ParticleNetLite, EdgeConvBlock

__all__ = ["DeepSetsJetTagger", "ParticleNetLite", "EdgeConvBlock"]
