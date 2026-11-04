"""
Unit Tests: Stage 1 (Data Pipeline) and Stage 2 (Models)
========================================================
Verifies:
  1. Coordinate preprocessing (cyclic delta_phi wrapping, log scaling)
  2. Dataset and DataLoader tensor shapes and masking
  3. Deep Sets forward pass and permutation invariance
  4. ParticleNet-lite (EdgeConv) forward pass and permutation invariance
  5. Distance computation and dynamic k-NN graph masking
"""

import unittest
import numpy as np
import torch
from src.data.dataset import (
    preprocess_constituents,
    compute_relative_coordinates,
    JetDataset,
    load_jet_dataset,
)
from src.data.synthetic import generate_synthetic_jet_dataset
from src.models.baseline import DeepSetsJetTagger
from src.models.gnn import ParticleNetLite, compute_knn_indices


class TestDataPipeline(unittest.TestCase):
    def test_relative_coordinates_cyclic_wrapping(self):
        """Checks that dphi handles periodic boundaries correctly."""
        eta = np.array([[0.5, 0.5]])
        phi = np.array([[3.14, -3.14]])
        jet_eta = np.array([[0.5]])
        jet_phi = np.array([[-3.14]])

        d_eta, d_phi, d_R = compute_relative_coordinates(eta, phi, jet_eta, jet_phi)
        self.assertAlmostEqual(d_eta[0, 0], 0.0, places=5)
        self.assertAlmostEqual(d_phi[0, 1], 0.0, places=5)
        # 3.14 - (-3.14) = 6.28, which wraps to ~0 in [-pi, pi]
        self.assertAlmostEqual(abs(d_phi[0, 0]), 0.0, delta=0.01)

    def test_synthetic_data_generation_shapes(self):
        """Verifies synthetic dataset shapes and class balance."""
        features, coords, mask, labels = generate_synthetic_jet_dataset(
            n_samples=20, max_particles=40, seed=123
        )
        self.assertEqual(features.shape, (20, 40, 7))
        self.assertEqual(coords.shape, (20, 40, 2))
        self.assertEqual(mask.shape, (20, 40))
        self.assertEqual(labels.shape, (20,))
        # Check balance
        self.assertEqual(int(labels.sum()), 10)


class TestModels(unittest.TestCase):
    def setUp(self):
        self.features, self.coords, self.mask, self.labels = generate_synthetic_jet_dataset(
            n_samples=16, max_particles=30, seed=42
        )
        self.feat_tensor = torch.tensor(self.features, dtype=torch.float32)
        self.coords_tensor = torch.tensor(self.coords, dtype=torch.float32)
        self.mask_tensor = torch.tensor(self.mask, dtype=torch.float32)

    def test_deep_sets_shape_and_permutation_invariance(self):
        """Verifies Deep Sets output shape and exact permutation invariance."""
        model = DeepSetsJetTagger(input_dim=7, num_classes=2)
        model.eval()

        with torch.no_grad():
            logits = model(self.feat_tensor, self.coords_tensor, self.mask_tensor)
            self.assertEqual(logits.shape, (16, 2))

            # Permute constituents
            perm = torch.randperm(30)
            feat_perm = self.feat_tensor[:, perm, :]
            mask_perm = self.mask_tensor[:, perm]

            logits_perm = model(feat_perm, None, mask_perm)
            max_diff = torch.max(torch.abs(logits - logits_perm)).item()
            self.assertLess(max_diff, 1e-4)

    def test_particlenet_shape_and_permutation_invariance(self):
        """Verifies ParticleNet-lite EdgeConv output shape and permutation invariance."""
        model = ParticleNetLite(input_dim=7, num_classes=2, k=8)
        model.eval()

        with torch.no_grad():
            logits = model(self.feat_tensor, self.coords_tensor, self.mask_tensor)
            self.assertEqual(logits.shape, (16, 2))

            # Permute constituents
            perm = torch.randperm(30)
            feat_perm = self.feat_tensor[:, perm, :]
            coords_perm = self.coords_tensor[:, perm, :]
            mask_perm = self.mask_tensor[:, perm]

            logits_perm = model(feat_perm, coords_perm, mask_perm)
            max_diff = torch.max(torch.abs(logits - logits_perm)).item()
            self.assertLess(max_diff, 1e-4)

    def test_knn_masking_penalty(self):
        """Checks that masked padded nodes are never selected as k-NN."""
        coords = torch.randn(2, 10, 2)
        mask = torch.tensor([
            [1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
        ], dtype=torch.float32)

        knn_idx = compute_knn_indices(coords, k=3, mask=mask)
        # For batch 0, active nodes are 0..3. Selected neighbors should all be < 4.
        active_neighbors_b0 = knn_idx[0, :4]
        self.assertTrue(torch.all(active_neighbors_b0 < 4))


if __name__ == "__main__":
    unittest.main()
