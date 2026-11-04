"""
Jet Dataset & Preprocessing Pipeline
====================================
Physics Motivation:
-------------------
Jets are collimated sprays of energetic particles originating from high-energy quarks or gluons.
Each jet contains variable numbers of constituents (particles).

To preserve physical symmetries (longitudinal boost invariance along the beam axis,
rotational invariance in the transverse plane, and translation invariance relative to the jet axis),
we express constituent coordinates relative to the overall jet momentum axis:

  - Relative Pseudorapidity: delta_eta_i = eta_i - eta_jet
  - Relative Azimuthal Angle: delta_phi_i = ((phi_i - phi_jet + pi) % (2*pi)) - pi
  - Angular Separation: delta_R_i = sqrt(delta_eta_i^2 + delta_phi_i^2)
  - Relative Momentum Fraction: log(p_T,i / p_T,jet)
  - Logarithmic Scaling: log(p_T,i), log(E_i) to tame wide dynamic range.
"""

from typing import Tuple, Optional, Dict
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class JetDataset(Dataset):
    """
    PyTorch Dataset representing particle jets as point clouds / graphs.

    Each jet is formatted as:
      - features: Tensor of shape (max_particles, num_features)
                  [delta_eta, delta_phi, log_pt, log_E, log_pt_rel, log_E_rel, delta_R]
      - coordinates: Tensor of shape (max_particles, 2)
                     [delta_eta, delta_phi] for geometric k-NN graph construction
      - mask: Boolean / Float Tensor of shape (max_particles,)
              1.0 for real particle constituents, 0.0 for zero-padded slots
      - label: Scalar integer tensor (0 = Quark jet, 1 = Gluon jet)
    """

    def __init__(
        self,
        features: np.ndarray,
        coordinates: np.ndarray,
        mask: np.ndarray,
        labels: np.ndarray,
    ):
        """
        Args:
            features: (N_jets, max_particles, num_features) array of particle kinematics.
            coordinates: (N_jets, max_particles, 2) array of spatial (delta_eta, delta_phi).
            mask: (N_jets, max_particles) binary array indicating active particles.
            labels: (N_jets,) array of binary class labels.
        """
        assert len(features) == len(labels), "Features and labels must have matching jet counts"
        
        self.features = torch.as_tensor(features, dtype=torch.float32)
        self.coordinates = torch.as_tensor(coordinates, dtype=torch.float32)
        self.mask = torch.as_tensor(mask, dtype=torch.float32)
        self.labels = torch.as_tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.features[idx],
            self.coordinates[idx],
            self.mask[idx],
            self.labels[idx],
        )


def compute_relative_coordinates(
    constituent_eta: np.ndarray,
    constituent_phi: np.ndarray,
    jet_eta: np.ndarray,
    jet_phi: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes delta_eta, delta_phi, and delta_R with proper periodic wrap-around at [-pi, pi].

    Args:
        constituent_eta: (N_jets, max_particles)
        constituent_phi: (N_jets, max_particles)
        jet_eta: (N_jets, 1) or (N_jets,)
        jet_phi: (N_jets, 1) or (N_jets,)

    Returns:
        delta_eta, delta_phi, delta_R
    """
    jet_eta = np.expand_dims(jet_eta, -1) if jet_eta.ndim == 1 else jet_eta
    jet_phi = np.expand_dims(jet_phi, -1) if jet_phi.ndim == 1 else jet_phi

    delta_eta = constituent_eta - jet_eta
    
    # Handle cyclic boundaries in azimuth (-pi to pi)
    dphi = constituent_phi - jet_phi
    delta_phi = (dphi + np.pi) % (2.0 * np.pi) - np.pi
    
    delta_R = np.sqrt(delta_eta**2 + delta_phi**2)
    return delta_eta, delta_phi, delta_R


def preprocess_constituents(
    pt: np.ndarray,
    eta: np.ndarray,
    phi: np.ndarray,
    energy: np.ndarray,
    mask: np.ndarray,
    jet_pt: Optional[np.ndarray] = None,
    jet_eta: Optional[np.ndarray] = None,
    jet_phi: Optional[np.ndarray] = None,
    jet_energy: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Constructs the standard ParticleNet 7-feature representation and 2D coordinates:
      Features: [delta_eta, delta_phi, log(pT), log(E), log(pT/jet_pT), log(E/jet_E), delta_R]
      Coordinates: [delta_eta, delta_phi]

    Masked particles (zeros) are carefully zeroed out to prevent numerical NaNs.
    """
    eps = 1e-8

    # If overall jet properties are not supplied, compute them from constituents sum
    if jet_pt is None:
        jet_pt = np.sum(pt * mask, axis=1, keepdims=True) + eps
    if jet_energy is None:
        jet_energy = np.sum(energy * mask, axis=1, keepdims=True) + eps
    if jet_eta is None:
        # Energy-weighted centroid
        weights = (energy * mask) / (jet_energy + eps)
        jet_eta = np.sum(eta * weights, axis=1, keepdims=True)
    if jet_phi is None:
        weights = (energy * mask) / (jet_energy + eps)
        jet_phi = np.sum(phi * weights, axis=1, keepdims=True)

    delta_eta, delta_phi, delta_R = compute_relative_coordinates(eta, phi, jet_eta, jet_phi)

    # Safe log transforms (masked padded slots get 0)
    safe_pt = np.maximum(pt, eps)
    safe_energy = np.maximum(energy, eps)
    log_pt = np.log(safe_pt) * mask
    log_E = np.log(safe_energy) * mask

    jet_pt_expanded = np.expand_dims(jet_pt, -1) if jet_pt.ndim == 1 else jet_pt
    jet_E_expanded = np.expand_dims(jet_energy, -1) if jet_energy.ndim == 1 else jet_energy

    log_pt_rel = np.log(np.maximum(safe_pt / (jet_pt_expanded + eps), eps)) * mask
    log_E_rel = np.log(np.maximum(safe_energy / (jet_E_expanded + eps), eps)) * mask

    delta_eta = delta_eta * mask
    delta_phi = delta_phi * mask
    delta_R = delta_R * mask

    features = np.stack(
        [delta_eta, delta_phi, log_pt, log_E, log_pt_rel, log_E_rel, delta_R],
        axis=-1,
    )
    coordinates = np.stack([delta_eta, delta_phi], axis=-1)

    return features, coordinates


def load_jet_dataset(
    npz_path: str,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Dict[str, JetDataset]:
    """
    Loads an NPZ file containing preprocessed or raw jet arrays and splits into
    'train', 'val', and 'test' JetDataset splits.
    """
    data = np.load(npz_path)
    features = data["features"]
    coordinates = data["coordinates"]
    mask = data["mask"]
    labels = data["labels"]

    n_samples = len(labels)
    indices = np.arange(n_samples)
    rng = np.random.RandomState(seed)
    rng.shuffle(indices)

    n_test = int(n_samples * test_ratio)
    n_val = int(n_samples * val_ratio)
    n_train = n_samples - n_val - n_test

    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]
    test_idx = indices[n_train + n_val :]

    splits = {
        "train": JetDataset(features[train_idx], coordinates[train_idx], mask[train_idx], labels[train_idx]),
        "val": JetDataset(features[val_idx], coordinates[val_idx], mask[val_idx], labels[val_idx]),
        "test": JetDataset(features[test_idx], coordinates[test_idx], mask[test_idx], labels[test_idx]),
    }
    return splits
