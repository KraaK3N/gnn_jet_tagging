"""
Physics-Informed Synthetic Jet Generator
=========================================
Generates synthetic Quark and Gluon jets following QCD soft-collinear radiation principles:

Physics Principles:
-------------------
1. Color Charges & Casimir Factors:
   - Quark jets carry fundamental color charge C_F = 4/3.
   - Gluon jets carry adjoint color charge C_A = 3.
   - The ratio C_A / C_F = 9/4 = 2.25 leads to roughly 2x higher gluon splitting probability.

2. Multiplicity:
   - Gluon jets have higher average constituent multiplicity (softer particles).
   - Quark jets have lower multiplicity and a harder central core.

3. Angular Dispersion:
   - Gluon radiation has a wider angular spread in (delta_eta, delta_phi).
   - Quark radiation is tightly collimated around the jet axis.

4. Energy Sharing (DGLAP Splitting):
   - Quark jets: Leading constituent carries a larger fraction of total pT.
   - Gluon jets: Energy is shared more equally among constituents.
"""

from typing import Tuple
import numpy as np
from src.data.dataset import preprocess_constituents


def generate_synthetic_jet_dataset(
    n_samples: int = 1000,
    max_particles: int = 60,
    jet_energy_mean: float = 100.0,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generates synthetic quark (label=0) and gluon (label=1) jets.

    Returns:
        features: (n_samples, max_particles, 7)
        coordinates: (n_samples, max_particles, 2)
        mask: (n_samples, max_particles)
        labels: (n_samples,)
    """
    rng = np.random.RandomState(seed)

    # Balanced binary classes: 0 = Quark, 1 = Gluon
    n_quarks = n_samples // 2
    n_gluons = n_samples - n_quarks
    labels = np.array([0] * n_quarks + [1] * n_gluons, dtype=np.int64)

    # Preallocate constituent arrays
    pt_arr = np.zeros((n_samples, max_particles), dtype=np.float32)
    eta_arr = np.zeros((n_samples, max_particles), dtype=np.float32)
    phi_arr = np.zeros((n_samples, max_particles), dtype=np.float32)
    energy_arr = np.zeros((n_samples, max_particles), dtype=np.float32)
    mask_arr = np.zeros((n_samples, max_particles), dtype=np.float32)

    for i in range(n_samples):
        is_gluon = labels[i] == 1
        
        # Total jet energy & pt
        jet_pT = rng.gamma(shape=10.0, scale=jet_energy_mean / 10.0)
        jet_eta = rng.uniform(-2.0, 2.0)
        jet_phi = rng.uniform(-np.pi, np.pi)

        if is_gluon:
            # Gluon: higher constituent multiplicity (~35-50 particles)
            n_constituents = int(np.clip(rng.poisson(lam=38), 12, max_particles))
            # Wider angular dispersion
            angular_scale = 0.18
            # Softer energy distribution (Dirichlet with smaller alphas)
            alphas = rng.uniform(0.3, 0.7, size=n_constituents)
        else:
            # Quark: lower constituent multiplicity (~18-28 particles)
            n_constituents = int(np.clip(rng.poisson(lam=22), 8, max_particles))
            # Narrower, more collimated core
            angular_scale = 0.09
            # Hard core: first particle carries large fraction of momentum
            alphas = rng.uniform(0.2, 0.5, size=n_constituents)
            alphas[0] += 2.5

        # Normalize energy fractions (sum to 1)
        z = rng.dirichlet(alphas)
        
        # Sort so leading particle comes first
        sort_idx = np.argsort(-z)
        z = z[sort_idx]

        # Angular positions relative to jet axis
        delta_eta = rng.normal(loc=0.0, scale=angular_scale, size=n_constituents)
        delta_phi = rng.normal(loc=0.0, scale=angular_scale, size=n_constituents)
        
        # Hardest constituent is near the center
        delta_eta[0] *= 0.2
        delta_phi[0] *= 0.2

        # Physical 4-momentum components
        pt_i = z * jet_pT
        energy_i = pt_i * np.cosh(jet_eta + delta_eta)

        pt_arr[i, :n_constituents] = pt_i
        eta_arr[i, :n_constituents] = jet_eta + delta_eta
        phi_arr[i, :n_constituents] = (jet_phi + delta_phi + np.pi) % (2.0 * np.pi) - np.pi
        energy_arr[i, :n_constituents] = energy_i
        mask_arr[i, :n_constituents] = 1.0

    features, coordinates = preprocess_constituents(
        pt=pt_arr,
        eta=eta_arr,
        phi=phi_arr,
        energy=energy_arr,
        mask=mask_arr,
    )

    # Shuffle the dataset
    shuffle_idx = rng.permutation(n_samples)
    return features[shuffle_idx], coordinates[shuffle_idx], mask_arr[shuffle_idx], labels[shuffle_idx]


def save_synthetic_dataset(output_path: str, n_samples: int = 1000, seed: int = 42):
    """Generates and saves the dataset to a compressed .npz archive."""
    features, coordinates, mask, labels = generate_synthetic_jet_dataset(
        n_samples=n_samples,
        seed=seed,
    )
    np.savez_compressed(
        output_path,
        features=features,
        coordinates=coordinates,
        mask=mask,
        labels=labels,
    )
    print(f"Saved {n_samples} synthetic jets to {output_path}")


if __name__ == "__main__":
    import os
    os.makedirs("data", exist_ok=True)
    save_synthetic_dataset("data/sample_jets.npz", n_samples=1000)
