"""
Real Pythia 8.2 Collider Dataset Downloader & Preprocessor
==========================================================
Fetches the official Pythia 8.2 Quark and Gluon jet dataset from Zenodo:
  - Zenodo Record: 10.5281/zenodo.3164691 (Komiske, Metodiev, Thaler)
  - Raw file: QG_jets.npz (100,000 genuine Pythia simulated jets)

Extracts constituent 4-momenta (pT, rapidity y, azimuth phi, energy E)
and converts to the standard 7-feature representation and 2D coordinates.
"""

import os
import urllib.request
import numpy as np

try:
    from tqdm import tqdm

    class DownloadProgressBar(tqdm):
        def update_to(self, b=1, bsize=1, tsize=None):
            if tsize is not None:
                self.total = tsize
            self.update(b * bsize - self.n)

except ImportError:
    class DownloadProgressBar:
        def __init__(self, *args, **kwargs):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def update_to(self, b=1, bsize=1, tsize=None):
            pass

from src.data.dataset import preprocess_constituents

ZENODO_QG_URL = "https://zenodo.org/records/3164691/files/QG_jets.npz?download=1"


def download_raw_pythia(dest_path: str = "data/raw/QG_jets.npz"):
    """Downloads the official 100k Pythia dataset from Zenodo."""
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    if os.path.exists(dest_path):
        print(f"[*] Raw Pythia dataset already exists at: {dest_path}")
        return

    print(f"[*] Downloading official Pythia 8.2 dataset from Zenodo (106 MB)...")
    with DownloadProgressBar(unit="B", unit_scale=True, miniters=1, desc="QG_jets.npz") as t:
        urllib.request.urlretrieve(ZENODO_QG_URL, filename=dest_path, reporthook=t.update_to)
    print(f"[✓] Download completed: {dest_path}")


def load_and_preprocess_pythia(
    raw_npz_path: str = "data/raw/QG_jets.npz",
    num_jets: int = 20000,
    seed: int = 42,
):
    """
    Loads raw Pythia data and precomputes the 7-feature constituent point cloud.

    Args:
        raw_npz_path: Path to QG_jets.npz
        num_jets: Number of jets to extract (e.g. 20,000 for 10% slice)
        seed: Random seed for shuffling

    Returns:
        features: (num_jets, max_particles, 7)
        coordinates: (num_jets, max_particles, 2)
        mask: (num_jets, max_particles)
        labels: (num_jets,) [0 = Quark, 1 = Gluon]
    """
    if not os.path.exists(raw_npz_path):
        download_raw_pythia(raw_npz_path)

    print(f"[*] Loading raw Pythia data from {raw_npz_path}...")
    data = np.load(raw_npz_path)
    X = data["X"][:num_jets]  # shape: (N, max_particles, 4) -> [pt, y, phi, pdgid]
    y = data["y"][:num_jets]  # shape: (N,) -> 1 = Quark, 0 = Gluon

    print(f"[*] Sliced {len(y)} real Pythia jets.")

    # Constituent variables
    pt = X[:, :, 0].astype(np.float32)
    rapidity = X[:, :, 1].astype(np.float32)
    phi = X[:, :, 2].astype(np.float32)

    # Active particle mask (particles with pT > 0)
    mask = (pt > 0.0).astype(np.float32)

    # Ultrarelativistic massless particle approximation: E = pT * cosh(y)
    energy = pt * np.cosh(rapidity) * mask

    print(f"[*] Preprocessing constituents into 7-feature representation...")
    features, coordinates = preprocess_constituents(
        pt=pt,
        eta=rapidity,
        phi=phi,
        energy=energy,
        mask=mask,
    )

    # Map labels: in EnergyFlow, 1 = Quark, 0 = Gluon.
    # We map to 0 = Quark (Signal), 1 = Gluon (Background) to match our metrics convention:
    labels = (1 - y).astype(np.int64)

    # Random shuffle
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(labels))

    return features[perm], coordinates[perm], mask[perm], labels[perm]


def prepare_pythia_dataset(
    output_path: str = "data/pythia_20k.npz",
    sample_output_path: str = "data/sample_jets.npz",
    num_jets: int = 20000,
    seed: int = 42,
):
    """
    Generates the processed 20k Pythia dataset and updates the 1k sample slice for demo.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    features, coordinates, mask, labels = load_and_preprocess_pythia(
        num_jets=num_jets,
        seed=seed,
    )

    print(f"[*] Saving processed {num_jets}-jet Pythia dataset to {output_path}...")
    np.savez_compressed(
        output_path,
        features=features,
        coordinates=coordinates,
        mask=mask,
        labels=labels,
    )

    # Also update sample_jets.npz with a 1,000-jet slice of real Pythia data for the demo!
    if sample_output_path:
        print(f"[*] Updating {sample_output_path} with 1,000 real Pythia jets for canned demo...")
        np.savez_compressed(
            sample_output_path,
            features=features[:1000],
            coordinates=coordinates[:1000],
            mask=mask[:1000],
            labels=labels[:1000],
        )

    print("[✓] Real Pythia data preparation complete!")


if __name__ == "__main__":
    prepare_pythia_dataset(num_jets=20000)
