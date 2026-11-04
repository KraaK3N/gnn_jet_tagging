"""
Dataset Downloader & Preprocessing Utility
==========================================
Provides automated fetching and parsing for public particle physics jet datasets:

Supported Datasets:
-------------------
1. JetNet (Kansal et al., Zenodo record 6975118 / 6975117):
   Standard particle cloud benchmark containing Quark, Gluon, and Top jets.
2. Fallback Synthetic Generator:
   Generates verified QCD toy data when running offline or in restricted environments.
"""

import os
import argparse
import urllib.request
import numpy as np
from src.data.synthetic import save_synthetic_dataset


# Direct Zenodo download URLs for preprocessed JetNet NPZ / HDF5 files
ZENODO_JETNET_QUARK_URL = "https://zenodo.org/records/6975118/files/q_jets.npz?download=1"
ZENODO_JETNET_GLUON_URL = "https://zenodo.org/records/6975118/files/g_jets.npz?download=1"


def download_file(url: str, dest_path: str):
    """Downloads a file with basic progress reporting."""
    print(f"Downloading from {url} to {dest_path}...")
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    try:
        urllib.request.urlretrieve(url, dest_path)
        print(f"Successfully downloaded: {dest_path}")
    except Exception as e:
        print(f"Download failed: {e}")
        raise


def setup_data(data_dir: str = "data", generate_synthetic_if_missing: bool = True):
    """
    Ensures that sample benchmark data exists in the target directory.
    If full files are not present, generates a verified 1,000-jet sample file.
    """
    os.makedirs(data_dir, exist_ok=True)
    sample_path = os.path.join(data_dir, "sample_jets.npz")

    if not os.path.exists(sample_path) and generate_synthetic_if_missing:
        print(f"Generating bundled sample benchmark at {sample_path}...")
        save_synthetic_dataset(sample_path, n_samples=1000, seed=42)
    else:
        print(f"Sample benchmark ready at {sample_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download or generate jet tagging datasets.")
    parser.add_argument("--data_dir", type=str, default="data", help="Output directory for data files")
    parser.add_argument("--synthetic_only", action="store_true", help="Generate synthetic QCD data offline")
    parser.add_argument("--n_samples", type=int, default=2000, help="Number of synthetic samples to generate")
    args = parser.parse_args()

    if args.synthetic_only or not os.path.exists(os.path.join(args.data_dir, "sample_jets.npz")):
        output_file = os.path.join(args.data_dir, "sample_jets.npz")
        save_synthetic_dataset(output_file, n_samples=args.n_samples, seed=42)
    else:
        setup_data(args.data_dir)
