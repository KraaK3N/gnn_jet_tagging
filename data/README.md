# Particle Jet Datasets

This directory manages the benchmark datasets for Quark vs. Gluon jet tagging.

## Datasets Overview

### 1. Bundled Verification Dataset (`sample_jets.npz`)
- **Size:** 1,000 jets (~1.2 MB compressed `.npz`)
- **Purpose:** Used for the zero-configuration TA canned demo (`./run_demo.sh`).
- **Splits:** 700 train / 150 val / 150 test.
- **Arrays:**
  - `features`: `(N, max_particles, 7)` — `[Δη, Δϕ, log(pT), log(E), log(pT/pT_jet), log(E/E_jet), ΔR]`
  - `coordinates`: `(N, max_particles, 2)` — `(Δη, Δϕ)` for dynamic $k$-NN graph construction.
  - `mask`: `(N, max_particles)` — Binary mask (1.0 for valid particles, 0.0 for padding).
  - `labels`: `(N,)` — Binary labels (0 = Quark / Signal, 1 = Gluon / Background).

---

### 2. Full Official Pythia 8.2 Collider Benchmark (`raw/QG_jets.npz` and `pythia_100k.npz`)
- **Source:** Open-science Pythia 8.2 Quark and Gluon dataset deposited on Zenodo:
  - **Zenodo DOI:** [10.5281/zenodo.3164691](https://doi.org/10.5281/zenodo.3164691)
  - **Authors:** Patrick T. Komiske, Eric M. Metodiev, Jesse Thaler (MIT).
- **Scale:** 100,000 genuine simulated collider jets (50,000 Quark + 50,000 Gluon).
- **Automated Fetching:** Executed via `notebooks/gnn.ipynb` or:
  ```python
  from src.data.pythia import prepare_pythia_dataset
  prepare_pythia_dataset(num_jets=100000)
  ```
