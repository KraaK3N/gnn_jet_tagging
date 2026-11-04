# Learned GNN Physics Simulator for Particle Jet Tagging

**Course:** CS6012 — Advanced Machine Learning / Deep Learning, IIT Bombay  
**Supervisor:** Prof. Sharat Chandran  
**Group 11:** Pradipta Sadhukhan (25D1080) & Rahul Kumar (25M2015)  
**Reference Paper:** J. Shlomi, P. Battaglia, J.-R. Vlimant, *Graph Neural Networks in Particle Physics*, arXiv:2007.13681 (2020)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 1. Project Overview

In high-energy collider physics (e.g., CERN Large Hadron Collider), collisions produce quarks and gluons that shower into collimated sprays of particles called **jets**. **Jet Tagging** is the task of inferring the originating particle from constituent particle kinematics.

Jets are naturally **point clouds / graphs** (unordered collections of particles with spatial and momentum relationships). Standard CNNs suffer from extreme detector sparsity, while RNNs impose an artificial sequence order on unordered sets. 

Following the review by Shlomi et al. (arXiv:2007.13681), this project implements and evaluates two representative jet-tagging paradigms:
1. **Deep Sets Baseline (No Edges):** Encodes each particle independently and pools representations using permutation-invariant aggregations ($O(N)$ complexity).
2. **Dynamic EdgeConv GNN (ParticleNet-lite):** Dynamically constructs a $k$-Nearest Neighbors ($k$-NN) graph in latent feature space and performs relational message passing along edges ($O(k \cdot N)$ complexity).

---

## 2. Experimental Benchmark Results

Evaluated on the held-out test split of the 1,000-jet Quark vs. Gluon benchmark dataset:

| Model Architecture | Graph Structure | Test Accuracy | ROC AUC | Background Rejection ($1/\varepsilon_B$ @ 50%) | Inference Latency (CPU) | Parameters |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Deep Sets Baseline** | No Edges ($E = \emptyset$) | **96.67%** | **0.9993** | $10,000$ | **0.23 ms / jet** | **67,330** |
| **Dynamic EdgeConv (ParticleNet-lite)** | Dynamic $k$-NN ($k=16$) | **100.00%** | **1.0000** | $10,000$ | **3.31 ms / jet** | **108,674** |

### Key Findings:
- **Accuracy & AUC:** Dynamic EdgeConv achieves near-perfect classification, capturing fine-grained multi-particle relational substructures that independent set encoding misses.
- **Computational Cost:** Deep Sets is **$14\times$ faster** in inference and trains in **~12 seconds** on CPU, demonstrating the classic efficiency vs. expressiveness trade-off in Graph Neural Networks.

---

## 3. Quickstart: 1-Click Canned Demo (For TA Evaluation)

The canned demo runs out-of-the-box on any standard CPU laptop with **zero configuration** and finishes in **< 10 seconds**:

```bash
# 1. Clone the repository
git clone https://github.com/SADHUKHAN-PRADIPTA/gnn-jet-tagging.git
cd gnn-jet-tagging

# 2. Install dependencies (CPU PyTorch, numpy, scikit-learn, matplotlib, pyyaml)
pip install -r requirements.txt

# 3. Run the executable canned demo
./run_demo.sh
```

### What `./run_demo.sh` Does:
1. Loads pre-trained checkpoints (`checkpoints/deepsets_best.pt` and `checkpoints/edgeconv_best.pt`).
2. Evaluates both models on the bundled benchmark slice (`data/sample_jets.npz`).
3. Computes accuracy, ROC AUC, background rejection, parameter counts, and inference latency per jet.
4. Generates and saves a publication-quality comparative ROC curve to `results/figures/demo_roc_curve.png`.

---

## 4. Repository Structure

```
gnn-jet-tagging/
├── README.md               # Visual project documentation & benchmark summary
├── README.txt              # Required: Honor Code pledge & LLM usage links
├── run_demo.sh             # 1-click executable demo script for teaching team
├── demo.py                 # Self-contained evaluation script
├── requirements.txt        # Minimal, robust Python dependencies
├── plan.md                 # Detailed project plan & rubric specifications
├── configs/
│   ├── baseline.yaml       # Hyperparameters for Deep Sets
│   └── gnn.yaml            # Hyperparameters for Dynamic EdgeConv GNN
├── data/
│   └── sample_jets.npz     # Bundled 1,000-jet benchmark dataset (< 1.5 MB)
├── checkpoints/
│   ├── deepsets_best.pt    # Pre-trained baseline model weights
│   └── edgeconv_best.pt    # Pre-trained Dynamic EdgeConv GNN weights
├── results/
│   ├── metrics_summary.json# Stored benchmark comparison numbers
│   └── figures/
│       ├── demo_roc_curve.png         # Comparative ROC curves
│       ├── deepsets_training_history.png
│       └── gnn_training_history.png
├── src/
│   ├── data/
│   │   ├── dataset.py      # JetDataset & 7-feature constituent preprocessing
│   │   ├── synthetic.py    # Physics-grounded QCD toy generator
│   │   └── download.py     # Public benchmark dataset fetcher
│   ├── models/
│   │   ├── baseline.py     # Deep Sets permutation-invariant model
│   │   └── gnn.py          # Vectorized Dynamic EdgeConv (ParticleNet-lite)
│   ├── training/
│   │   ├── train.py        # Multi-epoch training loop with early checkpointing
│   │   └── losses.py       # Loss functions with label smoothing
│   ├── evaluation/
│   │   ├── evaluate.py     # Test set evaluator & comparative benchmarking
│   │   └── metrics.py      # ROC AUC & R50 background rejection calculations
│   └── utils/
│       ├── config.py       # YAML configuration parser
│       ├── plotting.py     # Publication-quality Matplotlib figures
│       └── seed.py         # Deterministic seed configuration
├── tests/
│   └── test_graph_building.py # Unit tests (invariance, masking, shapes)
└── report/
    ├── final_report.tex    # LaTeX source for final project report
    └── final_report.pdf    # Compiled PDF report
```

---

## 5. Reproducing Training from Scratch

To retrain the models from scratch on CPU:

```bash
# Train the Deep Sets Baseline (~12 seconds on CPU)
python -m src.training.train --config configs/baseline.yaml

# Train the Dynamic EdgeConv GNN (~110 seconds on CPU)
python -m src.training.train --config configs/gnn.yaml

# Run Comparative Evaluation
python -m src.evaluation.evaluate
```

*(To run a quick 2-epoch test instead, append `--quick` to the training command).*

---

## 6. Running Unit Tests

To verify tensor dimensions, masking logic, cyclic angle wrapping, and exact mathematical permutation invariance:

```bash
python -m unittest tests/test_graph_building.py
```
*All 5 tests pass in ~0.1 seconds.*

---

## 7. Mathematical Formulations

### Feature Representation
Each particle $i$ in a jet is represented by 7 features:
$$x_i = \left[ \Delta \eta_i, \; \Delta \phi_i, \; \log p_{T, i}, \; \log E_i, \; \log \frac{p_{T, i}}{p_{T, \text{jet}}}, \; \log \frac{E_i}{E_{\text{jet}}}, \; \Delta R_i \right]$$

### Deep Sets Formulation
$$y = \rho \left( \bigoplus_{i \in \mathcal{V}} \phi(x_i) \right)$$
where $\phi$ is an MLP and $\bigoplus$ is masked concatenation of mean and max pooling.

### Dynamic EdgeConv Formulation
For each particle $i$ and neighbor $j \in \mathcal{N}_k(i)$ in the dynamic $k$-NN graph:
$$e_{ij} = \Theta([x_j - x_i, \, x_i])$$
$$x_i' = \max_{j \in \mathcal{N}_k(i)} e_{ij} + W_{\text{res}} x_i$$
Followed by global masked pooling and dense classification.
