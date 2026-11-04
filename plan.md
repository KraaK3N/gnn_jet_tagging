# Project Plan: Graph Neural Networks for Particle Jet Tagging

**Course:** CS6012 — Advanced Machine Learning / Deep Learning, IIT Bombay  
**Supervisor:** Prof. Sharat Chandran  
**Group 11:** Pradipta Sadhukhan (Roll: 25D1080) & Rahul Kumar (Roll: 25M2015)  
**Primary Reference:** J. Shlomi, P. Battaglia, J.-R. Vlimant, *Graph Neural Networks in Particle Physics*, arXiv:2007.13681 (2020)  
**Target Repository:** `gnn-jet-tagging` (linked in classroom50 / GitHub)

---

## 1. Executive Summary & Problem Formulation

### 1.1 The Physics Context
In high-energy collider physics (e.g., CERN LHC), high-energy collisions produce elementary particles (such as quarks or gluons) that undergo hadronization and showering, resulting in collimated sprays of energetic particles called **jets**.

**Jet Tagging** is the task of inferring the originating particle of a jet (e.g., distinguishing quark-initiated jets from gluon-initiated jets, or top-quark jets from generic QCD background) solely from the kinematic properties of its detected constituent particles.

### 1.2 Why Graph Neural Networks?
- **Grid/Image representations (CNNs):** Discretizing particle showers into calorimeters forces continuous coordinates onto fixed pixel grids. This creates extreme sparsity (>95% empty cells) and artifacts due to irregular detector geometries.
- **Sequence representations (RNNs/LSTMs):** Particles have no intrinsic physical order. Sorting them by transverse momentum ($p_T$) imposes an artificial sequence bias.
- **Point Clouds / Graphs (GNNs):** Jets are fundamentally unordered sets of particles with geometric distances ($\Delta R = \sqrt{\Delta \eta^2 + \Delta \phi^2}$) and momentum correlations. GNNs inherently preserve permutation invariance and naturally model relational substructure.

### 1.3 The Core Research Question
> **"How much does explicit relational message passing (dynamic EdgeConv) improve jet classification performance over permutation-invariant set pooling (Deep Sets), and does the performance boost justify the added computational complexity ($O(k \cdot N)$ vs. $O(N)$)?"**

---

## 2. Model Architecture Comparison

We implement and evaluate two primary architectures representing the opposite ends of the graph spectrum outlined in arXiv:2007.13681, plus one geometric ablation:

```
[Input Jet: N particles × D features]
           │
     ┌─────┴─────────────────────────────────┐
     ▼                                       ▼
┌─────────────────────────┐     ┌─────────────────────────┐
│       Model 1:          │     │       Model 2:          │
│       Deep Sets         │     │  Dynamic EdgeConv GNN   │
│   (No Edges Baseline)   │     │   (ParticleNet-lite)    │
├─────────────────────────┤     ├─────────────────────────┤
│ • Per-particle MLP φ(x) │     │ • Dynamic k-NN in latent│
│ • Global Pool (Sum/Mean)│     │   space (k = 16)        │
│ • Classifier MLP ρ(z)   │     │ • EdgeConv:             │
│                         │     │   e_ij = MLP(x_j-x_i,x_i│
│                         │     │ • Aggregation: Max-pool │
│                         │     │ • Global Mean/Max Pool  │
│                         │     │ • Classifier MLP        │
└────────────┬────────────┘     └────────────┬────────────┘
             ▼                               ▼
       [P(Quark/Gluon)]                [P(Quark/Gluon)]
```

### Model 1: Deep Sets Baseline (Energy Flow Network style)
- **Graph Structure:** Empty edge set ($E = \emptyset$).
- **Mathematical Formulation:**
  $$y = \rho \left( \sum_{i \in \mathcal{V}} \phi(x_i) \right)$$
  where $\phi: \mathbb{R}^D \to \mathbb{R}^{d}$ is a per-particle MLP and $\rho: \mathbb{R}^d \to \mathbb{R}^C$ is a jet-level classification MLP.
- **Complexity:** $O(N)$ operations and memory. Extremely fast, lightweight, and fully permutation-invariant.

### Model 2: Dynamic EdgeConv GNN (ParticleNet-lite)
- **Graph Structure:** Dynamically constructed $k$-Nearest Neighbors ($k$-NN) graph recalculated at each layer in the learned feature space.
- **Mathematical Formulation:**
  For node $i$ with neighbors $j \in \mathcal{N}_k(i)$:
  $$e_{ij} = \Theta \left( x_j - x_i, \, x_i \right)$$
  $$x_i' = \max_{j \in \mathcal{N}_k(i)} e_{ij}$$
  followed by global pooling and dense classification.
- **Complexity:** $O(k \cdot N)$ edge evaluations. Captures local geometric clustering and multi-scale substructure.

### Model 3: Static Geometric $k$-NN GNN (Ablation)
- Same EdgeConv architecture, but the $k$-NN graph is computed **once** using fixed physical detector coordinates ($\Delta \eta, \Delta \phi$) rather than dynamic latent feature distances.
- Answers whether dynamic latent neighborhood re-computation is necessary.

---

## 3. Data Strategy & Features

### 3.1 Constituent Input Features
Each jet consists of $N \approx 30\text{--}100$ particles. For each particle $i$, the feature vector $x_i \in \mathbb{R}^6$ includes:
1. $\Delta \eta_i = \eta_i - \eta_{\text{jet}}$ (relative pseudorapidity)
2. $\Delta \phi_i = \phi_i - \phi_{\text{jet}}$ (relative azimuthal angle)
3. $\log(p_{T, i})$ (logarithm of transverse momentum)
4. $\log(E_i)$ (logarithm of particle energy)
5. $\log(p_{T, i} / p_{T, \text{jet}})$ (relative momentum fraction)
6. $\Delta R_i = \sqrt{\Delta \eta_i^2 + \Delta \phi_i^2}$ (angular separation from jet axis)

### 3.2 Two-Tier Dataset Design (Crucial for TA Grading)
- **Tier 1 — Bundled Canned Test Slice (`data/sample_jets.npz`):**  
  A clean 1,000-jet preprocessed subset committed directly into the repository (< 2 MB). This guarantees that the TA canned demo runs **instantly offline** without downloading large files or crashing.
- **Tier 2 — Benchmark Dataset (`src/data/download.py`):**  
  Automated script to download and parse the standard open Quark-Gluon / JetNet benchmark dataset (~50,000 jets) for full model training.

---

## 4. Evaluation Protocol & Metrics

In high-energy physics, raw accuracy alone is insufficient. We report:

1. **ROC Curve & AUC (Area Under Curve):**  
   Measures discriminative power across all operational thresholds.
2. **Background Rejection Rate ($R_{50}$ and $R_{30}$):**  
   $$R_{X} = \frac{1}{\varepsilon_{\text{background}}} \quad \text{at} \quad \varepsilon_{\text{signal}} = X\%$$
   The standard collider metric showing how much background is suppressed while keeping 50% (or 30%) of signal jets.
3. **Classification Accuracy & Cross-Entropy Loss:**  
   Overall convergence and accuracy.
4. **Computational & Resource Benchmarks:**
   - Parameter count
   - Training time per epoch (seconds)
   - Inference latency per jet (milliseconds on CPU)
   - Peak memory usage

---

## 5. Deliverables & Submission Requirements Checklist

| Deliverable | Location | Description / Purpose |
| :--- | :--- | :--- |
| **1. Source Code** | `gnn-jet-tagging/src/` | Modular, thoroughly commented PyTorch code with full docstrings. |
| **2. Canned Demo Executable** | `run_demo.sh` + `demo.py` | Single-command executable for TAs: runs in <10 seconds on CPU. |
| **3. Pre-trained Checkpoints** | `checkpoints/` | Saved weights for Deep Sets and EdgeConv (`.pt` files). |
| **4. Honor Code & LLM Links** | `README.txt` | Academic pledge signed by both members, plus specific LLM usage URLs. |
| **5. GitHub Repository README** | `README.md` | Clean overview, architecture diagrams, benchmark table, reproduction steps. |
| **6. Project Report (PDF)** | `report/final_report.pdf` | Comprehensive 4–6 page LaTeX report with methodology, results & plots. |
| **7. 2–3 Minute Presentation** | `presentation/` | Slide deck (`slides.pdf`), video link (`video_link.txt`), and spoken script. |
| **8. Moodle Submissions** | Moodle upload | Final Report PDF and Presentation Video (or link) as per course instructions. |

---

## 6. The "TA-Proof" Canned Demo Specification

Per course guidelines:  
> *"There will be a canned demo or some sort of executable so that the teaching team can run your code with least amount of effort. Just imagine if you were the TA and try to minimize the work."*

### Demo Design Principles:
1. **Zero C++ Build Dependency:** Written in vectorized pure PyTorch (`torch.cdist`, `torch.topk`). Runs anywhere without `torch-scatter` / CUDA mismatch errors.
2. **One-Command Execution:**
   ```bash
   ./run_demo.sh
   ```
3. **What the TA sees in <10 seconds:**
   - Terminal prints: loading model checkpoints, running inference on `sample_jets.npz`.
   - Formatted ASCII performance summary table comparing Deep Sets vs. EdgeConv GNN.
   - Saves high-resolution ROC curve plot to `results/figures/demo_roc_curve.png`.
4. **Quick Training Option:**
   ```bash
   python -m src.training.train --quick
   ```
   Runs 2 micro-epochs on CPU in ~30 seconds if the TA wants to test the training loop from scratch.

---

## 7. Step-by-Step Implementation Roadmap

```
Phase 1: Environment & Data Pipeline
├── Set up minimal requirements.txt (pure PyTorch, numpy, matplotlib, pyyaml, scikit-learn)
├── Implement src/data/dataset.py (constituent parsing, centering, normalization)
└── Create data/sample_jets.npz (verified 1,000-jet benchmark slice)

Phase 2: Model Architecture Implementation
├── Implement src/models/baseline.py (Deep Sets / Energy Flow Network)
├── Implement src/models/gnn.py (Vectorized Dynamic EdgeConv / ParticleNet-lite)
└── Write unit tests in tests/ to verify tensor shapes and permutation invariance

Phase 3: Training & Metric Benchmarking
├── Implement src/training/train.py with checkpointing and loss logging
├── Implement src/evaluation/evaluate.py (ROC AUC, R50 background rejection)
├── Train Deep Sets and EdgeConv models; save weights to checkpoints/
└── Generate publication plots (ROC curves, confusion matrix, training loss)

Phase 4: Canned Demo Packaging
├── Create demo.py (loads checkpoints, runs sample slice, outputs metrics & plot)
├── Create executable run_demo.sh (chmod +x)
└── Verify execution on clean CPU environment in <10 seconds

Phase 5: Reports, Documentation & Presentation
├── Write README.txt (Honor code signed by Rahul & Pradipta + LLM prompt links)
├── Write README.md (Usage instructions, benchmark table, architecture)
├── Write report/final_report.tex & compile final_report.pdf (LaTeX)
└── Prepare presentation/slides.pdf and record 2–3 minute video presentation
```

---

## 8. Division of Responsibilities (Group 11)

- **Rahul Kumar (25M2015):** Data pipeline scaffolding, Deep Sets baseline implementation, Canned Demo scripting (`run_demo.sh`), evaluation metrics & ROC plotting, README documentation.
- **Pradipta Sadhukhan (25D1080):** Physics feature definitions & validation, EdgeConv GNN message-passing architecture, training experiments & hyperparameter tuning, LaTeX final report.
- **Joint Work:** Code review, presentation recording (2–3 min video), honor code declaration, and final Moodle submission verification.
