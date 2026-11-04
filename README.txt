================================================================================
CS6012: CODE-BASED PROJECT SUBMISSION
Learned GNN Physics Simulator for Particle Jet Tagging
================================================================================

Course:        CS6012 - Advanced Computer Science Project
Institution:   Indian Institute of Technology Bombay (IIT Bombay)
Instructor:    Prof. Sharat Chandran

Group:         Group 11
Team Members:
  1. Rahul Kumar         (Roll No: 25M2015) - Systems and Control Engineering
  2. Pradipta Sadhukhan  (Roll No: 25D1080) - Department of Physics

Repository:    https://github.com/KraaK3N/gnn_jet_tagging
Git Clone SSH: git@github.com:KraaK3N/gnn_jet_tagging.git


================================================================================
1. ACADEMIC HONOR CODE DECLARATION
================================================================================

I pledge on my honour that I have not given or received any unauthorized 
assistance on this assignment or any previous task.

We have maintained full academic integrity throughout this work:
1. All baseline algorithms, datasets, and review papers have been formally 
   attributed and cited (specifically, the foundational review paper:
   "Graph Neural Networks in Particle Physics" by J. Shlomi, P. Battaglia, 
   and J.-R. Vlimant, arXiv:2007.13681).
2. The dataset originates from the official Open Science Pythia 8.2 Collider 
   dataset deposited on Zenodo (DOI: 10.5281/zenodo.3164691) by P. Komiske, 
   E. Metodiev, and J. Thaler.
3. We have strictly adhered to the IIT Bombay Honor Code and Course Policy.

Signed:
  Rahul Kumar         (Roll: 25M2015)
  Pradipta Sadhukhan  (Roll: 25D1080)

================================================================================
2. DISCLOSURE OF AI / LLM ASSISTANCE & CHAT TRANSCRIPTS
================================================================================

In accordance with course guidelines regarding transparency in modern computational
and AI-assisted workflows, we disclose the use of Large Language Model (LLM) 
pair-programming tools during the execution of this project.

LLM Tools Employed:
- Google DeepMind Antigravity Coding Assistant (Gemini 3.8 flash reasoning model)
- OpenAI ChatGPT / Anthropic Claude for LaTeX formatting and conceptual brainstorming
- All this were use in the CLI interface mode.

LLM Web / Tool References:
- Antigravity / Gemini: https://gemini.google.com/
- OpenAI ChatGPT:       https://chatgpt.com/
- Anthropic Claude:     https://claude.ai/

Tasks Assisted by LLMs:
- Assisting in the vectorized tensor broadcasting math for pure-PyTorch dynamic 
  k-NN graph construction (avoiding fragile C++ CUDA extension build issues).
- Reviewing LaTeX syntax for the double-column conference report and Beamer deck.
- Designing the zero-dependency canned demonstration shell harness.



================================================================================
3. CANNED DEMO FOR TEACHING ASSISTANTS (MINIMUM EFFORT INSTRUCTIONS)
================================================================================

To allow the teaching team to evaluate this project with zero hassle, a fully 
automated canned demo script is provided. It runs completely on CPU in under 
10 seconds and requires no GPU, no build tools, and no data downloads.

Quick Start Command:
  $ cd code_based_project/gnn-jet-tagging
  $ ./run_demo.sh

What the Canned Demo Does:
1. Automatically verifies Python environment and loads pre-packaged PyTorch models.
2. Loads the bundled 1,000-jet benchmark dataset (data/sample_jets.npz).
3. Executes comparative inference between Deep Sets (No-Edges Baseline) and 
   Dynamic EdgeConv GNN (ParticleNet-lite).
4. Prints a formatted ASCII comparison table showing:
   - Test Classification Accuracy
   - Area Under ROC Curve (ROC-AUC)
   - Background Gluon Rejection at 50% Quark Efficiency (1 / eps_B)
   - Per-jet CPU Inference Latency (ms)
   - Parameter Count
5. Saves a publication-quality comparative ROC curve to:
   results/figures/demo_roc_curve.png


================================================================================
4. FULL 100,000-JET PYTHIA BENCHMARK ON GOOGLE COLAB (T4 GPU)
================================================================================

For full-scale evaluation on the official Pythia 8.2 Collider dataset (100,000 jets):
- Notebook: notebooks/gnn.ipynb
- Connect to Google Colab with a free T4 GPU runtime.
- Run all cells sequentially:
  Cell 1: GPU verification (Tesla T4, 16 GB VRAM)
  Cell 2: Automatic workspace directory resolution
  Cell 3: Zenodo Pythia dataset fetch & 7-feature constituent preprocessing
  Cell 4: Deep Sets Baseline training (15 epochs, ~64s on GPU)
  Cell 5: Dynamic EdgeConv GNN training (15 epochs, ~16m on GPU)
  Cell 6: Comparative evaluation on 15,000 held-out Pythia test jets
  Cell 7: Instant canned demo verification


================================================================================
5. REPOSITORY DIRECTORY LAYOUT
================================================================================

gnn-jet-tagging/
├── README.txt                 <-- This Honor Code & Course Declaration file
├── README.md                  <-- Complete project documentation and math guide
├── run_demo.sh                <-- Canned demo executable (chmod +x)
├── demo.py                    <-- Standalone TA grading script
├── configs/                   <-- Hyperparameter configs (baseline.yaml, gnn.yaml)
├── data/
│   ├── sample_jets.npz        <-- Bundled 1k verification slice for instant TA demo
│   └── raw/                   <-- Download destination for Zenodo QG_jets.npz
├── src/
│   ├── data/                  <-- JetDataset, constituent preprocessor, Pythia loader
│   ├── models/                <-- DeepSetsJetTagger and ParticleNetLite (EdgeConv)
│   ├── training/              <-- Training loop, CosineAnnealing/StepLR, Focal/BCE loss
│   ├── evaluation/            <-- Metrics (Acc, AUC, R50, R30) and evaluation harness
│   └── utils/                 <-- Configuration loader, plotting, reproducibility seeds
├── tests/
│   └── test_graph_building.py <-- 5 unit tests (permutation invariance, coordinate wrapping)
├── checkpoints/               <-- Pre-trained PyTorch model weights (*_best.pt)
├── results/
│   ├── figures/               <-- Generated ROC curves and training histories
│   └── metrics_summary.json   <-- Exact quantitative test benchmarks
├── report/
│   ├── final_report.tex       <-- Academic project report in LaTeX
│   └── final_report.pdf       <-- Compiled 4-page double-column PDF report
└── notebooks/
    └── gnn.ipynb              <-- Interactive Colab T4 GPU notebook
================================================================================
