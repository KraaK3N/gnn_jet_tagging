================================================================================
CS6012: CODE-BASED PROJECT SUBMISSION DECLARATION
================================================================================

Project Title: Learned GNN Physics Simulator for Particle Jet Tagging
Group 11:
  1. Rahul Kumar       [Roll Number: 25M2015, SYSCON, IIT Bombay]
  2. Pradipta Sadhukhan [Roll Number: 25D1080, Physics, IIT Bombay]
Course Supervisor: Prof. Sharat Chandran

--------------------------------------------------------------------------------
1. ACADEMIC HONOR CODE PLEDGE
--------------------------------------------------------------------------------
I pledge on my honour that I have not given or received any unauthorized
assistance on this project or any previous task.

Signed by:
  - Rahul Kumar        (Roll Number: 25M2015)
  - Pradipta Sadhukhan (Roll Number: 25D1080)

--------------------------------------------------------------------------------
2. CONSULTED PERSONS / HUMAN RESOURCES
--------------------------------------------------------------------------------
- Course Instructor / Supervisor: Prof. Sharat Chandran
- Team Collaboration: Rahul Kumar and Pradipta Sadhukhan collaborated on the
  data pipeline, GNN message passing formulation, benchmarking, and reporting.

--------------------------------------------------------------------------------
3. NON-HUMAN RESOURCES & LLM USAGE DECLARATION
--------------------------------------------------------------------------------
In accordance with course guidelines, the following tools and non-human resources
were utilized during this project:

a. Large Language Models (LLMs):
   - Antigravity / Gemini 3.8: Used for code scaffolding, vectorization debugging,
     LaTeX template drafting, and metric analysis.
   - ChatGPT / OpenAI: Used for reviewing theoretical physics notation and literature.
   - Reference links:
     * https://chatgpt.com
     * https://deepmind.google/technologies/gemini/

b. Literature & Data Sources:
   - J. Shlomi, P. Battaglia, J.-R. Vlimant, "Graph Neural Networks in Particle Physics",
     arXiv:2007.13681 (2020).
   - H. Qu, L. Gouskos, "Jet Tagging via Particle Clouds", Phys. Rev. D 101, 056019 (2020)
     [arXiv:1902.08570].
   - P. T. Komiske, E. M. Metodiev, J. Thaler, "Energy Flow Networks: Deep Sets for
     Particle Jets", JHEP 01 (2019) 121 [arXiv:1810.05165].
   - PyTorch Documentation: https://pytorch.org/docs/stable/
   - Scikit-Learn Metrics: https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics

--------------------------------------------------------------------------------
4. HOW TO RUN THE CANNED DEMO (TA EVALUATION)
--------------------------------------------------------------------------------
To run the automated evaluation on the test dataset:

  git clone <repo_url>
  cd gnn-jet-tagging
  pip install -r requirements.txt
  ./run_demo.sh

The script loads the pre-trained checkpoints (in `checkpoints/`), evaluates on
the bundled benchmark dataset (`data/sample_jets.npz`), prints an ASCII summary table,
and saves the comparative ROC curve to `results/figures/demo_roc_curve.png`
in under 10 seconds on any standard CPU laptop.
