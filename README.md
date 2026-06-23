# pinn-benchmark-ai-assistants

Benchmark study comparing AI coding assistants on the task of generating
Physics-Informed Neural Network (PINN) pipelines for the 1D Burgers' equation.

Final class project for **IA382A — Seminars in Computer Engineering (1S2026)**,
School of Electrical and Computer Engineering (FEEC), UNICAMP.

---

## Project Overview

This repository documents an evaluation of AI-generated PINN pipelines from a
**data engineering perspective**, across three layers:

| Layer | What is evaluated |
|---|---|
| **Physics Correctness** | PDE residual formulation; convergence vs. analytical baseline |
| **Pipeline Engineering Quality** | Modularity, reproducibility, training loop design |
| **Prompt Efficiency** | Iterations required to reach a functional implementation |

The benchmark task is solving the **1D viscous Burgers' equation**:

```
u_t + u·u_x = ν·u_xx
ν = 0.01/π,  x ∈ [-1, 1],  t ∈ [0, 1]
IC: u(x, 0) = -sin(πx)
BC: u(-1, t) = u(1, t) = 0
```

Inspired by the seminar *"Scientific Machine Learning and Quantum Utility:
A Near Future Perspective"* (FEEC Seminar Series, 2024).

---

## AI Tools Evaluated

| Tool | Category | Role |
|---|---|---|
| **Claude** | Conversational assistant | Primary pipeline generation |
| **Cursor** | IDE agent | File-aware iterative generation |
| **GitHub Copilot** | IDE agent | Within-category comparison with Cursor |
| **IBM Bob** *(conditional)* | Coding assistant (watsonx.ai) | IBM ecosystem evaluation |
| **Perplexity AI** | Search assistant | Literature retrieval, PDE fact-checking |
| **NotebookLM** | Document assistant | Seminar content processing |

---

## Repository Structure

```
pinn-benchmark-ai-assistants/
│
├── claude/
│   ├── iter1/
│   │   ├── burgers_pinn_claude_iter1.py   # Full pipeline, Iteration 1
│   │   └── prompt_iter1.md                # Exact prompt used
│   └── iter2/                             # If refinements were needed
│
├── cursor/
│   ├── iter1/
│   │   ├── burgers_pinn_cursor_iter1.py
│   │   └── prompt_iter1.md
│   └── screenshots/                       # IDE interaction screenshots
│
├── copilot/
│   └── iter1/
│       ├── burgers_pinn_copilot_iter1.py
│       └── prompt_iter1.md
│
├── ibm_bob/                               # Conditional — pending policy check
│
├── outputs/                               # Plots, loss curves, saved models
│   ├── burgers_solution_comparison.png
│   ├── burgers_loss_history.png
│   └── burgers_slices.png
│
├── report/
│   └── burgers_pinn_evaluation_report.pdf
│
├── requirements.txt
└── README.md
```

---

## Quickstart

```bash
git clone https://github.com/<your-username>/pinn-benchmark-ai-assistants.git
cd pinn-benchmark-ai-assistants

pip install -r requirements.txt

# Run Cursor's pipeline (Iteration 1)
python cursor/iter1/burgers_pinn_cursor_iter1.py

# Run Claude's pipeline (Iteration 1)
python claude/iter1/burgers_pinn_claude_iter1.py
```

Outputs (plots + model weights) are saved to `outputs/`.

---

## Requirements

```
torch>=2.0
numpy
scipy
matplotlib
```

---

## Results Summary

| Tool | Rel. L² Error | Final Loss | Prompt Iterations | Notes |
|---|---|---|---|---|
| Claude | 1.06e-01 | 1.78e-04 | 1 | 5000 Adam + 1000 L-BFGS steps |
| Cursor | 1.22e-01 | 1.93e-04 | 1 | 5000 Adam + 1000 L-BFGS steps |
| GitHub Copilot | — | — | — | Pending |
| IBM Bob | — | — | — | Conditional |

---

## Citation

If you use or build on this work, please cite:

```
Batista, C. A. F. (2026). Evaluating AI-Generated Physics-Informed Neural Network
Pipelines: A Data Engineering Perspective on Scientific Machine Learning.
Final Class Project, IA382A — Seminars in Computer Engineering, FEEC/UNICAMP.
```

---

## References

- Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). Physics-informed neural
  networks: A deep learning framework for solving forward and inverse problems
  involving nonlinear partial differential equations. *Journal of Computational
  Physics*, 378, 686–707.
- Karniadakis, G. E., et al. (2021). Physics-informed machine learning.
  *Nature Reviews Physics*, 3(6), 422–440.

---

*Project submitted to FEEC/UNICAMP, June 2026.*
