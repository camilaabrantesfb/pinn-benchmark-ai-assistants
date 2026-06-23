# Prompt — Claude Iteration 1

Generate a complete PyTorch PINN pipeline for the 1D viscous Burgers' equation:

```
u_t + u·u_x = ν·u_xx
ν = 0.01/π,  x ∈ [-1, 1],  t ∈ [0, 1]
IC: u(x, 0) = -sin(πx)
BC: u(-1, t) = u(1, t) = 0
```

Requirements:
- Modular code: separate functions for PDE residual, IC loss, BC loss, training loop, evaluation, and plotting.
- Latin Hypercube Sampling for collocation points, with a fixed random seed for reproducibility.
- Fully-connected network with tanh activations and Xavier initialization.
- Two-phase training: Adam followed by L-BFGS.
- Compute relative L2 error against an analytical reference solution (Cole-Hopf transform).
- Save model weights, loss history (JSON), and diagnostic plots (solution comparison, loss curves, solution slices) to an `outputs/` directory.

No follow-up corrections were required for the physics formulation; the pipeline ran successfully on the first generation.
