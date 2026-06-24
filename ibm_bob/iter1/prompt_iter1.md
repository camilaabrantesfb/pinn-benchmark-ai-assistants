# Prompt for IBM Bob - Iteration 1

## Task Description

Generate a complete PINN pipeline in PyTorch for the 1D Burgers' equation with the following specifications:

### Problem Setup
- **PDE**: u_t + u·u_x = ν·u_xx (1D viscous Burgers' equation)
- **Viscosity**: ν = 0.01/π
- **Domain**: x ∈ [-1, 1], t ∈ [0, 1]
- **Initial Condition**: u(x, 0) = -sin(πx)
- **Boundary Conditions**: u(-1, t) = 0, u(1, t) = 0

### Requirements

1. **Loss Components**:
   - PDE residual loss: u_t + u·u_x - ν·u_xx = 0
   - Initial condition loss: u(x, 0) = -sin(πx)
   - Boundary condition loss: u(±1, t) = 0

2. **Training Strategy**:
   - Two-phase optimization:
     - Phase 1: Adam optimizer (5000 epochs)
     - Phase 2: L-BFGS optimizer (1000 iterations)

3. **Validation**:
   - Compare against analytical Cole-Hopf solution
   - Output relative L² error

4. **Outputs**:
   - Solution comparison plots (PINN vs analytical)
   - Time slice comparisons
   - Loss history plots
   - Model weights
   - Training statistics (JSON)

### Expected Deliverables

- Complete, runnable Python script
- Proper documentation and comments
- Reproducible results with fixed random seeds
- Professional visualization
- Performance metrics

## Context

This is part of a benchmark study comparing AI coding assistants on generating Physics-Informed Neural Network pipelines. The implementation should follow best practices in scientific machine learning and be comparable to other AI-generated solutions (Claude, Cursor, GitHub Copilot).

## Reference

Based on the project structure defined in `README.md` of the `pinn-benchmark-ai-assistants` repository.