"""
=============================================================================
PINN PIPELINE — 1D Burgers' Equation
Cursor Iteration 1 | IA382A Final Project
=============================================================================

PHYSICS ASSUMPTIONS (explicitly stated for evaluation layer 1):
  PDE   : u_t + u·u_x = ν·u_xx   (viscous Burgers' equation)
  Domain: x ∈ [-1, 1],  t ∈ [0, 1]
  ν     : 0.01/π  ≈ 0.003183  (standard benchmark value, Raissi et al. 2019)
  IC    : u(x, 0) = -sin(πx)
  BC    : u(-1, t) = 0,  u(1, t) = 0  (Dirichlet, homogeneous)
  Analytical solution: Cole-Hopf transform via heat-kernel convolution

DESIGN DECISIONS (for evaluation layer 2 — pipeline engineering quality):
  • Separate functions for each loss term → modularity, easy to swap/ablate
  • Collocation points sampled with Latin Hypercube Sampling (LHS) → better
    space coverage than uniform random; reproducible via fixed seed
  • Xavier initialization → avoids vanishing/exploding gradients in tanh nets
  • Adam first, then L-BFGS → common two-phase strategy in PINN literature;
    Adam handles early rough convergence, L-BFGS polishes
  • Loss history logged every epoch → full reproducibility of training curve
  • Model, config, and loss history saved to disk → pipeline reproducibility

PROMPT ENGINEERING NOTE:
  This is Cursor Iteration 1 — generated from a single structured prompt
  referencing the repository README and specifying physics parameters,
  architecture, and evaluation criteria.
=============================================================================
"""

import json
import os

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from scipy import integrate
from scipy.stats import qmc
from torch.optim import Adam, LBFGS

# ---------------------------------------------------------------------------
# 0. Reproducibility
# ---------------------------------------------------------------------------
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Config] Device: {DEVICE}")

# ---------------------------------------------------------------------------
# 1. Physics Configuration
# ---------------------------------------------------------------------------
NU = 0.01 / np.pi
X_LEFT, X_RIGHT = -1.0, 1.0
T_START, T_END = 0.0, 1.0


def lhs(n_dims, n_samples, seed=SEED):
    """Latin Hypercube Sampling in [0, 1]^n_dims."""
    sampler = qmc.LatinHypercube(d=n_dims, seed=seed)
    return sampler.random(n=n_samples)


# ---------------------------------------------------------------------------
# 2. Analytical Solution (Cole-Hopf transform)
#    u = -2ν (∂φ/∂x) / φ,  where φ solves the heat equation φ_t = ν φ_xx
# ---------------------------------------------------------------------------

def _initial_potential(xi, nu=NU):
    """φ(x, 0) = exp(-1/(2ν) ∫_0^x u(s, 0) ds) for u(x, 0) = -sin(πx)."""
    integral_u0 = (np.cos(np.pi * xi) - 1.0) / np.pi
    return np.exp(-integral_u0 / (2.0 * nu))


def analytical_burgers(x, t, nu=NU):
    """
    Cole-Hopf solution evaluated at a single (x, t) point.

    For t = 0 the initial condition is returned exactly.
    For t > 0 the potential is obtained by convolving φ_0 with the heat kernel.
    """
    if t == 0.0:
        return -np.sin(np.pi * x)

    def integrand_phi(xi):
        phi0 = _initial_potential(xi, nu)
        kernel = np.exp(-((x - xi) ** 2) / (4.0 * nu * t))
        return phi0 * kernel

    def integrand_dphi(xi):
        phi0 = _initial_potential(xi, nu)
        kernel = np.exp(-((x - xi) ** 2) / (4.0 * nu * t))
        d_kernel = -(x - xi) / (2.0 * nu * t) * kernel
        return phi0 * d_kernel

    phi, _ = integrate.quad(integrand_phi, X_LEFT, X_RIGHT, limit=200)
    dphi, _ = integrate.quad(integrand_dphi, X_LEFT, X_RIGHT, limit=200)

    return -2.0 * nu * dphi / (phi + 1e-10)


def compute_analytical_grid(x_pts, t_pts):
    """Analytical solution on a structured (x, t) mesh."""
    X, T = np.meshgrid(x_pts, t_pts, indexing="ij")
    U_exact = np.zeros_like(X)
    for i, xi in enumerate(x_pts):
        for j, tj in enumerate(t_pts):
            U_exact[i, j] = analytical_burgers(xi, tj)
    return X, T, U_exact


# ---------------------------------------------------------------------------
# 3. Neural Network Architecture
# ---------------------------------------------------------------------------

class BurgersPINN(nn.Module):
    """Fully-connected network approximating u(x, t)."""

    def __init__(self, layers=None):
        super().__init__()
        if layers is None:
            layers = [2, 20, 20, 20, 20, 1]

        modules = []
        for i in range(len(layers) - 1):
            modules.append(nn.Linear(layers[i], layers[i + 1]))
            if i < len(layers) - 2:
                modules.append(nn.Tanh())
        self.net = nn.Sequential(*modules)
        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.net.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x, t):
        inputs = torch.cat([x, t], dim=1)
        return self.net(inputs)


# ---------------------------------------------------------------------------
# 4. Loss Functions (Layer 1 — Physics Correctness)
# ---------------------------------------------------------------------------

def pde_residual(model, x_col, t_col):
    """PDE residual r = u_t + u·u_x - ν·u_xx via automatic differentiation."""
    x_col = x_col.requires_grad_(True)
    t_col = t_col.requires_grad_(True)

    u = model(x_col, t_col)

    grads = torch.autograd.grad(
        u,
        [x_col, t_col],
        grad_outputs=torch.ones_like(u),
        create_graph=True,
    )
    u_x = grads[0]
    u_t = grads[1]

    u_xx = torch.autograd.grad(
        u_x,
        x_col,
        grad_outputs=torch.ones_like(u_x),
        create_graph=True,
    )[0]

    return u_t + u * u_x - NU * u_xx


def loss_pde(model, x_col, t_col):
    """Mean squared PDE residual over collocation points."""
    residual = pde_residual(model, x_col, t_col)
    return torch.mean(residual ** 2)


def loss_ic(model, x_ic, t_ic, u_ic):
    """Initial condition loss: u(x, 0) = -sin(πx)."""
    u_pred = model(x_ic, t_ic)
    return torch.mean((u_pred - u_ic) ** 2)


def loss_bc(model, x_bc, t_bc, u_bc):
    """Boundary condition loss: u(-1, t) = u(1, t) = 0."""
    u_pred = model(x_bc, t_bc)
    return torch.mean((u_pred - u_bc) ** 2)


def total_loss(
    model,
    x_col,
    t_col,
    x_ic,
    t_ic,
    u_ic,
    x_bc,
    t_bc,
    u_bc,
    w_pde=1.0,
    w_ic=1.0,
    w_bc=1.0,
):
    """Composite loss L = w_pde·L_pde + w_ic·L_ic + w_bc·L_bc."""
    L_pde = loss_pde(model, x_col, t_col)
    L_ic = loss_ic(model, x_ic, t_ic, u_ic)
    L_bc = loss_bc(model, x_bc, t_bc, u_bc)
    L_total = w_pde * L_pde + w_ic * L_ic + w_bc * L_bc
    return L_total, L_pde, L_ic, L_bc


# ---------------------------------------------------------------------------
# 5. Data Generation (Layer 2 — Reproducibility)
# ---------------------------------------------------------------------------

def generate_training_data(
    n_collocation=2000,
    n_ic=200,
    n_bc=200,
    device=DEVICE,
):
    """Generate collocation, initial-condition, and boundary-condition samples."""
    lhs_samples = lhs(2, n_collocation)
    x_col = lhs_samples[:, 0:1] * 2.0 - 1.0
    t_col = lhs_samples[:, 1:2]

    x_ic_np = np.linspace(X_LEFT, X_RIGHT, n_ic).reshape(-1, 1)
    t_ic_np = np.zeros_like(x_ic_np)
    u_ic_np = -np.sin(np.pi * x_ic_np)

    t_bc_np = np.random.uniform(T_START, T_END, (n_bc, 1))
    x_left = np.full((n_bc // 2, 1), X_LEFT)
    x_right = np.full((n_bc - n_bc // 2, 1), X_RIGHT)
    x_bc_np = np.vstack([x_left, x_right])
    u_bc_np = np.zeros((n_bc, 1))

    def to_tensor(arr):
        return torch.tensor(arr, dtype=torch.float32, device=device)

    return (
        to_tensor(x_col),
        to_tensor(t_col),
        to_tensor(x_ic_np),
        to_tensor(t_ic_np),
        to_tensor(u_ic_np),
        to_tensor(x_bc_np),
        to_tensor(t_bc_np),
        to_tensor(u_bc_np),
    )


# ---------------------------------------------------------------------------
# 6. Training Loop (Layer 2 — Training Loop Design)
# ---------------------------------------------------------------------------

def train_pinn(
    model,
    x_col,
    t_col,
    x_ic,
    t_ic,
    u_ic,
    x_bc,
    t_bc,
    u_bc,
    n_adam=5000,
    n_lbfgs=1000,
    lr_adam=1e-3,
    log_every=500,
):
    """
    Two-phase training:
      Phase 1 — Adam (n_adam steps): fast convergence from random init
      Phase 2 — L-BFGS (n_lbfgs steps): second-order polishing for low residual
    """
    history = {"total": [], "pde": [], "ic": [], "bc": [], "phase": []}

    optimizer_adam = Adam(model.parameters(), lr=lr_adam)
    print(f"\n[Training] Phase 1: Adam  ({n_adam} iterations)")

    for step in range(1, n_adam + 1):
        optimizer_adam.zero_grad()
        L, L_pde, L_ic, L_bc = total_loss(
            model, x_col, t_col, x_ic, t_ic, u_ic, x_bc, t_bc, u_bc
        )
        L.backward()
        optimizer_adam.step()

        history["total"].append(L.item())
        history["pde"].append(L_pde.item())
        history["ic"].append(L_ic.item())
        history["bc"].append(L_bc.item())
        history["phase"].append("adam")

        if step % log_every == 0 or step == 1:
            print(
                f"  Step {step:5d} | Total: {L.item():.4e} | "
                f"PDE: {L_pde.item():.4e} | IC: {L_ic.item():.4e} | "
                f"BC: {L_bc.item():.4e}"
            )

    print(f"\n[Training] Phase 2: L-BFGS  ({n_lbfgs} iterations)")
    optimizer_lbfgs = LBFGS(
        model.parameters(),
        lr=1.0,
        max_iter=n_lbfgs,
        tolerance_grad=1e-9,
        tolerance_change=1e-12,
        history_size=50,
        line_search_fn="strong_wolfe",
    )

    lbfgs_step = [0]

    def closure():
        optimizer_lbfgs.zero_grad()
        L, L_pde, L_ic, L_bc = total_loss(
            model, x_col, t_col, x_ic, t_ic, u_ic, x_bc, t_bc, u_bc
        )
        L.backward()

        history["total"].append(L.item())
        history["pde"].append(L_pde.item())
        history["ic"].append(L_ic.item())
        history["bc"].append(L_bc.item())
        history["phase"].append("lbfgs")

        lbfgs_step[0] += 1
        if lbfgs_step[0] % 100 == 0:
            print(
                f"  Step {lbfgs_step[0]:5d} | Total: {L.item():.4e} | "
                f"PDE: {L_pde.item():.4e} | IC: {L_ic.item():.4e} | "
                f"BC: {L_bc.item():.4e}"
            )
        return L

    optimizer_lbfgs.step(closure)
    return history


# ---------------------------------------------------------------------------
# 7. Evaluation (Layer 1 — Convergence vs. Analytical Baseline)
# ---------------------------------------------------------------------------

def evaluate_model(model, n_x=256, n_t=100, device=DEVICE):
    """Evaluate relative L² error against the Cole-Hopf analytical solution."""
    x_pts = np.linspace(X_LEFT, X_RIGHT, n_x)
    t_pts = np.linspace(T_START, T_END, n_t)

    print("\n[Evaluation] Computing analytical solution on fine mesh...")
    X_grid, T_grid, U_exact = compute_analytical_grid(x_pts, t_pts)

    x_flat = torch.tensor(
        X_grid.flatten()[:, None], dtype=torch.float32, device=device
    )
    t_flat = torch.tensor(
        T_grid.flatten()[:, None], dtype=torch.float32, device=device
    )

    with torch.no_grad():
        u_pred_flat = model(x_flat, t_flat).cpu().numpy().flatten()

    U_pred = u_pred_flat.reshape(n_x, n_t)
    U_err = np.abs(U_pred - U_exact)

    rel_l2 = np.linalg.norm(U_pred - U_exact) / (np.linalg.norm(U_exact) + 1e-10)
    print(f"[Evaluation] Relative L2 Error: {rel_l2:.6e}")

    return x_pts, t_pts, X_grid, T_grid, U_exact, U_pred, U_err, rel_l2


# ---------------------------------------------------------------------------
# 8. Visualization
# ---------------------------------------------------------------------------

def plot_results(
    x_pts,
    t_pts,
    U_exact,
    U_pred,
    U_err,
    history,
    rel_l2,
    save_dir="./outputs",
):
    os.makedirs(save_dir, exist_ok=True)

    fig = plt.figure(figsize=(15, 5))
    gs = gridspec.GridSpec(1, 3, figure=fig)

    titles = ["Analytical Solution", "PINN Prediction", "Absolute Error"]
    data = [U_exact.T, U_pred.T, U_err.T]
    cmaps = ["RdBu_r", "RdBu_r", "hot_r"]

    for idx, (field, title, cmap) in enumerate(zip(data, titles, cmaps)):
        ax = fig.add_subplot(gs[idx])
        im = ax.pcolormesh(x_pts, t_pts, field, cmap=cmap, shading="auto")
        plt.colorbar(im, ax=ax)
        ax.set_xlabel("x")
        ax.set_ylabel("t")
        ax.set_title(title)

    fig.suptitle(
        f"1D Burgers' PINN (ν={NU:.5f}) — Rel. L² Error: {rel_l2:.4e}",
        fontsize=12,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(f"{save_dir}/burgers_solution_comparison_cursor.png", dpi=150)
    plt.close()
    print(f"[Plot] Saved solution comparison → {save_dir}/burgers_solution_comparison_cursor.png")

    fig, ax = plt.subplots(figsize=(10, 4))
    steps = np.arange(len(history["total"]))
    ax.semilogy(steps, history["total"], label="Total", lw=1.5)
    ax.semilogy(steps, history["pde"], label="PDE", lw=1.0, ls="--")
    ax.semilogy(steps, history["ic"], label="IC", lw=1.0, ls=":")
    ax.semilogy(steps, history["bc"], label="BC", lw=1.0, ls="-.")

    phase_arr = np.array(history["phase"])
    switch_idx = np.where(phase_arr == "lbfgs")[0]
    if len(switch_idx) > 0:
        ax.axvline(
            switch_idx[0],
            color="gray",
            ls="--",
            alpha=0.5,
            label=f"L-BFGS start (step {switch_idx[0]})",
        )

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss (log scale)")
    ax.set_title("Training Loss History — Cursor Iteration 1")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{save_dir}/burgers_loss_history_cursor.png", dpi=150)
    plt.close()
    print(f"[Plot] Saved loss history → {save_dir}/burgers_loss_history_cursor.png")

    t_slices = [0.25, 0.50, 0.75]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    t_idx = [np.argmin(np.abs(t_pts - ts)) for ts in t_slices]

    for ax, ti, ts in zip(axes, t_idx, t_slices):
        ax.plot(x_pts, U_exact[:, ti], "k-", lw=2, label="Analytical")
        ax.plot(x_pts, U_pred[:, ti], "r--", lw=1.5, label="PINN")
        ax.set_title(f"t = {ts}")
        ax.set_xlabel("x")
        ax.set_ylabel("u(x,t)")
        ax.legend()

    fig.suptitle("Solution Slices — Analytical vs. PINN", fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/burgers_slices_cursor.png", dpi=150)
    plt.close()
    print(f"[Plot] Saved slice comparison → {save_dir}/burgers_slices_cursor.png")


# ---------------------------------------------------------------------------
# 9. Main Entry Point
# ---------------------------------------------------------------------------

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
    output_dir = os.path.join(repo_root, "outputs")

    print("=" * 60)
    print("  PINN for 1D Burgers' Equation — Cursor Iteration 1")
    print(f"  nu = {NU:.6f},  Domain: x in [-1,1], t in [0,1]")
    print("  IC: u(x,0) = -sin(pi*x)")
    print("  BC: u(-1,t) = u(1,t) = 0")
    print("=" * 60)

    print("\n[Data] Generating training data...")
    (
        x_col,
        t_col,
        x_ic,
        t_ic,
        u_ic,
        x_bc,
        t_bc,
        u_bc,
    ) = generate_training_data(n_collocation=2000, n_ic=200, n_bc=200)
    print(f"  Collocation pts : {x_col.shape[0]}")
    print(f"  IC pts          : {x_ic.shape[0]}")
    print(f"  BC pts          : {x_bc.shape[0]}")

    model = BurgersPINN(layers=[2, 20, 20, 20, 20, 1]).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n[Model] Parameters: {n_params}")

    history = train_pinn(
        model,
        x_col,
        t_col,
        x_ic,
        t_ic,
        u_ic,
        x_bc,
        t_bc,
        u_bc,
        n_adam=5000,
        n_lbfgs=1000,
        lr_adam=1e-3,
        log_every=500,
    )

    x_pts, t_pts, _, _, U_exact, U_pred, U_err, rel_l2 = evaluate_model(model)

    os.makedirs(output_dir, exist_ok=True)
    torch.save(
        model.state_dict(),
        os.path.join(output_dir, "burgers_pinn_cursor_iter1.pt"),
    )
    with open(os.path.join(output_dir, "loss_history_cursor_iter1.json"), "w") as f:
        json.dump(history, f)
    print("[Save] Model weights and loss history saved.")

    plot_results(
        x_pts, t_pts, U_exact, U_pred, U_err, history, rel_l2, save_dir=output_dir
    )

    print("\n" + "=" * 60)
    print("  EVALUATION SUMMARY (Cursor Iteration 1)")
    print("=" * 60)
    print(f"  Relative L2 Error  : {rel_l2:.4e}")
    print(f"  Total training steps: {len(history['total'])}")
    print(f"  Final total loss   : {history['total'][-1]:.4e}")
    print(f"  Final PDE loss     : {history['pde'][-1]:.4e}")
    print(f"  Final IC  loss     : {history['ic'][-1]:.4e}")
    print(f"  Final BC  loss     : {history['bc'][-1]:.4e}")
    print("=" * 60)

    return rel_l2, history


if __name__ == "__main__":
    main()
