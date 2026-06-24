"""
Complete PINN pipeline for the 1D viscous Burgers' equation.

Equation:
    u_t + u u_x = \nu u_xx
    \nu = 0.01 / \pi
    x \in [-1, 1],    t \in [0, 1]

Initial condition:
    u(x, 0) = -sin(\pi x)
Boundary conditions:
    u(-1, t) = 0,   u(1, t) = 0

This script trains a Physics-Informed Neural Network using:
    - PDE residual loss
    - initial condition loss
    - boundary condition loss
    - Adam optimizer followed by L-BFGS

The trained model is compared to the analytical Cole-Hopf solution, and the
relative L2 error is reported.
"""

import json
import os
import time

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as integrate
import torch
import torch.nn as nn
from scipy.stats import qmc
from torch.optim import Adam, LBFGS

# ---------------------------------------------------------------------------
# Reproducibility and device
# ---------------------------------------------------------------------------
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Config] Device: {DEVICE}")

# ---------------------------------------------------------------------------
# Physics configuration
# ---------------------------------------------------------------------------
NU = 0.01 / np.pi
X_LEFT = -1.0
X_RIGHT = 1.0
T_START = 0.0
T_END = 1.0

# ---------------------------------------------------------------------------
# Analytical solution via Cole-Hopf transform
# ---------------------------------------------------------------------------

def analytical_burgers(x, t, nu=NU):
    """Analytical Burgers solution using the Cole-Hopf heat kernel convolution."""
    x = float(x)
    t = float(t)
    if t == 0.0:
        return -np.sin(np.pi * x)

    def phi0(xi):
        integral_u0 = (np.cos(np.pi * xi) - 1.0) / np.pi
        return np.exp(-integral_u0 / (2.0 * nu))

    def integrand_phi(xi):
        return phi0(xi) * np.exp(-((x - xi) ** 2) / (4.0 * nu * t))

    def integrand_dphi(xi):
        return phi0(xi) * (-(x - xi) / (2.0 * nu * t)) * np.exp(-((x - xi) ** 2) / (4.0 * nu * t))

    phi, _ = integrate.quad(integrand_phi, X_LEFT, X_RIGHT, limit=200)
    dphi, _ = integrate.quad(integrand_dphi, X_LEFT, X_RIGHT, limit=200)
    u = -2.0 * nu * dphi / (phi + 1e-12)
    return u


def compute_analytical_grid(x_pts, t_pts):
    X, T = np.meshgrid(x_pts, t_pts, indexing="ij")
    U_exact = np.zeros_like(X)
    for i, xi in enumerate(x_pts):
        for j, tj in enumerate(t_pts):
            U_exact[i, j] = analytical_burgers(xi, tj)
    return X, T, U_exact

# ---------------------------------------------------------------------------
# PINN architecture
# ---------------------------------------------------------------------------

class BurgersPINN(nn.Module):
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
# Loss definitions
# ---------------------------------------------------------------------------

def pde_residual(model, x_col, t_col):
    x_col = x_col.requires_grad_(True)
    t_col = t_col.requires_grad_(True)
    u = model(x_col, t_col)

    u_x, u_t = torch.autograd.grad(
        u,
        [x_col, t_col],
        grad_outputs=torch.ones_like(u),
        create_graph=True,
    )

    u_xx = torch.autograd.grad(
        u_x,
        x_col,
        grad_outputs=torch.ones_like(u_x),
        create_graph=True,
    )[0]

    return u_t + u * u_x - NU * u_xx


def loss_pde(model, x_col, t_col):
    r = pde_residual(model, x_col, t_col)
    return torch.mean(r ** 2)


def loss_ic(model, x_ic, t_ic, u_ic):
    u_pred = model(x_ic, t_ic)
    return torch.mean((u_pred - u_ic) ** 2)


def loss_bc(model, x_bc, t_bc, u_bc):
    u_pred = model(x_bc, t_bc)
    return torch.mean((u_pred - u_bc) ** 2)


def total_loss(model, x_col, t_col, x_ic, t_ic, u_ic, x_bc, t_bc, u_bc,
               w_pde=1.0, w_ic=1.0, w_bc=1.0):
    L_pde = loss_pde(model, x_col, t_col)
    L_ic = loss_ic(model, x_ic, t_ic, u_ic)
    L_bc = loss_bc(model, x_bc, t_bc, u_bc)
    L_total = w_pde * L_pde + w_ic * L_ic + w_bc * L_bc
    return L_total, L_pde, L_ic, L_bc

# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

def latin_hypercube(n_dims, n_samples, seed=SEED):
    sampler = qmc.LatinHypercube(d=n_dims, seed=seed)
    return sampler.random(n=n_samples)


def generate_training_data(n_collocation=2000, n_ic=200, n_bc=200, device=DEVICE):
    collocation = latin_hypercube(2, n_collocation)
    x_col = collocation[:, 0:1] * (X_RIGHT - X_LEFT) + X_LEFT
    t_col = collocation[:, 1:2] * (T_END - T_START) + T_START

    x_ic = np.linspace(X_LEFT, X_RIGHT, n_ic).reshape(-1, 1)
    t_ic = np.zeros_like(x_ic)
    u_ic = -np.sin(np.pi * x_ic)

    t_bc = np.random.uniform(T_START, T_END, (n_bc, 1))
    x_bc = np.vstack([
        np.full((n_bc // 2, 1), X_LEFT),
        np.full((n_bc - n_bc // 2, 1), X_RIGHT),
    ])
    u_bc = np.zeros((n_bc, 1))

    def to_tensor(arr):
        return torch.tensor(arr, dtype=torch.float32, device=device)

    return (
        to_tensor(x_col),
        to_tensor(t_col),
        to_tensor(x_ic),
        to_tensor(t_ic),
        to_tensor(u_ic),
        to_tensor(x_bc),
        to_tensor(t_bc),
        to_tensor(u_bc),
    )

# ---------------------------------------------------------------------------
# Training pipeline
# ---------------------------------------------------------------------------

def train(model, data, n_adam=5000, n_lbfgs=500):
    x_col, t_col, x_ic, t_ic, u_ic, x_bc, t_bc, u_bc = data
    optimizer = Adam(model.parameters(), lr=1e-3)
    loss_history = {
        "adam": [],
        "lbfgs": [],
    }

    model.train()
    for step in range(1, n_adam + 1):
        optimizer.zero_grad()
        L_total, L_pde, L_ic, L_bc = total_loss(
            model, x_col, t_col, x_ic, t_ic, u_ic, x_bc, t_bc, u_bc
        )
        L_total.backward()
        optimizer.step()

        if step % 250 == 0 or step == 1:
            loss_history["adam"].append({
                "step": step,
                "total": L_total.item(),
                "pde": L_pde.item(),
                "ic": L_ic.item(),
                "bc": L_bc.item(),
            })
            print(
                f"Adam step {step:5d} | total={L_total.item():.4e} "
                f"pde={L_pde.item():.4e} ic={L_ic.item():.4e} bc={L_bc.item():.4e}"
            )

    lbfgs_optimizer = LBFGS(model.parameters(), max_iter=n_lbfgs, tolerance_grad=1e-9,
                            tolerance_change=1e-12, history_size=50, line_search_fn="strong_wolfe")

    def closure():
        lbfgs_optimizer.zero_grad()
        L_total, L_pde, L_ic, L_bc = total_loss(
            model, x_col, t_col, x_ic, t_ic, u_ic, x_bc, t_bc, u_bc
        )
        L_total.backward(retain_graph=True)
        loss_history["lbfgs"].append({
            "step": len(loss_history["lbfgs"]) + 1,
            "total": L_total.item(),
            "pde": L_pde.item(),
            "ic": L_ic.item(),
            "bc": L_bc.item(),
        })
        if len(loss_history["lbfgs"]) % 50 == 0:
            print(
                f"L-BFGS iter {len(loss_history['lbfgs']):4d} | "
                f"total={L_total.item():.4e} pde={L_pde.item():.4e} "
                f"ic={L_ic.item():.4e} bc={L_bc.item():.4e}"
            )
        return L_total

    print("Starting L-BFGS optimization...")
    lbfgs_optimizer.step(closure)
    return loss_history

# ---------------------------------------------------------------------------
# Evaluation utilities
# ---------------------------------------------------------------------------

def evaluate_model(model, x_pts, t_pts):
    model.eval()
    X, T = np.meshgrid(x_pts, t_pts, indexing="ij")
    X_flat = X.reshape(-1, 1)
    T_flat = T.reshape(-1, 1)

    with torch.no_grad():
        x_tensor = torch.tensor(X_flat, dtype=torch.float32, device=DEVICE)
        t_tensor = torch.tensor(T_flat, dtype=torch.float32, device=DEVICE)
        u_pred = model(x_tensor, t_tensor).cpu().numpy().reshape(X.shape)

    _, _, u_exact = compute_analytical_grid(x_pts, t_pts)
    error = u_pred - u_exact
    rel_l2 = np.linalg.norm(error) / np.linalg.norm(u_exact)
    return X, T, u_pred, u_exact, error, rel_l2


def plot_results(x_pts, t_pts, u_pred, u_exact, rel_l2, loss_history, outdir="outputs"):
    os.makedirs(outdir, exist_ok=True)
    X, T = np.meshgrid(x_pts, t_pts, indexing="ij")

    fig = plt.figure(figsize=(12, 8))
    gs = gridspec.GridSpec(2, 2, figure=fig, width_ratios=[1, 1], height_ratios=[1, 1])

    ax0 = fig.add_subplot(gs[0, 0])
    c0 = ax0.pcolormesh(T, X, u_exact, shading="auto", cmap="viridis")
    fig.colorbar(c0, ax=ax0)
    ax0.set_title("Analytical solution")
    ax0.set_xlabel("t")
    ax0.set_ylabel("x")

    ax1 = fig.add_subplot(gs[0, 1])
    c1 = ax1.pcolormesh(T, X, u_pred, shading="auto", cmap="viridis")
    fig.colorbar(c1, ax=ax1)
    ax1.set_title("PINN prediction")
    ax1.set_xlabel("t")

    ax2 = fig.add_subplot(gs[1, :])
    adam_steps = [entry["step"] for entry in loss_history["adam"]]
    adam_total = [entry["total"] for entry in loss_history["adam"]]
    lbfgs_steps = [entry["step"] for entry in loss_history["lbfgs"]]
    lbfgs_total = [entry["total"] for entry in loss_history["lbfgs"]]
    ax2.semilogy(adam_steps, adam_total, "-o", label="Adam")
    ax2.semilogy([s + adam_steps[-1] for s in lbfgs_steps], lbfgs_total, "-s", label="L-BFGS")
    ax2.set_xlabel("iteration")
    ax2.set_ylabel("total loss")
    ax2.set_title(f"Training loss (relative L2 error = {rel_l2:.3e})")
    ax2.legend()
    ax2.grid(True)

    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "burgers_pinn_solution_and_loss.png"), dpi=200)
    plt.close(fig)

    stats = {
        "rel_l2_error": float(rel_l2),
        "adam_loss_final": float(adam_total[-1]) if adam_total else None,
        "lbfgs_loss_final": float(lbfgs_total[-1]) if lbfgs_total else None,
    }
    with open(os.path.join(outdir, "burgers_pinn_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)


def plot_comparison(x_pts, t_pts, u_exact, u_pred, u_err, rel_l2, save_dir="outputs"):
    """Generate a 3-panel comparison plot: analytical, PINN prediction, and error."""
    os.makedirs(save_dir, exist_ok=True)

    fig = plt.figure(figsize=(15, 5))
    gs = gridspec.GridSpec(1, 3, figure=fig)

    titles = ["Analytical Solution", "PINN Prediction", "Absolute Error"]
    data = [u_exact.T, u_pred.T, u_err.T]
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
    plt.savefig(f"{save_dir}/burgers_solution_comparison_copilot.png", dpi=150)
    plt.close()
    print(f"[Plot] Saved solution comparison → {save_dir}/burgers_solution_comparison_copilot.png")

# ---------------------------------------------------------------------------
# Main execution
# ---------------------------------------------------------------------------

def main():
    model = BurgersPINN().to(DEVICE)
    data = generate_training_data(n_collocation=3000, n_ic=200, n_bc=200, device=DEVICE)

    start_time = time.time()
    loss_history = train(model, data, n_adam=5000, n_lbfgs=500)
    elapsed = time.time() - start_time
    print(f"Training completed in {elapsed:.1f} seconds.")

    x_eval = np.linspace(X_LEFT, X_RIGHT, 100)
    t_eval = np.linspace(T_START, T_END, 100)
    _, _, u_pred, u_exact, u_err, rel_l2 = evaluate_model(model, x_eval, t_eval)
    print(f"Relative L2 error versus analytical Cole-Hopf solution: {rel_l2:.6e}")

    plot_results(x_eval, t_eval, u_pred, u_exact, rel_l2, loss_history)
    plot_comparison(x_eval, t_eval, u_exact, u_pred, u_err, rel_l2)
    torch.save(model.state_dict(), os.path.join("outputs", "burgers_pinn_copilot_iter1.pth"))
    print("Saved model and plots to outputs/.")


if __name__ == "__main__":
    main()
