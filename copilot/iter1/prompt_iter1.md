# Prompt — GitHub Copilot Iteration 1

Consider the readme file then generate a complete PINN pipeline in PyTorch for the 1D Burgers' equation with ν=0.01/π, domain x∈[-1,1], t∈[0,1], IC: u(x,0)=-sin(πx), BC: u(±1,t)=0. Include PDE residual loss, boundary condition loss, initial condition loss, a two-phase Adam+L-BFGS training loop, and comparison against the analytical Cole-Hopf solution. Output relative L² error.
