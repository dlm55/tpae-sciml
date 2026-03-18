"""
part3_pinn/pinn_inverse.py
===========================

Part 3: Physics-Informed Neural Network — Inverse Problem

THE SCIENTIFIC QUESTION
-----------------------
Suppose you're in the lab and you've just done a new IP synthesis. You measure
polymer conversion X(t) at a few time points using UV-VIS absorbance — but you
DON'T know the rate constant k. Maybe it changed because you used a different
solvent, or a different PTMO batch, or a different temperature.

Can you RECOVER k just from those sparse noisy observations?

This is the INVERSE PROBLEM: given output data, infer unknown parameters.
Classical approach: nonlinear least squares (requires good initial guess, can
get stuck in local minima).
PINN approach: embed the ODE as a loss term, treat k as a trainable parameter,
and optimize jointly. The physics constraint regularizes the solution.

HOW A PINN WORKS HERE
---------------------
We train a neural network u(t) ≈ [A(t), B(t)] that:

  1. FITS the sparse observations:
       L_data = (1/N_obs) * sum_i [ (u(t_i) - y_obs_i)^2 ]

  2. SATISFIES the ODE at collocation points (no labels needed):
       L_physics = (1/N_col) * sum_j [ (du/dt|_{t_j} + k*u_A(t_j)*u_B(t_j))^2 ]
                  + same for component B

  3. TOTAL LOSS: L = L_data + lambda * L_physics

The key trick: k is NOT fixed — it's a TRAINABLE PARAMETER initialized at
a wrong value (k_init). As training progresses, k is updated by gradient
descent simultaneously with the network weights. If the physics loss drives
the network toward the true ODE solution, k converges to k_true = 0.8.

We demonstrate this for the 20% PTMO composition with:
- Only 15 sparse noisy observations (realistic UV-VIS scenario)
- Collocation points at 100 times (no labels — just physics enforcement)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

os.makedirs("figures", exist_ok=True)
np.random.seed(42)

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11,
    "axes.titlesize": 12, "axes.labelsize": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150,
})

K_TRUE   = 0.8     # true rate constant (what we want to recover)
K_INIT   = 0.2     # deliberately wrong starting guess
LAM      = 5.0     # physics loss weight lambda

# ── PINN Network ──────────────────────────────────────────────────────────────
class PINN:
    """
    Neural network u_theta(t) → [A(t), B(t)]

    Architecture: scalar t → hidden(32) → hidden(32) → 2 outputs
    Activation: tanh (smooth, infinite derivatives — essential for computing
                du/dt via automatic differentiation through the network)

    k is a trainable scalar parameter initialized at K_INIT.
    """
    def __init__(self, hidden=32, k_init=K_INIT):
        h = hidden
        # Network weights
        self.W1 = np.random.randn(1, h) * np.sqrt(2.0/(1+h))
        self.b1 = np.zeros(h)
        self.W2 = np.random.randn(h, h) * np.sqrt(2.0/(h+h))
        self.b2 = np.zeros(h)
        self.W3 = np.random.randn(h, 2) * np.sqrt(2.0/(h+2)) * 0.01
        self.b3 = np.zeros(2)

        # Trainable physics parameter — this is what we're trying to recover
        self.log_k = np.array([np.log(k_init)])   # use log to keep k > 0

        # Adam states for all parameters
        self._init_adam()

    @property
    def k(self):
        """Current value of k (always positive via exp)."""
        return float(np.exp(self.log_k[0]))

    def _all_params(self):
        return [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3, self.log_k]

    def _init_adam(self):
        self.m = [np.zeros_like(p) for p in self._all_params()]
        self.v = [np.zeros_like(p) for p in self._all_params()]
        self.t_adam = 0

    def forward(self, t):
        """
        t: (N,1) array of time points
        Returns: (N,2) predicted [A(t), B(t)]
        Caches intermediates for backward.
        """
        self._t  = t
        self._z1 = t  @ self.W1 + self.b1
        self._h1 = np.tanh(self._z1)
        self._z2 = self._h1 @ self.W2 + self.b2
        self._h2 = np.tanh(self._z2)
        self._out = self._h2 @ self.W3 + self.b3
        return self._out

    def forward_with_grad(self, t):
        """
        Compute u(t) AND du/dt using the chain rule through the network.
        This is analytical automatic differentiation for a scalar input t.

        du/dt = dout/dh2 * dh2/dz2 * dz2/dh1 * dh1/dz1 * dz1/dt
              = W3^T * (1-h2^2) * W2^T * (1-h1^2) * W1^T

        This is EXACT — no finite differences needed.
        """
        u    = self.forward(t)
        # Backprop to compute du/dt analytically
        # Compute du_j/dt for each output j separately
        dudt = np.zeros((t.shape[0], 2))
        for j in range(2):
            ej   = np.zeros((t.shape[0], 2)); ej[:, j] = 1.0
            dh2  = ej @ self.W3.T * (1 - self._h2**2)     # (N, h)
            dh1  = dh2 @ self.W2.T * (1 - self._h1**2)    # (N, h)
            dudt[:, j] = (dh1 @ self.W1.T).ravel()        # (N,)
        return u, dudt

    def losses(self, t_obs, y_obs, t_col):
        """
        Compute data loss and physics loss.

        t_obs: (N_obs, 1) — observation times
        y_obs: (N_obs, 2) — observed [A, B] concentrations
        t_col: (N_col, 1) — collocation times (no labels needed)

        Returns: L_data, L_phys, total loss, and gradients
        """
        k = self.k

        # ── Data loss ──────────────────────────────────────────────────────
        u_obs = self.forward(t_obs)                     # (N_obs, 2)
        res_d = u_obs - y_obs                           # (N_obs, 2)
        L_data = np.mean(res_d**2)

        # ── Physics loss at collocation points ─────────────────────────────
        # ODE residual: du/dt + k*A*B != 0 if network doesn't satisfy ODE
        u_col, dudt_col = self.forward_with_grad(t_col)  # (N_col, 2), (N_col, 2)
        A_col = u_col[:, 0]
        B_col = u_col[:, 1]
        rate  = k * A_col * B_col                        # (N_col,)

        # Residuals: should be zero if ODE is satisfied
        res_A = dudt_col[:, 0] + rate   # dA/dt + k*A*B = 0
        res_B = dudt_col[:, 1] + rate   # dB/dt + k*A*B = 0
        L_phys = np.mean(res_A**2) + np.mean(res_B**2)

        L_total = L_data + LAM * L_phys

        # ── Gradients ─────────────────────────────────────────────────────
        # Grad of L_data w.r.t. network output at obs points
        d_out_data = 2.0 * res_d / len(t_obs)

        # Grad of L_phys w.r.t. u_col and dudt_col
        # dL/d(res_A) = 2*res_A/N, dL/d(du_A/dt) = dL/d(res_A)
        # Backprop through dudt is more complex; we approximate via output grad
        d_resA = 2.0 * res_A / len(t_col)
        d_resB = 2.0 * res_B / len(t_col)
        # Grad w.r.t. u_col (through rate = k*A*B):
        d_ucol = np.zeros_like(u_col)
        d_ucol[:, 0] += k * B_col * (d_resA + d_resB)   # d(rate)/d(A)
        d_ucol[:, 1] += k * A_col * (d_resA + d_resB)   # d(rate)/d(B)
        d_out_phys = LAM * d_ucol

        # Grad w.r.t. log_k (through k = exp(log_k))
        dk = np.sum((d_resA + d_resB) * rate)    # dk/d(log_k) = k
        d_logk = np.array([LAM * dk])

        return L_data, L_phys, L_total, d_out_data, d_out_phys, d_logk

    def backward_data(self, d_out):
        """Backprop through network for given d_out at obs points."""
        N = d_out.shape[0]
        dW3 = self._h2.T @ d_out / N
        db3 = d_out.mean(0)
        dh2 = d_out @ self.W3.T * (1 - self._h2**2)
        dW2 = self._h1.T @ dh2 / N
        db2 = dh2.mean(0)
        dh1 = dh2 @ self.W2.T * (1 - self._h1**2)
        dW1 = self._t.T @ dh1 / N
        db1 = dh1.mean(0)
        return [dW1, db1, dW2, db2, dW3, db3]

    def step(self, t_obs, y_obs, t_col, lr=1e-3):
        """One full optimization step."""
        L_d, L_p, L_tot, g_data, g_phys, g_logk = self.losses(
            t_obs, y_obs, t_col)

        # Backward through network for data loss (at obs points)
        self.forward(t_obs)
        grads_data = self.backward_data(g_data)

        # Backward through network for physics loss (at col points)
        self.forward(t_col)
        grads_phys = self.backward_data(g_phys)

        # Combine
        param_grads = [gd + LAM*gp for gd,gp in zip(grads_data, grads_phys)]
        param_grads.append(g_logk)   # gradient for log_k

        # Adam update
        self.t_adam += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        new_params = []
        for i, (p, g) in enumerate(zip(self._all_params(), param_grads)):
            self.m[i] = b1*self.m[i] + (1-b1)*g
            self.v[i] = b2*self.v[i] + (1-b2)*g**2
            mh = self.m[i]/(1-b1**self.t_adam)
            vh = self.v[i]/(1-b2**self.t_adam)
            new_params.append(p - lr*mh/(np.sqrt(vh)+eps))
        (self.W1,self.b1,self.W2,self.b2,
         self.W3,self.b3,self.log_k) = new_params

        return L_d, L_p, L_tot


# ── Load data ─────────────────────────────────────────────────────────────────
df     = pd.read_csv("data/ip_kinetics_training.csv")
df_20  = df[df["label"]=="20% PTMO"].reset_index(drop=True)
t_full = df_20["t"].values

# True trajectory
y_true_full = df_20[["A_true","B_true"]].values
A0 = float(df_20["A_true"].iloc[0])
B0 = float(df_20["B_true"].iloc[0])

# Sparse observations (15 points — simulating UV-VIS measurements)
N_OBS  = 15
obs_idx = np.linspace(0, len(t_full)-1, N_OBS, dtype=int)
t_obs  = t_full[obs_idx].reshape(-1, 1)
y_obs  = df_20[["A_noisy","B_noisy"]].values[obs_idx]

# Collocation points (100 points — no labels, just physics enforcement)
N_COL  = 100
t_col  = np.linspace(0, 10, N_COL).reshape(-1, 1)

print(f"Observations: {N_OBS}  |  Collocation points: {N_COL}")
print(f"True k = {K_TRUE}  |  Initial k = {K_INIT}")

# ── Train PINN ────────────────────────────────────────────────────────────────
pinn = PINN(hidden=32, k_init=K_INIT)
N_EPOCHS   = 3000
history    = {"L_data": [], "L_phys": [], "L_total": [], "k": []}

print(f"\nTraining PINN for {N_EPOCHS} epochs (lambda={LAM})...")
for ep in range(N_EPOCHS):
    L_d, L_p, L_t = pinn.step(t_obs, y_obs, t_col, lr=5e-4)
    history["L_data"].append(L_d)
    history["L_phys"].append(L_p)
    history["L_total"].append(L_t)
    history["k"].append(pinn.k)
    if ep % 500 == 0 or ep == N_EPOCHS-1:
        print(f"  Epoch {ep:5d}  L_data={L_d:.2e}  L_phys={L_p:.2e}  "
              f"k={pinn.k:.4f}  (true={K_TRUE})")

print(f"\nFinal recovered k = {pinn.k:.4f}  (true = {K_TRUE})")
print(f"Error: {abs(pinn.k - K_TRUE)/K_TRUE * 100:.1f}%")


# ── Figure 9: Loss curves and k convergence ───────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
fig.suptitle("Part 3 — PINN Inverse Problem: Loss Curves and Parameter Recovery",
             fontsize=12)

axes[0].semilogy(history["L_data"], color="#2166ac", lw=2, label="Data loss")
axes[0].set(xlabel="Epoch", ylabel="Loss (log)", title="Data Loss L_data")

axes[1].semilogy(history["L_phys"], color="#d73027", lw=2, label="Physics loss")
axes[1].set(xlabel="Epoch", ylabel="Loss (log)", title="Physics Loss L_phys")

axes[2].axhline(K_TRUE, color="black", ls="--", lw=2, label=f"True k={K_TRUE}")
axes[2].axhline(K_INIT, color="gray",  ls=":",  lw=1.5, label=f"Initial k={K_INIT}")
axes[2].plot(history["k"], color="#1a9641", lw=2.5, label="Recovered k(epoch)")
axes[2].set(xlabel="Epoch", ylabel="Rate constant k  (L/mol/s)",
            title=f"Parameter Recovery\nFinal k={pinn.k:.4f} vs True k={K_TRUE}")
axes[2].legend(fontsize=9)

plt.tight_layout()
plt.savefig("figures/fig9_pinn_losses.png", bbox_inches="tight")
plt.close()
print("\nSaved figures/fig9_pinn_losses.png")


# ── Figure 10: PINN solution vs true trajectory ───────────────────────────────
t_plot = np.linspace(0, 10, 200).reshape(-1,1)
u_pred = pinn.forward(t_plot)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
fig.suptitle(f"Part 3 — PINN Solution vs True Trajectory\n"
             f"k recovered = {pinn.k:.4f}  (true = {K_TRUE}, init = {K_INIT})",
             fontsize=12)

for ax, s, name, col in zip([ax1,ax2],[0,1],
    ["[A] Diamine (mol/L)","[B] Diacyl Chloride (mol/L)"],
    ["#2166ac","#d73027"]):
    ax.plot(t_full, y_true_full[:,s],  "k--", lw=2,   label="True ODE", zorder=3)
    ax.scatter(t_obs.ravel(), y_obs[:,s], s=60, color="black",
               zorder=5, label=f"Sparse observations (N={N_OBS})")
    ax.plot(t_plot.ravel(), u_pred[:,s], color=col, lw=2.5,
            label="PINN solution", zorder=4)
    ax.set(xlabel="Time (s)", ylabel=name, title=name.split("(")[0])
    ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig("figures/fig10_pinn_solution.png", bbox_inches="tight")
plt.close()
print("Saved figures/fig10_pinn_solution.png")


# ── Figure 11: Sensitivity to lambda (physics loss weight) ───────────────────
lambdas     = [0.1, 1.0, 5.0, 20.0]
k_recovered = []
print("\nSweeping lambda values...")

for lam in lambdas:
    LAM = lam
    p_tmp = PINN(hidden=32, k_init=K_INIT)
    for ep in range(2000):
        p_tmp.step(t_obs, y_obs, t_col, lr=5e-4)
    k_recovered.append(p_tmp.k)
    print(f"  lambda={lam:5.1f}  k_recovered={p_tmp.k:.4f}")

LAM = 5.0  # reset

fig, ax = plt.subplots(figsize=(7, 4.5))
bars = ax.bar([str(l) for l in lambdas], k_recovered,
              color=["#d73027","#fc8d59","#1a9641","#2166ac"],
              width=0.5, edgecolor="white")
ax.axhline(K_TRUE, color="black", ls="--", lw=2, label=f"True k = {K_TRUE}")
ax.axhline(K_INIT, color="gray",  ls=":",  lw=1.5, label=f"Initial k = {K_INIT}")
for bar, v in zip(bars, k_recovered):
    ax.text(bar.get_x()+bar.get_width()/2, v+0.01, f"{v:.3f}",
            ha="center", va="bottom", fontsize=10)
ax.set(xlabel="Physics loss weight λ", ylabel="Recovered k  (L/mol/s)",
       title="Part 3 — Sensitivity of k Recovery to λ\n"
             "(Higher λ → stronger physics enforcement)")
ax.legend()
plt.tight_layout()
plt.savefig("figures/fig11_lambda_sensitivity.png", bbox_inches="tight")
plt.close()
print("Saved figures/fig11_lambda_sensitivity.png")


# ── Print final summary ───────────────────────────────────────────────────────
print(f"\n{'='*50}")
print(f"PINN INVERSE PROBLEM SUMMARY")
print(f"{'='*50}")
print(f"  True k           = {K_TRUE:.4f} L/mol/s")
print(f"  Initial k (wrong) = {K_INIT:.4f} L/mol/s")
print(f"  Recovered k       = {pinn.k:.4f} L/mol/s")
print(f"  Recovery error    = {abs(pinn.k-K_TRUE)/K_TRUE*100:.1f}%")
print(f"  Observations used = {N_OBS} (vs 300 total time points)")
print(f"{'='*50}")
