"""
part2_neural_ode/neural_ode.py
==============================

Part 2: Neural ODE for Interfacial Polymerization Kinetics

WHAT IS A NEURAL ODE?
---------------------
A standard ODE says:  dy/dt = f(y, t)   — f is KNOWN physics
A Neural ODE says:    dy/dt = f_theta(y) — f is a LEARNED neural network

The key idea: instead of assuming we know the kinetic model (k*A*B),
we let a neural network LEARN the dynamics directly from data. This is
powerful when:
  - Real IP kinetics deviate from the well-mixed model (film diffusion)
  - Rate constants vary with temperature, solvent, PTMO content
  - The NN can capture effects not analytically modeled

IMPLEMENTATION
--------------
We backpropagate analytically through unrolled Euler steps.
At each step:  y_{n+1} = y_n + dt * f_theta(y_n)

Gradient flows: dL/dtheta = sum_t [ dt * dL/dy_{t+1} * df/dtheta ]

This is BPTT (backpropagation through time) for the ODE.
Mathematically equivalent to the adjoint method for short time horizons.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

os.makedirs("figures", exist_ok=True)
np.random.seed(0)

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11,
    "axes.titlesize": 12, "axes.labelsize": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150,
})

# ── Neural Network ────────────────────────────────────────────────────────────
class NeuralODE:
    """
    MLP: input(2) → hidden(16) → hidden(16) → output(2)
    Input:  [A, B]     — concentrations
    Output: [dA/dt, dB/dt] — learned derivatives
    Activation: tanh
    """
    def __init__(self, hidden=16):
        h = hidden
        self.W1 = np.random.randn(2, h) * np.sqrt(2.0/(2+h))
        self.b1 = np.zeros(h)
        self.W2 = np.random.randn(h, h) * np.sqrt(2.0/(h+h))
        self.b2 = np.zeros(h)
        self.W3 = np.random.randn(h, 2) * np.sqrt(2.0/(h+2)) * 0.1
        self.b3 = np.zeros(2)
        self.m  = [np.zeros_like(p) for p in self._params()]
        self.v  = [np.zeros_like(p) for p in self._params()]
        self.t_adam = 0

    def _params(self):
        return [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3]

    def forward(self, x):
        """x: (N,2). Returns (N,2) and caches for backward."""
        self._x  = x
        self._h1 = np.tanh(x  @ self.W1 + self.b1)
        self._h2 = np.tanh(self._h1 @ self.W2 + self.b2)
        return self._h2 @ self.W3 + self.b3

    def backward(self, d_out):
        """d_out: (N,2). Returns param grads."""
        N  = d_out.shape[0]
        dW3 = self._h2.T @ d_out / N
        db3 = d_out.mean(0)
        dh2 = d_out @ self.W3.T * (1 - self._h2**2)
        dW2 = self._h1.T @ dh2 / N
        db2 = dh2.mean(0)
        dh1 = dh2 @ self.W2.T * (1 - self._h1**2)
        dW1 = self._x.T @ dh1 / N
        db1 = dh1.mean(0)
        return [dW1, db1, dW2, db2, dW3, db3]

    def adam_step(self, grads, lr=3e-3):
        self.t_adam += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        new = []
        for i, (p, g) in enumerate(zip(self._params(), grads)):
            self.m[i] = b1*self.m[i] + (1-b1)*g
            self.v[i] = b2*self.v[i] + (1-b2)*g**2
            mh = self.m[i]/(1-b1**self.t_adam)
            vh = self.v[i]/(1-b2**self.t_adam)
            new.append(p - lr*mh/(np.sqrt(vh)+eps))
        (self.W1,self.b1,self.W2,self.b2,self.W3,self.b3) = new

    def predict(self, y0, t_eval, n_sub=5):
        """Roll out trajectory from y0 over t_eval using Euler."""
        y = y0.copy(); traj = [y.copy()]
        for i in range(len(t_eval)-1):
            dt = (t_eval[i+1]-t_eval[i])/n_sub
            for _ in range(n_sub):
                dydt = self.forward(y[np.newaxis,:])[0]
                y    = np.maximum(y + dt*dydt, 0.0)
            traj.append(y.copy())
        return np.array(traj)

    def train_step(self, y0, t_eval, y_target, lr=3e-3, n_sub=5):
        """One full BPTT step over the trajectory."""
        T = len(t_eval)

        # --- Forward: collect states ---
        states = [y0.copy()]
        y = y0.copy()
        for i in range(T-1):
            dt = (t_eval[i+1]-t_eval[i])/n_sub
            for _ in range(n_sub):
                dydt = self.forward(y[np.newaxis,:])[0]
                y    = np.maximum(y + dt*dydt, 0.0)
            states.append(y.copy())
        states = np.array(states)   # (T,2)

        # --- Loss ---
        diff  = states - y_target   # (T,2)
        loss  = np.mean(diff**2)

        # --- Backward through Euler steps ---
        accum = [np.zeros_like(p) for p in self._params()]
        for i in range(T-1):
            dt    = (t_eval[i+1]-t_eval[i])/n_sub
            x_in  = states[i:i+1,:]          # (1,2)
            self.forward(x_in)               # restore cache
            d_out = dt * diff[i+1:i+2,:] * 2.0/(T*2)
            step_g = self.backward(d_out)
            for j,sg in enumerate(step_g):
                accum[j] += sg

        self.adam_step(accum, lr=lr)
        return loss


# ── Load data ─────────────────────────────────────────────────────────────────
df = pd.read_csv("data/ip_kinetics_training.csv")

TRAIN  = "20% PTMO"
TESTS  = ["10% PTMO", "30% PTMO", "40% PTMO"]

df_tr  = df[df["label"]==TRAIN].reset_index(drop=True)
t_full = df_tr["t"].values

stride = 5
t_sub  = t_full[::stride]
y_sub  = df_tr[["A_noisy","B_noisy"]].values[::stride]
y0_tr  = np.array([df_tr["A_true"].iloc[0], df_tr["B_true"].iloc[0]])

print(f"Training on {TRAIN}  |  {len(t_sub)} time points")

# ── Train ─────────────────────────────────────────────────────────────────────
model   = NeuralODE(hidden=16)
history = []
N_EPOCHS = 600

print(f"Training Neural ODE for {N_EPOCHS} epochs...")
for ep in range(N_EPOCHS):
    loss = model.train_step(y0_tr, t_sub, y_sub, lr=3e-3)
    history.append(loss)
    if ep % 100 == 0 or ep == N_EPOCHS-1:
        print(f"  Epoch {ep:4d}  Loss: {loss:.2e}")
print("Done.\n")

# ── Figure 5: Loss curve ──────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7,4))
ax.semilogy(history, color="#2166ac", lw=2)
ax.set(xlabel="Epoch", ylabel="MSE Loss (log scale)",
       title=f"Part 2 — Neural ODE Training Loss\nTrained on {TRAIN}")
ax.axhline(history[-1], ls="--", color="gray", lw=1,
           label=f"Final: {history[-1]:.2e}")
ax.legend()
plt.tight_layout()
plt.savefig("figures/fig5_training_loss.png", bbox_inches="tight")
plt.close()
print("Saved figures/fig5_training_loss.png")

# ── Figure 6: Fit on training composition ────────────────────────────────────
y_pred = model.predict(y0_tr, t_full)
y_true = df_tr[["A_true","B_true"]].values
y_noisy= df_tr[["A_noisy","B_noisy"]].values

fig, (ax1,ax2) = plt.subplots(1,2,figsize=(11,4.5))
fig.suptitle(f"Part 2 — Neural ODE Fit: {TRAIN}", fontsize=12)
for ax,s,name,col in zip([ax1,ax2],[0,1],
    ["[A] Diamine (mol/L)","[B] Diacyl Chloride (mol/L)"],
    ["#2166ac","#d73027"]):
    ax.scatter(t_full[::4], y_noisy[::4,s], s=8, color="gray",
               alpha=0.5, label="Noisy data", zorder=2)
    ax.plot(t_full, y_true[:,s],  "k--", lw=1.5, label="True ODE",  zorder=3)
    ax.plot(t_full, y_pred[:,s],  color=col, lw=2.5,
            label="Neural ODE", zorder=4)
    ax.set(xlabel="Time (s)", ylabel=name, title=name.split("(")[0])
    ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig("figures/fig6_neural_ode_fit.png", bbox_inches="tight")
plt.close()
print("Saved figures/fig6_neural_ode_fit.png")

# ── Figure 7: Generalization ──────────────────────────────────────────────────
fig, axes = plt.subplots(len(TESTS), 2, figsize=(11, 4*len(TESTS)))
fig.suptitle("Part 2 — Neural ODE Generalization to Unseen Compositions",
             fontsize=12)
gen_errors = {}
for row, tlabel in enumerate(TESTS):
    df_t  = df[df["label"]==tlabel].reset_index(drop=True)
    y0_t  = np.array([df_t["A_true"].iloc[0], df_t["B_true"].iloc[0]])
    y_t   = df_t[["A_true","B_true"]].values
    y_p   = model.predict(y0_t, t_full)
    mae   = np.mean(np.abs(y_p - y_t), axis=0)
    gen_errors[tlabel] = mae
    for col,s,name,c in zip(range(2),[0,1],
        ["[A] Diamine","[B] Diacyl Chloride"],
        ["#2166ac","#d73027"]):
        ax = axes[row,col]
        ax.plot(t_full, y_t[:,s],  "k--", lw=1.5, label="True ODE")
        ax.plot(t_full, y_p[:,s],  color=c, lw=2.5, label="Neural ODE")
        ax.set_title(f"{tlabel} — {name}  (MAE={mae[s]:.5f})")
        ax.set(xlabel="Time (s)", ylabel=name)
        ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig("figures/fig7_neural_ode_generalization.png", bbox_inches="tight")
plt.close()
print("Saved figures/fig7_neural_ode_generalization.png")

# ── Figure 8: Learned rate surface vs true ───────────────────────────────────
A_grid = np.linspace(0.03, 0.10, 40)
B_grid = np.linspace(0.04, 0.08, 40)
AA, BB = np.meshgrid(A_grid, B_grid)
rate_true = 0.8 * AA * BB
AB_flat   = np.column_stack([AA.ravel(), BB.ravel()])
dydt_nn   = model.forward(AB_flat)
rate_nn   = np.abs(dydt_nn[:,0]).reshape(AA.shape)

fig, axes = plt.subplots(1,3,figsize=(14,4.5))
fig.suptitle("Part 2 — Learned vs True Reaction Rate Surface", fontsize=12)
vmax = rate_true.max()
kw   = dict(levels=20, vmin=0, vmax=vmax)
im1 = axes[0].contourf(AA,BB,rate_true, cmap="Blues", **kw)
axes[0].set(xlabel="[A]",ylabel="[B]",title="True: k·[A]·[B]")
plt.colorbar(im1,ax=axes[0],label="mol/L/s")
im2 = axes[1].contourf(AA,BB,rate_nn,   cmap="Reds",  **kw)
axes[1].set(xlabel="[A]",ylabel="[B]",title="Neural ODE learned rate")
plt.colorbar(im2,ax=axes[1],label="mol/L/s")
diff = np.abs(rate_nn-rate_true)
im3 = axes[2].contourf(AA,BB,diff, cmap="Greens", levels=20)
axes[2].set(xlabel="[A]",ylabel="[B]",title=f"|error|  mean={diff.mean():.4f}")
plt.colorbar(im3,ax=axes[2],label="mol/L/s")
plt.tight_layout()
plt.savefig("figures/fig8_learned_dynamics.png", bbox_inches="tight")
plt.close()
print("Saved figures/fig8_learned_dynamics.png")

# ── Summary ───────────────────────────────────────────────────────────────────
print("── Generalization MAE (mol/L) ──────────────────────")
for lbl, mae in gen_errors.items():
    print(f"  {lbl:<12}  [A]: {mae[0]:.5f}  [B]: {mae[1]:.5f}")
