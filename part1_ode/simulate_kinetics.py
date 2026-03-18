"""
part1_ode/simulate_kinetics.py
==============================

Part 1: Physics-based ODE simulation of interfacial polymerization (IP) kinetics.

PHYSICAL BACKGROUND
-------------------
In interfacial polymerization, a diamine (monomer A, aqueous phase) reacts with
a diacyl chloride (monomer B, organic phase) at their shared interface. In the
well-mixed approximation (valid early in the reaction before a thick film forms),
the kinetics reduce to a coupled second-order ODE system:

    d[A]/dt = -k * [A] * [B]      (diamine consumed)
    d[B]/dt = -k * [A] * [B]      (diacyl chloride consumed)
    dP/dt   = +k * [A] * [B]      (polymer produced)

where k is the bimolecular rate constant (L/mol/s).

KEY PHYSICS:
- Second-order reaction: rate depends on BOTH concentrations simultaneously
- If [A]0 = [B]0 (stoichiometric), both go to zero together → highest Mw
- If [A]0 ≠ [B]0, the limiting reagent controls final conversion
- In real IP, [A]0 varies with PTMO content (PTMO diamine vs nylon diamine ratio)
- The polymer growth rate dP/dt peaks early and decays as monomers are consumed

COMPOSITION-DEPENDENT INITIAL CONDITIONS
-----------------------------------------
Each TPAE composition has a different effective diamine concentration [A]0
because PTMO soft segments contribute ether-diamine end groups while Nylon-6,10
hard segments contribute hexamethylene diamine. We model this as:

    [A]0 = w_nylon * [A]0_nylon + w_ptmo * [A]0_ptmo

where PTMO diamines are less reactive (lower effective concentration) due to
steric hindrance from the flexible ether chain.
"""

import numpy as np
from scipy.integrate import solve_ivp
import pandas as pd
import matplotlib.pyplot as plt
import os

os.makedirs("figures", exist_ok=True)
os.makedirs("data",    exist_ok=True)

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11,
    "axes.titlesize": 12, "axes.labelsize": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150,
})

# ── Physical parameters ───────────────────────────────────────────────────────
k_true   = 0.8    # L/mol/s  — bimolecular rate constant (true value we'll recover in Part 3)
t_span   = (0, 10)  # seconds — IP is fast
t_eval   = np.linspace(0, 10, 300)

# Composition-dependent initial diamine concentration [A]0
# Nylon-6,10 diamine (hexamethylene diamine): highly reactive, [A]0 ~ 0.10 mol/L
# PTMO diamine (ether diamine): less reactive, [A]0 ~ 0.04 mol/L
# Diacyl chloride [B]0 is fixed in the organic phase: 0.08 mol/L

A0_nylon = 0.10   # mol/L
A0_ptmo  = 0.04   # mol/L
B0_fixed = 0.08   # mol/L

compositions = [
    {"label": "5% PTMO",  "w_ptmo": 0.05, "w_nylon": 0.95},
    {"label": "10% PTMO", "w_ptmo": 0.10, "w_nylon": 0.90},
    {"label": "20% PTMO", "w_ptmo": 0.20, "w_nylon": 0.80},
    {"label": "30% PTMO", "w_ptmo": 0.30, "w_nylon": 0.70},
    {"label": "40% PTMO", "w_ptmo": 0.40, "w_nylon": 0.60},
    {"label": "60% PTMO", "w_ptmo": 0.60, "w_nylon": 0.40},
]

COLORS = plt.cm.viridis(np.linspace(0.1, 0.9, len(compositions)))

# ── ODE system ────────────────────────────────────────────────────────────────
def ip_odes(t, y, k):
    """
    Coupled ODE system for interfacial polymerization.

    State vector y = [A, B, P]
      A : diamine concentration   (mol/L)
      B : diacyl chloride conc.   (mol/L)
      P : polymer concentration   (mol/L, proxy for conversion)

    Returns dy/dt.
    """
    A, B, P = y
    A = max(A, 0.0)   # clamp to avoid negative concentrations
    B = max(B, 0.0)
    rate = k * A * B
    dAdt = -rate
    dBdt = -rate
    dPdt = +rate
    return [dAdt, dBdt, dPdt]


def conversion(sol, A0):
    """Fractional monomer A conversion: X = (A0 - A(t)) / A0"""
    return (A0 - sol.y[0]) / A0


def mw_proxy(sol):
    """
    Degree of polymerization proxy using Carothers equation:
        Xn = 1 / (1 - p)
    where p = conversion of limiting reagent.
    Capped at 1000 to avoid divergence near complete conversion.
    """
    A = sol.y[0]
    B = sol.y[1]
    A0 = A[0]
    B0 = B[0]
    # conversion of limiting reagent
    if A0 <= B0:
        p = np.clip((A0 - A) / A0, 0, 0.999)
    else:
        p = np.clip((B0 - B) / B0, 0, 0.999)
    return np.clip(1.0 / (1.0 - p), 1, 1000)


# ── Simulate all compositions ─────────────────────────────────────────────────
results = []
for comp in compositions:
    A0 = comp["w_nylon"] * A0_nylon + comp["w_ptmo"] * A0_ptmo
    B0 = B0_fixed
    y0 = [A0, B0, 0.0]
    sol = solve_ivp(ip_odes, t_span, y0, args=(k_true,),
                    t_eval=t_eval, method="RK45", rtol=1e-8, atol=1e-10)
    comp["A0"] = A0
    comp["B0"] = B0
    comp["sol"] = sol
    comp["X"]   = conversion(sol, A0)
    comp["Xn"]  = mw_proxy(sol)
    comp["X_final"] = comp["X"][-1]
    comp["Xn_final"] = comp["Xn"][-1]
    results.append(comp)
    print(f"{comp['label']:10s}  [A]0={A0:.4f}  [B]0={B0:.3f}  "
          f"X_final={comp['X_final']:.3f}  Xn_final={comp['Xn_final']:.1f}")


# ── Figure 1: Concentration trajectories ─────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
fig.suptitle("Part 1 — IP Kinetics: Concentration & Conversion Trajectories\n"
             "ODE system solved with RK45", fontsize=12)

ax_A, ax_B, ax_P = axes

for i, comp in enumerate(results):
    sol = comp["sol"]
    c   = COLORS[i]
    lbl = comp["label"]
    ax_A.plot(sol.t, sol.y[0], color=c, lw=2,   label=lbl)
    ax_B.plot(sol.t, sol.y[1], color=c, lw=2,   label=lbl)
    ax_P.plot(sol.t, comp["X"], color=c, lw=2,  label=lbl)

ax_A.set(xlabel="Time (s)", ylabel="[A] Diamine (mol/L)",    title="Diamine [A](t)")
ax_B.set(xlabel="Time (s)", ylabel="[B] Diacyl Chloride (mol/L)", title="Diacyl Chloride [B](t)")
ax_P.set(xlabel="Time (s)", ylabel="Fractional Conversion X", title="Monomer Conversion X(t)")
ax_P.set_ylim(0, 1.05)

for ax in axes:
    ax.legend(fontsize=8, loc="best")

plt.tight_layout()
plt.savefig("figures/fig1_concentration_trajectories.png", bbox_inches="tight")
plt.close()
print("\nSaved figures/fig1_concentration_trajectories.png")


# ── Figure 2: Reaction rate dP/dt and degree of polymerization ───────────────
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
fig.suptitle("Part 1 — Reaction Rate and Degree of Polymerization vs Time", fontsize=12)

ax_rate, ax_xn = axes

for i, comp in enumerate(results):
    sol  = comp["sol"]
    c    = COLORS[i]
    lbl  = comp["label"]
    # reaction rate = k*A*B at each time step
    rate = k_true * sol.y[0] * sol.y[1]
    ax_rate.plot(sol.t, rate, color=c, lw=2, label=lbl)
    ax_xn.plot(sol.t, comp["Xn"],  color=c, lw=2, label=lbl)

ax_rate.set(xlabel="Time (s)", ylabel="Reaction Rate k·[A]·[B]  (mol/L/s)",
            title="Reaction Rate vs Time\n(peaks at t=0, decays as monomers consumed)")
ax_xn.set(xlabel="Time (s)", ylabel="Degree of Polymerization Xn",
          title="Carothers Degree of Polymerization\n(proxy for molecular weight)")

for ax in axes:
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig("figures/fig2_rate_and_mw.png", bbox_inches="tight")
plt.close()
print("Saved figures/fig2_rate_and_mw.png")


# ── Figure 3: Numerical stability — RK45 vs Euler ────────────────────────────
# Use 20% PTMO composition as test case
comp_test = results[2]   # 20% PTMO
A0 = comp_test["A0"]
B0 = comp_test["B0"]
y0 = [A0, B0, 0.0]

# Forward Euler with three step sizes
def euler_solve(y0, t_span, dt, k):
    t_list = [t_span[0]]
    y_list = [np.array(y0, dtype=float)]
    t = t_span[0]
    y = np.array(y0, dtype=float)
    while t < t_span[1]:
        dydt = np.array(ip_odes(t, y, k))
        y = y + dt * dydt
        y = np.maximum(y, 0.0)
        t = min(t + dt, t_span[1])
        t_list.append(t)
        y_list.append(y.copy())
    return np.array(t_list), np.array(y_list)

sol_rk45 = solve_ivp(ip_odes, t_span, y0, args=(k_true,),
                     t_eval=t_eval, method="RK45", rtol=1e-8, atol=1e-10)

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
fig.suptitle("Part 1 — Numerical Stability: Forward Euler vs RK45\n"
             "(20% PTMO composition)", fontsize=12)

dt_values  = [2.0, 0.5, 0.1]
euler_cols = ["#d73027", "#fc8d59", "#fee08b"]
labels_eu  = ["Euler Δt=2.0s (unstable)", "Euler Δt=0.5s", "Euler Δt=0.1s"]

for ax, species, ylabel, title in zip(
    axes,
    [0, 1, None],
    ["[A] Diamine (mol/L)", "[B] Diacyl Chloride (mol/L)", "Fractional Conversion X"],
    ["Diamine [A](t)", "Diacyl Chloride [B](t)", "Conversion X(t)"]
):
    for dt, col, lbl in zip(dt_values, euler_cols, labels_eu):
        t_eu, y_eu = euler_solve(y0, t_span, dt, k_true)
        if species is not None:
            ax.plot(t_eu, y_eu[:, species], "--", color=col, lw=1.8, label=lbl, alpha=0.85)
        else:
            X_eu = (A0 - y_eu[:, 0]) / A0
            ax.plot(t_eu, X_eu, "--", color=col, lw=1.8, label=lbl, alpha=0.85)

    if species is not None:
        ax.plot(sol_rk45.t, sol_rk45.y[species], "k-", lw=2.5, label="RK45 (reference)")
    else:
        X_rk = (A0 - sol_rk45.y[0]) / A0
        ax.plot(sol_rk45.t, X_rk, "k-", lw=2.5, label="RK45 (reference)")

    ax.set(xlabel="Time (s)", ylabel=ylabel, title=title)
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig("figures/fig3_numerical_stability.png", bbox_inches="tight")
plt.close()
print("Saved figures/fig3_numerical_stability.png")


# ── Figure 4: Final conversion and Xn vs composition ─────────────────────────
ptmo_pct    = [c["w_ptmo"] * 100 for c in results]
X_finals    = [c["X_final"]  for c in results]
Xn_finals   = [c["Xn_final"] for c in results]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))
fig.suptitle("Part 1 — Final Conversion and Degree of Polymerization vs Composition",
             fontsize=12)

ax1.plot(ptmo_pct, X_finals,  "o-", color="#2166ac", lw=2, ms=8)
ax1.set(xlabel="PTMO Content (wt%)", ylabel="Final Conversion X∞",
        title="Final Monomer Conversion\n(lower PTMO → [A]0 closer to [B]0 → higher conversion)")
ax1.set_ylim(0, 1.05)
for x, y in zip(ptmo_pct, X_finals):
    ax1.annotate(f"{y:.2f}", (x, y), textcoords="offset points",
                 xytext=(0, 8), ha="center", fontsize=9)

ax2.plot(ptmo_pct, Xn_finals, "s-", color="#1a9641", lw=2, ms=8)
ax2.set(xlabel="PTMO Content (wt%)", ylabel="Final Degree of Polymerization Xn",
        title="Carothers Xn at t=10s\n(proxy for molecular weight)")
for x, y in zip(ptmo_pct, Xn_finals):
    ax2.annotate(f"{y:.0f}", (x, y), textcoords="offset points",
                 xytext=(0, 8), ha="center", fontsize=9)

plt.tight_layout()
plt.savefig("figures/fig4_final_conversion_vs_composition.png", bbox_inches="tight")
plt.close()
print("Saved figures/fig4_final_conversion_vs_composition.png")


# ── Save training data for Part 2 ─────────────────────────────────────────────
np.random.seed(42)
noise_std = 0.005   # mol/L — realistic measurement noise for UV-VIS

all_rows = []
for comp in results:
    sol = comp["sol"]
    for j, t in enumerate(sol.t):
        A_noisy = max(sol.y[0, j] + np.random.normal(0, noise_std), 0.0)
        B_noisy = max(sol.y[1, j] + np.random.normal(0, noise_std), 0.0)
        all_rows.append({
            "w_ptmo":   comp["w_ptmo"],
            "label":    comp["label"],
            "t":        t,
            "A_true":   sol.y[0, j],
            "B_true":   sol.y[1, j],
            "P_true":   sol.y[2, j],
            "A_noisy":  A_noisy,
            "B_noisy":  B_noisy,
            "X_true":   comp["X"][j],
        })

df_train = pd.DataFrame(all_rows)
df_train.to_csv("data/ip_kinetics_training.csv", index=False)
print(f"\nSaved data/ip_kinetics_training.csv  ({len(df_train)} rows)")

# Summary table
print("\n── Summary: Final state by composition ──────────────────────")
print(f"{'Composition':<12} {'[A]0':>8} {'[B]0':>8} {'X∞':>8} {'Xn∞':>8}")
print("─" * 50)
for c in results:
    print(f"{c['label']:<12} {c['A0']:>8.4f} {c['B0']:>8.4f} "
          f"{c['X_final']:>8.3f} {c['Xn_final']:>8.1f}")
