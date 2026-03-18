# Neural ODEs and Physics-Informed Learning for Interfacial Polymerization Kinetics

**Course:** Scientific Machine Learning  
**Author:** Dylan McAfee
**Date:** March 18 2026

---

## Overview

This project applies Scientific Machine Learning to understand the reaction kinetics
of interfacial polymerization (IP) in the synthesis of thermoplastic polyamide elastomer
(TPAE) foams — my active PhD research. TPAEs consisting of PTMO soft segments and
Nylon-6,10 hard segments are promising sustainable alternatives to neoprene for wetsuit
applications.

**The central scientific questions:**
1. How do IP reaction kinetics evolve for each TPAE composition?
2. Can a Neural ODE learn the reaction dynamics from noisy time-series data alone?
3. Can a PINN recover unknown kinetic parameters (rate constant k) from sparse measurements?
4. How do reaction outcomes connect to final material properties (Tg, Tm, E)?

---

## Project Structure

```
tpae-sciml/
├── part1_ode/
│   └── simulate_kinetics.py     # ODE simulation, numerical stability analysis
├── part2_neural_ode/
│   └── neural_ode.py            # Neural ODE with BPTT, generalization tests
├── part3_pinn/
│   └── pinn_inverse.py          # PINN inverse problem: recovering k from sparse data
├── coda_properties/
│   └── structure_property.py    # Kinetics → Xn → Tg/Tm/E via Flory-Fox
├── figures/                     # All generated figures (13 total)
├── report/
│   └── report.tex               # 2-page LaTeX report (conference abstract style)
├── requirements.txt
└── README.md
```

---

## Compositions Studied

| Sample | PTMO (wt%) | Nylon-6,10 (wt%) | [A]₀ (mol/L) |
|--------|-----------|------------------|---------------|
| 1      | 5         | 95               | 0.097         |
| 2      | 10        | 90               | 0.094         |
| 3      | 20        | 80               | 0.088         |
| 4      | 30        | 70               | 0.082         |
| 5      | 40        | 60               | 0.076         |
| 6      | 60        | 40               | 0.064         |

[A]₀ decreases with PTMO content because PTMO ether-diamines are less reactive
than hexamethylene diamine (Nylon-6,10 component).

---

## Part 1 — ODE Simulation (Jan 22 lecture)

**Physics:** Second-order bimolecular reaction in the well-mixed approximation:

$$\frac{d[A]}{dt} = -k[A][B], \quad \frac{d[B]}{dt} = -k[A][B]$$

**Key findings:**
- Higher PTMO content → lower [A]₀ → slightly higher final conversion (limiting reagent effect)
- Forward Euler with Δt=2.0s diverges; RK45 gives stable, accurate trajectories
- Carothers equation connects final conversion to degree of polymerization Xn

**Figures:** fig1 (trajectories), fig2 (rate + Xn), fig3 (stability), fig4 (final conversion vs composition)

---

## Part 2 — Neural ODE (Jan 29 lecture)

**Architecture:** MLP input(2)→hidden(16)→hidden(16)→output(2), tanh activations  
**Training:** BPTT through unrolled Euler steps (equivalent to adjoint method), Adam optimizer, 600 epochs  
**Training data:** 20% PTMO composition with Gaussian noise (σ=0.005 mol/L)  
**Test:** Generalize to 10%, 30%, 40% PTMO (unseen initial conditions)

**Key findings:**
- Neural ODE fits training trajectory with MAE < 10⁻³ mol/L
- Generalizes to unseen compositions with MAE < 4×10⁻³ mol/L
- Learned rate surface qualitatively recovers bilinear structure of k·[A]·[B]
- Systematic error in high-concentration region: insufficient training coverage

**Figures:** fig5 (loss), fig6 (fit), fig7 (generalization), fig8 (rate surface)

---

## Part 3 — PINN Inverse Problem (Feb 26 lecture)

**Setup:** Only 15 sparse noisy observations of [A](t), [B](t)  
**Goal:** Recover rate constant k (initialized at k₀=0.2, true k=0.8 L/mol/s)

**PINN loss:**
$$\mathcal{L} = \mathcal{L}_\text{data} + \lambda \mathcal{L}_\text{physics}$$

where the physics loss enforces the ODE residual at 100 collocation points (no labels needed).
k is a trainable parameter updated by gradient descent alongside network weights.

**Key findings:**
- k recovered: 0.72 L/mol/s (9.9% error from 15 observations, starting at k=0.2)
- Monotonic convergence over 3000 epochs
- λ sensitivity: moderate physics weighting (λ=5) optimal; too-high λ over-constrains
- Identifiability limited by narrow concentration range — realistic challenge in IP systems

**Figures:** fig9 (losses + k convergence), fig10 (PINN solution), fig11 (λ sensitivity)

---

## Coda — Kinetics → Structure → Properties

Connects reaction outcomes to material design:
- Carothers equation: X∞ → Xn → Mn
- Fox equation: Tg from composition blending
- Flory-Fox correction: Tg(Mn) = Tg,∞ − K/Mn
  - Higher PTMO → lower Xn → stronger chain-end plasticization → Tg suppressed by ~30°C
- Elastic modulus E via logarithmic rule of mixtures

**Figure:** fig12 (kinetics→properties), fig13 (full project summary)

---

## How to Reproduce

```bash
# Install dependencies
pip install -r requirements.txt

# Part 1: ODE simulation
python part1_ode/simulate_kinetics.py

# Part 2: Neural ODE
python part2_neural_ode/neural_ode.py

# Part 3: PINN inverse problem
python part3_pinn/pinn_inverse.py

# Coda: structure-property
python coda_properties/structure_property.py

# Compile report (requires pdflatex)
cd report && pdflatex report.tex && pdflatex report.tex
```

---

## Course Connections

| Course Topic | Where Used |
|---|---|
| ODE theory + numerical solvers (Jan 22) | Part 1: RK45 vs Euler stability |
| Neural ODEs + adjoint method (Jan 29) | Part 2: BPTT through Euler steps |
| PINNs (Feb 26) | Part 3: physics loss + trainable k |
| Deep learning fundamentals (Jan 15) | MLP architecture, Adam optimizer |
| Differentiable physics (Mar 12) | Analytical du/dt via chain rule in PINN |

---

## Limitations

- Well-mixed ODE ignores interfacial film diffusion (key mechanism in real IP)
- Synthetic data generated from same physics → no true model mismatch in training
- k identifiability limited by narrow concentration range in early-time observations
- Real experimental data (UV-VIS, NMR kinetics) will be integrated as synthesis progresses

---

## References

- Chen et al. (2018). Neural Ordinary Differential Equations. *NeurIPS*
- Raissi et al. (2019). Physics-informed neural networks. *J. Comput. Phys.* 378, 686–707
- Karniadakis et al. (2021). Physics-informed machine learning. *Nature Reviews Physics* 3, 422–440
- Gong et al. (2021). *Macromolecular Materials and Engineering*
- Fox & Flory (1950). *J. Appl. Phys.* 21, 581
