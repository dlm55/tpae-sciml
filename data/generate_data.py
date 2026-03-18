"""
generate_data.py
================
Generates a physically grounded synthetic dataset for TPAE copolymers
consisting of PTMO (poly(tetramethylene oxide)) soft segments and
Nylon-6,10 hard segments, using group contribution theory.

Methods
-------
- Van Krevelen group contribution theory for Tg, Tm, E
- Fox equation for copolymer Tg blending
- Rule of mixtures for modulus blending
- Gaussian noise added to simulate experimental scatter

Compositions studied (PTMO wt%): 5, 10, 20, 30, 40, 60
Complement is Nylon-6,10:          95, 90, 80, 70, 60, 40
"""

import numpy as np
import pandas as pd
import os

np.random.seed(42)

# ── Group contribution estimates ─────────────────────────────────────────────
# Nylon 6,10 repeat unit: -(NH-(CH2)6-NH-CO-(CH2)8-CO)-
# Van Krevelen estimates (literature-backed)
TG_NYLON610   = 323.0   # K  (~50 °C)
TM_NYLON610   = 494.0   # K  (~221 °C)
E_NYLON610    = 2800.0  # MPa (rubbery above Tg, glassy below; we use 25 °C value)

# PTMO repeat unit: -(O-(CH2)4)- , Mn ~ 1000-2000 g/mol soft segment
# Soft, low-Tg rubbery block
TG_PTMO       = 190.0   # K  (~-83 °C)
TM_PTMO       = 315.0   # K  (approx, for high-MW PTMO)
E_PTMO        = 5.0     # MPa (very soft rubbery block)

# ── Composition grid ──────────────────────────────────────────────────────────
PTMO_fractions = np.array([0.05, 0.10, 0.20, 0.30, 0.40, 0.60])
NYLON_fractions = 1.0 - PTMO_fractions

# Fine grid for smooth curves (used in plotting)
PTMO_fine = np.linspace(0.01, 0.65, 200)
NYLON_fine = 1.0 - PTMO_fine

# ── Property models ──────────────────────────────────────────────────────────

def fox_tg(w_ptmo, w_nylon, tg_ptmo=TG_PTMO, tg_nylon=TG_NYLON610):
    """Fox equation for copolymer glass transition temperature (K)."""
    return 1.0 / (w_ptmo / tg_ptmo + w_nylon / tg_nylon)

def linear_tm(w_ptmo, w_nylon, tm_ptmo=TM_PTMO, tm_nylon=TM_NYLON610):
    """Linear rule of mixtures for melting temperature (K)."""
    return w_ptmo * tm_ptmo + w_nylon * tm_nylon

def log_modulus(w_ptmo, w_nylon, e_ptmo=E_PTMO, e_nylon=E_NYLON610):
    """
    Log rule of mixtures for elastic modulus (MPa).
    More physically appropriate than linear mixing for soft/hard composites.
    """
    log_E = w_ptmo * np.log(e_ptmo) + w_nylon * np.log(e_nylon)
    return np.exp(log_E)

def thermal_conductivity(w_ptmo):
    """
    Empirical thermal conductivity estimate (W/m·K).
    Nylon-6,10: ~0.28 W/m·K; PTMO foam: ~0.04 W/m·K (closed-cell air pockets)
    Linear interpolation between limits.
    """
    k_nylon = 0.28
    k_ptmo  = 0.04
    return w_ptmo * k_ptmo + (1 - w_ptmo) * k_nylon

# ── Generate dataset at the 6 experimental compositions ──────────────────────
noise_level = 0.03  # 3% Gaussian noise to simulate experimental scatter

Tg_true   = fox_tg(PTMO_fractions, NYLON_fractions)
Tm_true   = linear_tm(PTMO_fractions, NYLON_fractions)
E_true    = log_modulus(PTMO_fractions, NYLON_fractions)
k_true    = thermal_conductivity(PTMO_fractions)

Tg_noisy  = Tg_true  * (1 + np.random.normal(0, noise_level, len(PTMO_fractions)))
Tm_noisy  = Tm_true  * (1 + np.random.normal(0, noise_level, len(PTMO_fractions)))
E_noisy   = E_true   * (1 + np.random.normal(0, noise_level * 2, len(PTMO_fractions)))
k_noisy   = k_true   * (1 + np.random.normal(0, noise_level, len(PTMO_fractions)))

df = pd.DataFrame({
    "PTMO_wt_frac":    PTMO_fractions,
    "Nylon610_wt_frac": NYLON_fractions,
    "Tg_K":            Tg_noisy,
    "Tg_C":            Tg_noisy - 273.15,
    "Tm_K":            Tm_noisy,
    "Tm_C":            Tm_noisy - 273.15,
    "E_MPa":           E_noisy,
    "k_W_mK":          k_noisy,
})

os.makedirs("data", exist_ok=True)
df.to_csv("data/tpae_properties.csv", index=False)
print("Saved data/tpae_properties.csv")
print(df.to_string(index=False, float_format="{:.3f}".format))

# ── Also save fine-grid ground truth for plotting ────────────────────────────
df_fine = pd.DataFrame({
    "PTMO_wt_frac":     PTMO_fine,
    "Nylon610_wt_frac": NYLON_fine,
    "Tg_K":             fox_tg(PTMO_fine, NYLON_fine),
    "Tm_K":             linear_tm(PTMO_fine, NYLON_fine),
    "E_MPa":            log_modulus(PTMO_fine, NYLON_fine),
    "k_W_mK":           thermal_conductivity(PTMO_fine),
})
df_fine.to_csv("data/tpae_properties_fine.csv", index=False)
print("\nSaved data/tpae_properties_fine.csv (fine grid for plotting)")
