"""
models.py
=========
Compares three approaches to predicting TPAE properties from composition:
  1. Physics-only  — group contribution theory (no ML)
  2. Baseline NN   — fully data-driven MLP, no physics
  3. Physics-guided NN — MLP trained on residuals from physics model (delta learning)

Delta learning is a standard Physics-Informed/guided strategy:
  y_pred = y_physics(x) + NN(x)
The NN only needs to learn the deviation from physics, which is much
easier with small datasets — directly relevant to our 6-sample scenario.

All models are evaluated with leave-one-out cross-validation (LOOCV)
because n=6 is too small for a proper train/test split.
"""

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
warnings.filterwarnings("ignore")

# ── Load data ─────────────────────────────────────────────────────────────────
df = pd.read_csv("data/tpae_properties.csv")
X = df[["PTMO_wt_frac"]].values          # single input: PTMO weight fraction
targets = ["Tg_K", "Tm_K", "E_MPa"]
target_labels = {
    "Tg_K":  "Glass Transition Temp (K)",
    "Tm_K":  "Melting Temp (K)",
    "E_MPa": "Elastic Modulus (MPa)",
}

# ── Group contribution physics predictions (same as generate_data.py) ─────────
def fox_tg(w):
    return 1.0 / (w / 190.0 + (1 - w) / 323.0)

def linear_tm(w):
    return w * 315.0 + (1 - w) * 494.0

def log_modulus(w):
    return np.exp(w * np.log(5.0) + (1 - w) * np.log(2800.0))

physics_fns = {"Tg_K": fox_tg, "Tm_K": linear_tm, "E_MPa": log_modulus}


# ── LOOCV evaluation ──────────────────────────────────────────────────────────
def loocv(X, y, model_fn):
    """Leave-one-out CV. model_fn(X_train, y_train, X_test) -> y_pred scalar."""
    preds = []
    for i in range(len(X)):
        mask = np.ones(len(X), dtype=bool)
        mask[i] = False
        y_pred = model_fn(X[mask], y[mask], X[[i]])
        preds.append(y_pred[0])
    return np.array(preds)


def baseline_nn_fn(X_train, y_train, X_test):
    sc = StandardScaler()
    Xt = sc.fit_transform(X_train)
    nn = MLPRegressor(hidden_layer_sizes=(16, 16), max_iter=5000,
                      random_state=0, learning_rate_init=0.01)
    nn.fit(Xt, y_train)
    return nn.predict(sc.transform(X_test))


def physics_guided_fn(phys_fn, X_train, y_train, X_test):
    """Delta learning: NN learns residual = y - physics(x)."""
    w_train = X_train[:, 0]
    w_test  = X_test[:, 0]
    res_train = y_train - phys_fn(w_train)
    sc = StandardScaler()
    Xt = sc.fit_transform(X_train)
    nn = MLPRegressor(hidden_layer_sizes=(16, 16), max_iter=5000,
                      random_state=0, learning_rate_init=0.01)
    nn.fit(Xt, res_train)
    residual_pred = nn.predict(sc.transform(X_test))
    return phys_fn(w_test) + residual_pred


# ── Run experiments ───────────────────────────────────────────────────────────
results = {}

for target in targets:
    y = df[target].values
    w = X[:, 0]
    phys_fn = physics_fns[target]

    y_physics = phys_fn(w)
    y_baseline = loocv(X, y, baseline_nn_fn)
    y_guided   = loocv(X, y,
                       lambda Xtr, ytr, Xte, fn=phys_fn:
                       physics_guided_fn(fn, Xtr, ytr, Xte))

    results[target] = {
        "y_true":    y,
        "y_physics": y_physics,
        "y_baseline": y_baseline,
        "y_guided":   y_guided,
        "mae_physics":  mean_absolute_error(y, y_physics),
        "mae_baseline": mean_absolute_error(y, y_baseline),
        "mae_guided":   mean_absolute_error(y, y_guided),
        "r2_physics":   r2_score(y, y_physics),
        "r2_baseline":  r2_score(y, y_baseline),
        "r2_guided":    r2_score(y, y_guided),
    }

# ── Print summary table ───────────────────────────────────────────────────────
print(f"\n{'='*65}")
print(f"{'Target':<12} {'Model':<20} {'MAE':>10} {'R²':>8}")
print(f"{'='*65}")
for target in targets:
    r = results[target]
    unit = "K" if "K" in target else "MPa"
    for model, mae, r2 in [
        ("Physics only",    r["mae_physics"],  r["r2_physics"]),
        ("Baseline NN",     r["mae_baseline"], r["r2_baseline"]),
        ("Physics-guided",  r["mae_guided"],   r["r2_guided"]),
    ]:
        print(f"{target:<12} {model:<20} {mae:>8.2f} {unit}  {r2:>6.3f}")
    print(f"{'-'*65}")

# ── Save results for plotting ─────────────────────────────────────────────────
rows = []
for target in targets:
    r = results[target]
    rows.append({
        "target": target,
        "mae_physics":  r["mae_physics"],
        "mae_baseline": r["mae_baseline"],
        "mae_guided":   r["mae_guided"],
        "r2_physics":   r["r2_physics"],
        "r2_baseline":  r["r2_baseline"],
        "r2_guided":    r["r2_guided"],
    })
pd.DataFrame(rows).to_csv("data/model_metrics.csv", index=False)
print("\nSaved data/model_metrics.csv")

# Save per-target predictions
for target in targets:
    r = results[target]
    pd.DataFrame({
        "PTMO_wt_frac": X[:, 0],
        "y_true":        r["y_true"],
        "y_physics":     r["y_physics"],
        "y_baseline":    r["y_baseline"],
        "y_guided":      r["y_guided"],
    }).to_csv(f"data/predictions_{target}.csv", index=False)

print("Saved per-target prediction CSVs")
