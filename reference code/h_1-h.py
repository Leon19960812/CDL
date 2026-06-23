import numpy as np
import matplotlib.pyplot as plt
import scienceplots
from scipy.optimize import brentq

plt.style.use(['science', 'notebook', 'grid'])

# ----------------------------
# Color semantics (your new logic)
# ----------------------------
COLOR_PLATE_SIMPLY   = 'tab:orange'   # plate family (simply)
COLOR_PLATE_CLAMPED  = '#C45A00'      # deep/burnt orange (plate family, clamped)
COLOR_CLASSICAL      = 'tab:red'      # classical reference
COLOR_EXP_SIMPLY     = '#3B6DBD'      # blue experimental points
COLOR_EXP_CLAMPED    = 'red'          # red experimental points
# (reserved for later) COLOR_MEMBRANE = 'tab:blue'

# ----------------------------
# Helper functions
# ----------------------------
def coth(x):
    return np.cosh(x) / np.sinh(x)

def cot(x):
    return np.cos(x) / np.sin(x)

# ----------------------------
# Simply supported:  h1/h = λ / (tan(λ/2) + tanh(λ/2))
# Equipoise at λ = π
# ----------------------------
eps = 1e-6
lam_simply = np.linspace(eps, np.pi - 1e-4, 600)
h1_over_h_simply = lam_simply / (np.tan(lam_simply / 2) + np.tanh(lam_simply / 2))
lam_eq_simply = np.pi
h1_eq_simply = 0.0

# ----------------------------
# Clamped: h1/h = (λ/4) * (coth(λ/2) + cot(λ/2))
# Equipoise when coth(λ/2) + cot(λ/2) = 0  -> first root ≈ 4.73
# ----------------------------
f_eq = lambda lam: coth(lam / 2) + cot(lam / 2)

# Root bracket: first nontrivial root is between π and 2π
lam_eq_clamped = brentq(f_eq, np.pi + 1e-3, 2*np.pi - 1e-3)
h1_eq_clamped = 0.0

lam_clamped = np.linspace(eps, lam_eq_clamped - 1e-4, 600)
h1_over_h_clamped = (lam_clamped / 4.0) * (coth(lam_clamped / 2) + cot(lam_clamped / 2))

# ----------------------------
# Experimental points:
# explicit digitized coordinates for easy manual tuning.
# ----------------------------
lam_exp_h1_simply = np.array([0.20, 0.55, 0.85, 1.05, 1.90, 2.15, 2.40, 2.80, 3.00])
h1_exp_simply = np.array([0.9900, 0.9892, 0.9856, 0.9998, 0.8786, 0.7654, 0.6380, 0.3075, 0.1242])

lam_exp_h1_clamped = np.array([0.85, 1.55, 2.30, 3.00, 3.50, 3.70, 4.40, 4.60])
h1_exp_clamped = np.array([0.9893, 0.9820, 0.9505, 0.8633, 0.7306, 0.6589, 0.2757, 0.1358])

# ----------------------------
# Classical: h1/h = 1 (configuration-independent)
# Use a common λ-range that covers both curves up to clamped equipoise
# ----------------------------
lam_max = lam_eq_clamped
lam_classical = np.linspace(0.0, lam_max, 400)
h1_over_h_classical = np.ones_like(lam_classical)

# ----------------------------
# Plot
# ----------------------------
fig, ax = plt.subplots(figsize=(7, 5))

# Present (CDL) — plate family, two boundary conditions
ax.plot(lam_simply,  h1_over_h_simply,  color=COLOR_PLATE_SIMPLY,  lw=2.0,
        label='Plate (simply), present model')
ax.plot(lam_clamped, h1_over_h_clamped, color=COLOR_PLATE_CLAMPED, lw=2.0,
        label='Plate (clamped), present model')

# Classical reference
ax.plot(lam_classical, h1_over_h_classical, color=COLOR_CLASSICAL, ls=':', lw=2.0,
        label='Classical model')

# Experimental points
ax.scatter(lam_exp_h1_simply, h1_exp_simply, marker='s', s=52,
           facecolors=COLOR_EXP_SIMPLY, edgecolors='#1f3e75', linewidths=0.8,
           label='Experiment(simply)', zorder=6)
ax.scatter(lam_exp_h1_clamped, h1_exp_clamped, marker='s', s=52,
           facecolors=COLOR_EXP_CLAMPED, edgecolors='#7a0000', linewidths=0.8,
           label='Experiment(clamped)', zorder=6)

# Equipoise markers
ax.plot(lam_eq_simply,  h1_eq_simply,  'o', color='k', ms=6, zorder=5)
ax.plot(lam_eq_clamped, h1_eq_clamped, 'o', color='k', ms=6, zorder=5)

# Annotations (two equipoise points)
ax.annotate(
    r'Equipoise (simply)',
    xy=(lam_eq_simply, h1_eq_simply),
    xytext=(lam_eq_simply - 2.2, 0.45),
    arrowprops=dict(arrowstyle='-|>', lw=0.8),
    fontsize=16
)
ax.annotate(
    r'Equipoise (clamped)',
    xy=(lam_eq_clamped, h1_eq_clamped),
    xytext=(lam_eq_clamped - 1.8, 0.45),
    arrowprops=dict(arrowstyle='-|>', lw=0.8),
    fontsize=16
)

# Axes
ax.set_xlabel(r'$\lambda$')
ax.set_ylabel(r'$h_1/h$')
ax.set_xlim(0.0, lam_max+0.05)
ax.set_ylim(-0.05, 1.05)

ax.legend(fontsize=12)
plt.tight_layout()
plt.show()

print(f"Clamped equipoise root: lambda ≈ {lam_eq_clamped:.6f}")
