import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq
import scienceplots

# This script plots h2/h and a/l for simply-supported and clamped plates.
plt.style.use(['science', 'notebook', 'grid'])

# ----------------------------
# Color semantics
# ----------------------------
COLOR_PLATE_SIMPLY = 'tab:orange'
COLOR_PLATE_CLAMPED = '#C45A00'
COLOR_CLASSICAL = 'tab:red'

COLOR_EXP_SIMPLY = '#3B6DBD'   # blue experimental points
COLOR_EXP_CLAMPED = 'red'      # red experimental points

l = 1.0

# =========================================================
# 1) Simply supported
# =========================================================
def det_simply(beta, lam):
    """det(beta, lam)=0 gives beta in (0, lam/2)."""
    d1 = (beta - lam / 2) * np.sinh(beta) - np.cosh(beta)
    d2 = (beta - lam / 2) * np.sin(beta) + np.cos(beta)
    return d1 * np.cos(beta) - d2 * np.cosh(beta)


def solve_beta_simply(lam):
    f = lambda b: det_simply(b, lam)
    low, high = 1e-8, lam / 2 - 1e-6

    try:
        return brentq(f, low, high)
    except ValueError:
        scan = np.linspace(low, high, 800)
        vals = det_simply(scan, lam)
        sign = np.sign(vals)
        idx = np.where(np.diff(sign) != 0)[0]
        if len(idx) == 0:
            return np.nan
        i = idx[0]
        return brentq(f, scan[i], scan[i + 1])


def beta_to_a(beta, lam):
    return beta * l / lam


def G_of_beta(beta):
    return (np.tan(beta) + np.tanh(beta)) / (np.tan(beta) - np.tanh(beta))


def h2_over_h_simply(lam):
    """
    h2/h = (lam^2/(2l))*delta*G + (lam^4/(6l^3))*delta^3
    """
    beta = solve_beta_simply(lam)
    if not np.isfinite(beta):
        return np.nan

    a = beta_to_a(beta, lam)
    delta = l / 2 - a
    G = G_of_beta(beta)

    return (lam**2 / (2 * l)) * delta * G + (lam**4 / (6 * l**3)) * delta**3


# =========================================================
# 2) Clamped
# =========================================================
def det_clamped(beta, lam):
    """Clamped liquid-prevails determinant."""
    s = 0.5 - beta / lam
    A11 = (s**2 * lam**2 + 2.0) * np.sinh(beta) + 2.0 * lam * s * np.cosh(beta)
    A12 = (s**2 * lam**2 - 2.0) * np.sin(beta) - 2.0 * lam * s * np.cos(beta)
    return A11 * np.cos(beta) - A12 * np.cosh(beta)


def solve_beta_clamped(lam):
    f = lambda b: det_clamped(b, lam)
    low, high = 1e-8, lam / 2 - 1e-6

    try:
        return brentq(f, low, high)
    except ValueError:
        scan = np.linspace(low, high, 1200)
        vals = det_clamped(scan, lam)
        sign = np.sign(vals)
        idx = np.where(np.diff(sign) != 0)[0]
        if len(idx) == 0:
            return np.nan
        i = idx[0]
        return brentq(f, scan[i], scan[i + 1])


def H_of_beta(beta):
    return 1.0 / (np.tanh(beta) - np.tan(beta))


def h2_over_h_clamped(lam):
    """
    h2/h = (lam^4 s^3)/6 + (lam^3 s^2)/2 * 1/(tanh(beta) - tan(beta))
    """
    beta = solve_beta_clamped(lam)
    if not np.isfinite(beta):
        return np.nan

    s = 0.5 - beta / lam
    return (lam**4 * s**3) / 6.0 + (lam**3 * s**2) / 2.0 * H_of_beta(beta)


# =========================================================
# 3) lambda ranges + compute arrays
# =========================================================
eps = 1e-3
lam_min_simply = np.pi + eps


def coth(x):
    return np.cosh(x) / np.sinh(x)


def cot(x):
    return np.cos(x) / np.sin(x)


f_eq_clamped = lambda lam: coth(lam / 2) + cot(lam / 2)
lam_eq_clamped = brentq(f_eq_clamped, np.pi + 1e-3, 2 * np.pi - 1e-3)

lam_max = 6.0
lam_simply = np.linspace(lam_min_simply, min(lam_max, 7.0), 500)
lam_clamped = np.linspace(lam_eq_clamped + eps, min(lam_max, 7.0), 500)

h2_simply = np.array([h2_over_h_simply(lam) for lam in lam_simply])
h2_clamped = np.array([h2_over_h_clamped(lam) for lam in lam_clamped])

a_over_l_simply = np.array([beta_to_a(solve_beta_simply(lam), lam) for lam in lam_simply])
a_over_l_clamped = np.array([beta_to_a(solve_beta_clamped(lam), lam) for lam in lam_clamped])

# =========================================================
# 3.1) Experimental points (digitized from current images)
# NOTE: These are approximate. If any point is off, only edit these arrays.
# =========================================================

# h2/h experimental points
lam_exp_h2_simply = np.array([3.2, 3.8, 4.20, 4.45])
h2_exp_simply = np.array([0.10, 1.40, 2.80, 3.90])

lam_exp_h2_clamped = np.array([4.80, 4.95, 5.15, 5.40, 5.60, 5.80, 5.95])
h2_exp_clamped = np.array([0.05, 0.20, 0.50, 1.00, 1.52, 2.02, 2.52])

# a/l experimental points
lam_exp_a_simply = np.array([3.35, 3.70, 4.10, 4.50, 5.30])
a_exp_simply = np.array([0.455, 0.385, 0.335, 0.300, 0.250])

lam_exp_a_clamped = np.array([4.80, 4.95, 5.15, 5.35, 5.60, 5.80, 5.95])
a_exp_clamped = np.array([0.452, 0.405, 0.367, 0.336, 0.300, 0.283, 0.268])

# =========================================================
# 4) Plot h2/h
# =========================================================
fig, ax = plt.subplots(figsize=(7, 5))

ax.plot(np.pi, 0, 'o', color='k', markersize=6, zorder=5)
ax.annotate('Equipoise (simply)', xy=(np.pi, 0), xytext=(np.pi, 2),
            arrowprops=dict(arrowstyle='-|>', lw=0.8), fontsize=14)

ax.plot(lam_eq_clamped, 0, 'o', color='k', markersize=6, zorder=5)
ax.annotate('Equipoise (clamped)', xy=(lam_eq_clamped, 0), xytext=(lam_eq_clamped - 0.2, 2),
            arrowprops=dict(arrowstyle='-|>', lw=0.8), fontsize=14)

ax.plot(lam_simply, h2_simply, color=COLOR_PLATE_SIMPLY, lw=2.0,
        label='Present model(simply)')
ax.plot(lam_clamped, h2_clamped, color=COLOR_PLATE_CLAMPED, lw=2.0,
        label='Present model(clamped)')

# Experimental points
ax.scatter(lam_exp_h2_simply, h2_exp_simply, marker='s', s=52,
           facecolors=COLOR_EXP_SIMPLY, edgecolors='#1f3e75', linewidths=0.8,
           label='Experiment(simply)', zorder=6)
ax.scatter(lam_exp_h2_clamped, h2_exp_clamped, marker='s', s=52,
           facecolors=COLOR_EXP_CLAMPED, edgecolors='#7a0000', linewidths=0.8,
           label='Experiment(clamped)', zorder=6)

lam_all = np.linspace(0.0, lam_max, 400)
ax.plot(lam_all, np.zeros_like(lam_all), color=COLOR_CLASSICAL, ls=':', lw=2.0,
        label='Classical model')

ax.set_xlabel(r'$\lambda$')
ax.set_ylabel(r'$h_2/h$')
ax.set_xlim(lam_min_simply - 0.05, lam_max)
ax.set_ylim(bottom=-0.2, top=5)
ax.legend(loc='upper right', frameon=True, fontsize=11)
plt.tight_layout()
plt.show()

# =========================================================
# 5) Plot a/l
# =========================================================
fig, ax = plt.subplots(figsize=(7, 5))

ax.plot(np.pi, 0.5, 'o', color='k', markersize=6, zorder=5)
ax.annotate('Equipoise (simply)', xy=(np.pi, 0.5), xytext=(np.pi + 0.5, 0.45),
            arrowprops=dict(arrowstyle='-|>', lw=0.8), fontsize=14)

ax.plot(lam_eq_clamped, 0.5, 'o', color='k', markersize=6, zorder=5)
ax.annotate('Equipoise (clamped)', xy=(lam_eq_clamped, 0.5), xytext=(lam_eq_clamped + 0.2, 0.45),
            arrowprops=dict(arrowstyle='-|>', lw=0.8), fontsize=14)

ax.plot(lam_simply, a_over_l_simply, color=COLOR_PLATE_SIMPLY, lw=2.0,
        label='Present model(simply)')
ax.plot(lam_clamped, a_over_l_clamped, color=COLOR_PLATE_CLAMPED, lw=2.0,
        label='Present model(clamped)')

# Experimental points
ax.scatter(lam_exp_a_simply, a_exp_simply, marker='s', s=52,
           facecolors=COLOR_EXP_SIMPLY, edgecolors='#1f3e75', linewidths=0.8,
           label='Experiment(simply)', zorder=6)
ax.scatter(lam_exp_a_clamped, a_exp_clamped, marker='s', s=52,
           facecolors=COLOR_EXP_CLAMPED, edgecolors='#7a0000', linewidths=0.8,
           label='Experiment(clamped)', zorder=6)

ax.plot(lam_all, 0.5 * np.ones_like(lam_all), color=COLOR_CLASSICAL, ls=':', lw=2.0,
        label='Classical model')

ax.set_xlabel(r'$\lambda$')
ax.set_ylabel(r'$a/l$')
ax.set_xlim(lam_min_simply - 0.05, lam_max)
ax.legend(loc='lower left', frameon=True, bbox_to_anchor=(0.00, 0.00), fontsize=11)
plt.tight_layout()
plt.show()

print(f'Clamped equipoise lambda ~= {lam_eq_clamped:.6f}')
