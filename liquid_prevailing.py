import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter
from pathlib import Path

try:
    import scienceplots  # noqa: F401
    plt.style.use(["science", "notebook", "grid"])
except ImportError:
    plt.style.use("default")


# This script plots the liquid-prevailing CDL solution for a simply
# supported one-dimensional Foppl plate under two parameter scans:
# fixed tau with varying kappa, and fixed kappa with varying tau.

pi = np.pi

l = 1.0
h = 0.02
x = np.linspace(-0.5, 0.5, 401)

fixed_tau = pi**2
fixed_kappa = 2.0 * pi**4
kappa_ratios = [1.25, 1.50, 2.00, 4.00]
tau_ratios = [0.80, 0.60, 0.40, 0.20]

kappa_present_label_x = [0.0, 0.0, 0.0, 0.0]
kappa_classical_label_x = [0.1, 0.1, 0.1, 0.1]
tau_present_label_x = [0.0, 0.0, 0.0, 0.0]
tau_classical_label_x = [0.1, 0.1, 0.1, 0.1]

# Match structure_prevail.py / phase.py's liquid-structure palette.
c_present_model = "#4C78A8"   # muted blue
c_classical_model = "#E69F00"  # ochre orange

non_equipoise_linestyle = "--"
output_dir = Path("Picture") / "liquid_prevailing"


def kappa_equipoise(tau):
    return pi**4 + pi**2 * tau


def lambdas_from_tau_kappa(tau, kappa):
    lambda2_sq = (-tau + np.sqrt(tau**2 + 4.0 * kappa)) / 2.0
    lambda2 = np.sqrt(lambda2_sq)
    lambda1 = np.sqrt(lambda2_sq + tau)
    return lambda1, lambda2


def coth(value):
    return 1.0 / np.tanh(value)


def free_boundary_equation(alpha, lambda1, lambda2, tau):
    """Eq. (4.18), written with alpha = a/l and l = 1."""
    mu_l = np.sqrt(tau)
    beta = 0.5 - alpha
    return (
        lambda1**3 * np.sinh(lambda1 * alpha)
        - lambda2**3 * np.cosh(lambda1 * alpha) * np.tan(lambda2 * alpha)
        + mu_l
        * (lambda1**2 + lambda2**2)
        * coth(mu_l * beta)
        * np.cosh(lambda1 * alpha)
    )


def solve_contact_half_width(tau, kappa):
    lambda1, lambda2 = lambdas_from_tau_kappa(tau, kappa)
    if lambda2 <= pi:
        raise ValueError("Liquid-prevailing solution requires lambda2 > pi.")

    eps = 1e-6
    tan_poles = [(0.5 * pi + n * pi) / lambda2 for n in range(8)]
    upper_candidates = [0.5 - eps]
    upper_candidates.extend(p - eps for p in tan_poles if eps < p < 0.5)
    upper = min(upper_candidates)

    grid = np.linspace(eps, upper, 2000)
    vals = free_boundary_equation(grid, lambda1, lambda2, tau)
    finite = np.isfinite(vals)
    grid = grid[finite]
    vals = vals[finite]

    sign_changes = np.where(np.sign(vals[:-1]) * np.sign(vals[1:]) < 0)[0]
    if len(sign_changes) == 0:
        raise RuntimeError("Could not bracket the liquid contact boundary.")

    low = grid[sign_changes[0]]
    high = grid[sign_changes[0] + 1]
    f_low = free_boundary_equation(low, lambda1, lambda2, tau)

    for _ in range(80):
        mid = 0.5 * (low + high)
        f_mid = free_boundary_equation(mid, lambda1, lambda2, tau)
        if np.sign(f_low) * np.sign(f_mid) <= 0:
            high = mid
        else:
            low = mid
            f_low = f_mid

    return 0.5 * (low + high)


def liquid_prevailing_zeta_over_h(x_over_l, tau, kappa):
    """Eq. (4.19a,b), normalized by h, for lambda_2 > pi."""
    lambda1, lambda2 = lambdas_from_tau_kappa(tau, kappa)
    alpha = solve_contact_half_width(tau, kappa)
    beta = 0.5 - alpha
    mu_l = np.sqrt(tau)

    cosh_a = np.cosh(lambda1 * alpha)
    tan_a = np.tan(lambda2 * alpha)
    G = 2.0 * lambda2 * np.sinh(lambda1 * alpha) - 2.0 * lambda1 * tan_a * cosh_a
    # Eq. definition of M has denominator mu*l^2. With l = 1,
    # mu = sqrt(T/D) = sqrt(tau), so this is mu_l, not tau.
    M = (
        lambda1 * np.sinh(lambda1 * alpha)
        + lambda2 * cosh_a * tan_a
        + (lambda1**2 + lambda2**2) / mu_l * cosh_a * coth(mu_l * beta)
    )
    scale = lambda1 * lambda2 / G

    z_contact = scale * (
        np.cosh(lambda1 * x_over_l)
        - cosh_a / np.cos(lambda2 * alpha) * np.cos(lambda2 * x_over_l)
        - beta * M
        + (lambda1**2 + lambda2**2) / tau * cosh_a
    )

    s = 0.5 - np.abs(x_over_l)
    z_free = scale * (
        -s * M
        + (lambda1**2 + lambda2**2)
        / (tau * np.sinh(mu_l * beta))
        * cosh_a
        * np.sinh(mu_l * s)
    )

    z = np.where(np.abs(x_over_l) <= alpha, z_contact, z_free)
    return z, alpha


def classical_zeta_over_h(x_over_l, tau, kappa):
    """Classical constant-load solution, Eq. (4.23), normalized by h."""
    if np.isclose(tau, 0.0):
        return kappa * (
            x_over_l**4 / 24.0
            - x_over_l**2 / 16.0
            + 5.0 / 384.0
        )

    sqrt_tau = np.sqrt(tau)
    return (
        kappa / (2.0 * tau) * (1.0 / 4.0 - x_over_l**2)
        + kappa / tau**2
        * (np.cosh(sqrt_tau * x_over_l) / np.cosh(sqrt_tau / 2.0) - 1.0)
    )


def equipoise_zeta_over_h(x_over_l):
    return pi / 2.0 * np.cos(pi * x_over_l)


def draw_labeled_curve(
    ax,
    x_vals,
    z_vals,
    label,
    color,
    linestyle,
    linewidth,
    label_x=0.0,
):
    gap = 15
    idx_label = np.argmin(np.abs(x_vals - label_x))
    z_plot = z_vals.copy()
    z_plot[idx_label - gap:idx_label + gap + 1] = np.nan

    ax.plot(
        x_vals,
        z_plot,
        color=color,
        linewidth=linewidth,
        linestyle=linestyle,
    )
    ax.text(
        x_vals[idx_label],
        z_vals[idx_label],
        label,
        fontsize=12,
        color=color,
        ha="center",
        va="center",
    )


def draw_contact_markers(ax, alpha, z_vals):
    idx_pos = np.argmin(np.abs(x - alpha))
    idx_neg = np.argmin(np.abs(x + alpha))
    ax.scatter(
        [x[idx_neg], x[idx_pos]],
        [z_vals[idx_neg], z_vals[idx_pos]],
        color=c_present_model,
        s=20,
        zorder=6,
    )


def plot_fixed_tau_scan_kappa(ax):
    kappa_eq = kappa_equipoise(fixed_tau)

    for ratio, present_label_x, classical_label_x in zip(
        kappa_ratios,
        kappa_present_label_x,
        kappa_classical_label_x,
    ):
        kappa = ratio * kappa_eq
        z, alpha = liquid_prevailing_zeta_over_h(x, fixed_tau, kappa)
        z_classical = classical_zeta_over_h(x, fixed_tau, kappa)

        label = rf"${ratio:.2f}$"
        draw_labeled_curve(
            ax,
            x,
            z_classical,
            label,
            c_classical_model,
            non_equipoise_linestyle,
            1.2,
            label_x=classical_label_x,
        )
        draw_labeled_curve(
            ax,
            x,
            z,
            label,
            c_present_model,
            non_equipoise_linestyle,
            1.5,
            label_x=present_label_x,
        )
        draw_contact_markers(ax, alpha, z)

    draw_labeled_curve(
        ax,
        x,
        classical_zeta_over_h(x, fixed_tau, kappa_eq),
        r"$1.00$",
        c_classical_model,
        "-",
        1.2,
        label_x=0.1,
    )
    draw_labeled_curve(
        ax,
        x,
        equipoise_zeta_over_h(x),
        r"$1.00$",
        c_present_model,
        "-",
        2.2,
        label_x=0.0,
    )

    ax.set_title(rf"Fixed $\tau=\pi^2$")
    ax.set_xlabel(r"$x/l$")
    ax.set_ylabel(r"$\zeta/h$")
    ax.text(
        0.03,
        0.95,
        r"labels: $\kappa/\kappa_{\mathrm{eq}}$",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
    )


def plot_fixed_kappa_scan_tau(ax):
    tau_eq = (fixed_kappa - pi**4) / pi**2

    for ratio, present_label_x, classical_label_x in zip(
        tau_ratios,
        tau_present_label_x,
        tau_classical_label_x,
    ):
        tau = ratio * tau_eq
        z, alpha = liquid_prevailing_zeta_over_h(x, tau, fixed_kappa)
        z_classical = classical_zeta_over_h(x, tau, fixed_kappa)

        label = rf"${ratio:.2g}$"
        draw_labeled_curve(
            ax,
            x,
            z_classical,
            label,
            c_classical_model,
            non_equipoise_linestyle,
            1.2,
            label_x=classical_label_x,
        )
        draw_labeled_curve(
            ax,
            x,
            z,
            label,
            c_present_model,
            non_equipoise_linestyle,
            1.5,
            label_x=present_label_x,
        )
        draw_contact_markers(ax, alpha, z)

    draw_labeled_curve(
        ax,
        x,
        classical_zeta_over_h(x, tau_eq, fixed_kappa),
        r"$1.00$",
        c_classical_model,
        "-",
        1.2,
        label_x=0.1,
    )
    draw_labeled_curve(
        ax,
        x,
        equipoise_zeta_over_h(x),
        r"$1.00$",
        c_present_model,
        "-",
        2.2,
        label_x=0.0,
    )

    ax.set_title(rf"Fixed $\kappa=2\pi^4$")
    ax.set_xlabel(r"$x/l$")
    ax.set_ylabel(r"$\zeta/h$")
    ax.text(
        0.03,
        0.95,
        r"labels: $\tau/\tau_{\mathrm{eq}}$",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
    )


def finish_axis(ax):
    ax.grid(True, alpha=0.35)
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.legend(
        handles=[
            Line2D([0], [0], color="black", linestyle="--", linewidth=1.5,
                   label="Non-equipoise"),
            Line2D([0], [0], color="black", linestyle="-", linewidth=2.2,
                   label="Equipoise"),
            Line2D([0], [0], color=c_present_model,
                   linestyle=non_equipoise_linestyle, linewidth=1.5,
                   label="Present model"),
            Line2D([0], [0], color=c_classical_model,
                   linestyle=non_equipoise_linestyle, linewidth=1.2,
                   label="Classical model"),
            Line2D([0], [0], color=c_present_model, marker="o",
                   linestyle="None", markersize=4, label=r"$x=\pm a$"),
        ],
        loc="lower right",
        frameon=True,
        fontsize=9,
    )
    ax.invert_yaxis()


def save_figure(fig, filename):
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_dir / f"{filename}.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def plot_fixed_tau_figure():
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    plot_fixed_tau_scan_kappa(ax)
    finish_axis(ax)
    save_figure(fig, "fixed_tau_scan_kappa")


def plot_fixed_kappa_figure():
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    plot_fixed_kappa_scan_tau(ax)
    finish_axis(ax)
    save_figure(fig, "fixed_kappa_scan_tau")


def plot_liquid_prevailing():
    plot_fixed_tau_figure()
    plot_fixed_kappa_figure()


if __name__ == "__main__":
    plot_liquid_prevailing()
