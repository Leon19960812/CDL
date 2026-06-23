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


# This script plots the plate/structure-prevailing CDL solution for a
# simply supported one-dimensional plate under two parameter scans:
# fixed tau with varying kappa, and fixed kappa with varying tau.

pi = np.pi

l = 1.0
h = 0.02
x = np.linspace(-0.5, 0.5, 401)

fixed_tau = pi**2
fixed_kappa = 2.0 * pi**4
kappa_ratios = [0.20, 0.40, 0.60, 0.80, 0.95]
tau_ratios = [9.00, 4.00, 2.00, 1.25, 1.00]

# Match phase.py's liquid-structure palette.
c_present_model = "#4C78A8"   # muted blue
c_classical_model = "#E69F00"  # ochre orange

non_equipoise_linestyle = "--"
output_dir = Path("Picture") / "structure_prevailing"


def lambda1_from_tau_lambda2(tau, lambda2):
    return np.sqrt(lambda2**2 + tau)


def kappa_equipoise(tau):
    return pi**4 + pi**2 * tau


def lambdas_from_tau_kappa(tau, kappa):
    lambda2_sq = (-tau + np.sqrt(tau**2 + 4.0 * kappa)) / 2.0
    lambda2 = np.sqrt(lambda2_sq)
    lambda1 = np.sqrt(lambda2_sq + tau)
    return lambda1, lambda2


def plate_prevailing_zeta_over_h(x_over_l, tau, lambda2):
    """Eq. (4.6), normalized by h, for lambda_2 < pi."""
    lambda1 = lambda1_from_tau_lambda2(tau, lambda2)

    F = (
        lambda2**2 / lambda1 * np.tanh(lambda1 / 2.0)
        + lambda1**2 / lambda2 * np.tan(lambda2 / 2.0)
    )

    hyperbolic_part = (
        lambda2**2
        / (2.0 * F * np.cosh(lambda1 / 2.0))
        * np.cosh(lambda1 * x_over_l)
    )
    trigonometric_part = (
        lambda1**2
        / (2.0 * F * np.cos(lambda2 / 2.0))
        * np.cos(lambda2 * x_over_l)
    )
    offset = (lambda1**2 + lambda2**2) / (2.0 * F)

    return hyperbolic_part + trigonometric_part - offset


def classical_zeta_over_h(x_over_l, tau, kappa):
    """Classical constant-load solution of y'''' - tau y'' = kappa."""
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


def plot_fixed_tau_scan_kappa(ax):
    kappa_eq = kappa_equipoise(fixed_tau)

    for ratio in kappa_ratios:
        kappa = ratio * kappa_eq
        _, lambda2 = lambdas_from_tau_kappa(fixed_tau, kappa)
        z = plate_prevailing_zeta_over_h(x, fixed_tau, lambda2)
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
            label_x=0.1,
        )
        draw_labeled_curve(
            ax,
            x,
            z,
            label,
            c_present_model,
            non_equipoise_linestyle,
            1.5,
            label_x=0.0,
        )

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

    for ratio in tau_ratios:
        tau = ratio * pi**2
        if np.isclose(tau, tau_eq):
            continue

        _, lambda2 = lambdas_from_tau_kappa(tau, fixed_kappa)
        z = plate_prevailing_zeta_over_h(x, tau, lambda2)
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
            label_x=0.1,
        )
        draw_labeled_curve(
            ax,
            x,
            z,
            label,
            c_present_model,
            non_equipoise_linestyle,
            1.5,
            label_x=0.0,
        )

    if tau_eq >= 0:
        draw_labeled_curve(
            ax,
            x,
            classical_zeta_over_h(x, tau_eq, fixed_kappa),
            rf"${tau_eq / pi**2:.2g}$",
            c_classical_model,
            "-",
            1.2,
            label_x=0.1,
        )
        draw_labeled_curve(
            ax,
            x,
            equipoise_zeta_over_h(x),
            rf"${tau_eq / pi**2:.2g}$",
            c_present_model,
            "-",
            2.2,
            label_x=0.0,
        )

    ax.set_title(rf"Fixed $\kappa=2\pi^4$")
    ax.set_xlabel(r"$x/l$")
    ax.text(
        0.03,
        0.95,
        r"labels: $\tau/\pi^2$",
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
    ax.set_ylabel(r"$\zeta/h$")
    finish_axis(ax)
    save_figure(fig, "fixed_kappa_scan_tau")


def plot_plate_prevailing():
    plot_fixed_tau_figure()
    plot_fixed_kappa_figure()


if __name__ == "__main__":
    plot_plate_prevailing()
