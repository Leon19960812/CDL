import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# Basic settings
# ============================================================
pi = np.pi

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["mathtext.fontset"] = "dejavusans"
plt.rcParams["font.size"] = 11

# ----------------------------
# Colors
# ----------------------------
# Liquid-structure partition
c_structure = "#4C78A8"   # muted blue
c_liquid = "#E69F00"      # ochre orange

# Bending-tension partition
c_bending = "#009E73"     # bluish green
c_tension = "#CC79A7"     # muted purple/magenta

c_invalid = "#D9D9D9"     # neutral gray
line_color = "#222222"
secondary_line_color = "#666666"
secondary_line_style = "-."
secondary_line_width = 1.4
secondary_line_alpha = 0.8
fill_alpha = 0.6
invalid_alpha = 0.6
output_dir = Path("Picture") / "Regime"


def finish_figure(fig, ax, filename):
    output_dir.mkdir(parents=True, exist_ok=True)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / f"{filename}.png", dpi=600, bbox_inches="tight")
    # fig.savefig(output_dir / f"{filename}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_liquid_structure_tau_kappa():
    fig, ax = plt.subplots(figsize=(5.4, 4.4))

    tau_min, tau_max = 0, 25
    kappa_min, kappa_max = 0, 380

    tau = np.linspace(tau_min, tau_max, 600)
    kappa_eq = pi**4 + pi**2 * tau
    tau_star = pi**2

    ax.fill_between(
        tau, kappa_min, np.minimum(kappa_eq, kappa_max),
        color=c_structure, alpha=fill_alpha
    )
    ax.fill_between(
        tau, np.minimum(kappa_eq, kappa_max), kappa_max,
        color=c_liquid, alpha=fill_alpha
    )

    ax.plot(tau, kappa_eq, color=line_color, linewidth=2.2)
    ax.axvline(
        tau_star, color=secondary_line_color, linestyle=secondary_line_style,
        linewidth=secondary_line_width, alpha=secondary_line_alpha
    )

    ax.text(4.0, 90, "Overfilled", fontsize=11)
    ax.text(4.0, 280, "Under-filled", fontsize=11)
    ax.text(13.0, 210, r'$\kappa=\pi^4+\pi^2\tau$', fontsize=11)
    ax.text(tau_star + 0.35, 40, r'$\tau=\pi^2$', fontsize=10,
            color=secondary_line_color)

    ax.set_xlim(tau_min, tau_max)
    ax.set_ylim(kappa_min, kappa_max)
    ax.set_xlabel(r'$\tau$')
    ax.set_ylabel(r'$\kappa$')

    finish_figure(fig, ax, "phase_liquid_structure_tau_kappa")


def plot_liquid_structure_lambda2_lambda1():
    fig, ax = plt.subplots(figsize=(5.4, 4.4))

    x_min, x_max = 0, 8
    y_min, y_max = 0, 8

    lam2 = np.linspace(x_min, x_max, 800)
    diag = lam2
    boundary_x = pi
    balance = np.sqrt(lam2**2 + pi**2)

    ax.fill_between(
        lam2, y_min, np.minimum(diag, y_max),
        color=c_invalid, alpha=invalid_alpha
    )

    mask_left = lam2 <= boundary_x
    ax.fill_between(
        lam2[mask_left], diag[mask_left], y_max,
        color=c_structure, alpha=fill_alpha
    )

    mask_right = lam2 >= boundary_x
    ax.fill_between(
        lam2[mask_right], diag[mask_right], y_max,
        color=c_liquid, alpha=fill_alpha
    )

    ax.axvline(boundary_x, color=line_color, linewidth=2.2)
    ax.plot(
        lam2, balance, color=secondary_line_color,
        linestyle=secondary_line_style, linewidth=secondary_line_width,
        alpha=secondary_line_alpha
    )
    ax.plot(lam2, diag, color="gray", linestyle="--", linewidth=1.4)

    ax.text(0.1, 6.8, "Overfilled", fontsize=11)
    ax.text(3.20, 6.8, "Under-filled", fontsize=11)
    ax.text(pi + 0.15, 1.0, r'$\lambda_2=\pi$', fontsize=11)
    ax.text(3.5, 6.0, r'$\lambda_1^2-\lambda_2^2=\pi^2$', fontsize=10,
            color=secondary_line_color)
    ax.text(
        5.2, 4.5, r'$\lambda_1=\lambda_2$',
        fontsize=10, color="gray", rotation=35
    )

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel(r'$\lambda_2$')
    ax.set_ylabel(r'$\lambda_1$')

    finish_figure(fig, ax, "phase_liquid_structure_lambda2_lambda1")


def plot_liquid_structure_lambdam_lambdap():
    fig, ax = plt.subplots(figsize=(5.4, 4.4))

    xm_min, xm_max = 0.5, 12
    yp_min, yp_max = 0, 8

    lam_m = np.linspace(xm_min, xm_max, 1500)
    lam_p_balance = np.sqrt(pi * lam_m)

    lam_m_right = lam_m[lam_m > pi + 1e-6]
    lam_p_eq = pi / (1 - pi**2 / lam_m_right**2)**0.25

    mask_left = lam_m <= pi
    ax.fill_between(
        lam_m[mask_left], yp_min, yp_max,
        color=c_structure, alpha=fill_alpha
    )

    lam_p_eq_clip = np.minimum(lam_p_eq, yp_max)

    ax.fill_between(
        lam_m_right, yp_min, lam_p_eq_clip,
        color=c_structure, alpha=fill_alpha
    )

    mask_curve_visible = lam_p_eq < yp_max
    ax.fill_between(
        lam_m_right[mask_curve_visible], lam_p_eq[mask_curve_visible], yp_max,
        color=c_liquid, alpha=fill_alpha
    )

    ax.plot(lam_m_right, lam_p_eq, color=line_color, linewidth=2.2)
    ax.plot(
        lam_m, lam_p_balance, color=secondary_line_color,
        linestyle=secondary_line_style, linewidth=secondary_line_width,
        alpha=secondary_line_alpha
    )
    ax.axvline(pi, color="gray", linestyle="--", linewidth=1.5)

    ax.text(1.2, 2.0, "Overfilled", fontsize=11)
    ax.text(6.5, 6.8, "Under-filled", fontsize=11)
    ax.text(
        5.5, 3.7,
        r'$\lambda_p^4\left(1-\pi^2/\lambda_m^2\right)=\pi^4$',
        fontsize=10
    )
    ax.text(5.8, 5.0, r'$\lambda_p^2=\pi\lambda_m$', fontsize=10,
            color=secondary_line_color)
    ax.text(pi + 0.12, 0.7, r'$\lambda_m=\pi$', fontsize=10, color="gray")

    ax.set_xlim(xm_min, xm_max)
    ax.set_ylim(yp_min, yp_max)
    ax.set_xlabel(r'$\lambda_m=(\rho g l^2/T)^{1/2}$')
    ax.set_ylabel(r'$\lambda_p=(\rho g l^4/D)^{1/4}$')

    finish_figure(fig, ax, "phase_liquid_structure_lambdam_lambdap")


def plot_bending_tension_tau_kappa():
    fig, ax = plt.subplots(figsize=(5.4, 4.4))

    tau_min, tau_max = 0, 25
    kappa_min, kappa_max = 0, 380

    tau = np.linspace(tau_min, tau_max, 600)
    tau_star = pi**2
    kappa_eq = pi**4 + pi**2 * tau

    ax.fill_between(
        tau[tau <= tau_star], kappa_min, kappa_max,
        color=c_bending, alpha=fill_alpha
    )
    ax.fill_between(
        tau[tau >= tau_star], kappa_min, kappa_max,
        color=c_tension, alpha=fill_alpha
    )

    ax.axvline(tau_star, color=line_color, linewidth=2.2)
    ax.plot(
        tau, kappa_eq, color=secondary_line_color,
        linestyle=secondary_line_style, linewidth=secondary_line_width,
        alpha=secondary_line_alpha
    )

    ax.text(1.0, 320, "Bending-prevailing", fontsize=11)
    ax.text(tau_star + 0.35, 320, "Tension-prevailing", fontsize=11)
    ax.text(tau_star + 0.35, 40, r'$\tau=\pi^2$', fontsize=11)
    ax.text(13.0, 210, r'$\kappa=\pi^4+\pi^2\tau$', fontsize=10,
            color=secondary_line_color)

    ax.set_xlim(tau_min, tau_max)
    ax.set_ylim(kappa_min, kappa_max)
    ax.set_xlabel(r'$\tau$')
    ax.set_ylabel(r'$\kappa$')

    finish_figure(fig, ax, "phase_bending_tension_tau_kappa")


def plot_bending_tension_lambda2_lambda1():
    fig, ax = plt.subplots(figsize=(5.4, 4.4))

    x_min, x_max = 0, 8
    y_min, y_max = 0, 8

    lam2 = np.linspace(x_min, x_max, 1200)
    diag = lam2
    balance = np.sqrt(lam2**2 + pi**2)

    ax.fill_between(
        lam2, y_min, np.minimum(diag, y_max),
        color=c_invalid, alpha=invalid_alpha
    )

    upper_bending = np.minimum(balance, y_max)
    ax.fill_between(
        lam2, diag, upper_bending,
        color=c_bending, alpha=fill_alpha
    )

    mask_tension = balance < y_max
    ax.fill_between(
        lam2[mask_tension], balance[mask_tension], y_max,
        color=c_tension, alpha=fill_alpha
    )

    ax.plot(lam2, balance, color=line_color, linewidth=2.2)
    ax.axvline(
        pi, color=secondary_line_color, linestyle=secondary_line_style,
        linewidth=secondary_line_width, alpha=secondary_line_alpha
    )
    ax.plot(lam2, diag, color="gray", linestyle="--", linewidth=1.4)

    ax.text(0.1, 2.8, "Bending-prevailing", fontsize=11)
    ax.text(0.1, 6.7, "Tension-prevailing", fontsize=11)
    ax.text(4.3, 6.5, r'$\lambda_1^2-\lambda_2^2=\pi^2$', fontsize=10)
    ax.text(pi + 0.15, 1.0, r'$\lambda_2=\pi$', fontsize=10,
            color=secondary_line_color)
    ax.text(
        5.2, 4.5, r'$\lambda_1=\lambda_2$',
        fontsize=10, color="gray", rotation=35
    )

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel(r'$\lambda_2$')
    ax.set_ylabel(r'$\lambda_1$')

    finish_figure(fig, ax, "phase_bending_tension_lambda2_lambda1")


def plot_bending_tension_lambdam_lambdap():
    fig, ax = plt.subplots(figsize=(5.4, 4.4))

    xm_min, xm_max = 0.5, 12
    yp_min, yp_max = 0, 8

    lam_m = np.linspace(xm_min, xm_max, 1500)
    lam_p_balance = np.sqrt(pi * lam_m)
    lam_m_right = lam_m[lam_m > pi + 1e-6]
    lam_p_eq = pi / (1 - pi**2 / lam_m_right**2)**0.25

    ax.fill_between(
        lam_m, yp_min, np.minimum(lam_p_balance, yp_max),
        color=c_bending, alpha=fill_alpha
    )

    mask_tension = lam_p_balance < yp_max
    ax.fill_between(
        lam_m[mask_tension], lam_p_balance[mask_tension], yp_max,
        color=c_tension, alpha=fill_alpha
    )

    ax.plot(lam_m, lam_p_balance, color=line_color, linewidth=2.2)
    ax.plot(
        lam_m_right, lam_p_eq, color=secondary_line_color,
        linestyle=secondary_line_style, linewidth=secondary_line_width,
        alpha=secondary_line_alpha
    )

    ax.text(1.5, 1.7, "Bending-prevailing", fontsize=11)
    ax.text(1.5, 6.6, "Tension-prevailing", fontsize=11)
    ax.text(5.8, 5.0, r'$\lambda_p^2=\pi\lambda_m$', fontsize=11)
    ax.text(
        5.5, 3.7,
        r'$\lambda_p^4\left(1-\pi^2/\lambda_m^2\right)=\pi^4$',
        fontsize=10, color=secondary_line_color
    )

    ax.set_xlim(xm_min, xm_max)
    ax.set_ylim(yp_min, yp_max)
    ax.set_xlabel(r'$\lambda_m=(\rho g l^2/T)^{1/2}$')
    ax.set_ylabel(r'$\lambda_p=(\rho g l^4/D)^{1/4}$')

    finish_figure(fig, ax, "phase_bending_tension_lambdam_lambdap")


if __name__ == "__main__":
    plot_liquid_structure_tau_kappa()
    plot_liquid_structure_lambda2_lambda1()
    plot_liquid_structure_lambdam_lambdap()
    plot_bending_tension_tau_kappa()
    plot_bending_tension_lambda2_lambda1()
    plot_bending_tension_lambdam_lambdap()
