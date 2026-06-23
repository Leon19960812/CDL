from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter
from scipy.optimize import brentq

try:
    import scienceplots  # noqa: F401

    plt.style.use(["science", "notebook", "grid"])
except Exception:
    plt.style.use("default")

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 13,
        "axes.titlesize": 13,
        "legend.fontsize": 11,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "mathtext.fontset": "dejavuserif",
        "text.usetex": False,
    }
)


COLOR_PRESENT = "tab:orange"
COLOR_MARKER = "black"
FIGSIZE_FIXED_KAPPA_SCAN_TAU = (5.6, 4.6)
LABEL_FONT_SIZE = plt.rcParams["legend.fontsize"]
LABEL_GAP = 36
D_HATS = np.array([0.50, 0.75, 1.00, 1.25, 1.50])
T_HAT_VALUES = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
OUTPUT_DIR = Path("fig_fixed_D_tension_profiles")

RHO = 1000.0
G = 9.8
L = 0.03
KAPPA0 = 2.0 * np.pi**4
D0 = RHO * G * L**4 / KAPPA0
TAU0 = (KAPPA0 - np.pi**4) / np.pi**2
T0 = D0 * TAU0 / L**2


def coth(x):
    return 1.0 / np.tanh(x)


def kappa_for_d(d_hat):
    return KAPPA0 / d_hat


def transition_t_hat(d_hat):
    return 2.0 - d_hat


def lambdas(t_hat, d_hat=1.0):
    tau = t_hat * TAU0 / d_hat
    kappa = kappa_for_d(d_hat)
    root = np.sqrt(tau**2 + 4.0 * kappa)
    lambda_1 = np.sqrt(0.5 * (root + tau))
    lambda_2 = np.sqrt(0.5 * (root - tau))
    return lambda_1, lambda_2, tau


def find_root_by_scan(func, q_min, q_max, n=1400):
    qs = np.linspace(q_min, q_max, n)
    vals = np.array([func(q) for q in qs])
    finite = np.isfinite(vals)
    qs = qs[finite]
    vals = vals[finite]

    for i in range(len(qs) - 1):
        if vals[i] == 0.0:
            return qs[i]
        if vals[i] * vals[i + 1] < 0.0:
            return brentq(func, qs[i], qs[i + 1])

    raise RuntimeError("No physical root found.")


def solve_q_liquid(t_hat, d_hat=1.0):
    lambda_1, lambda_2, tau = lambdas(t_hat, d_hat)
    m = np.sqrt(tau)

    def free_boundary(q):
        return (
            lambda_1**3 * np.sinh(lambda_1 * q)
            - lambda_2**3 * np.cosh(lambda_1 * q) * np.tan(lambda_2 * q)
            + m
            * (lambda_1**2 + lambda_2**2)
            * coth(m * (0.5 - q))
            * np.cosh(lambda_1 * q)
        )

    q_min = 1e-8
    q_max = min(0.5 - 1e-8, np.pi / (2.0 * lambda_2) - 1e-8)
    return find_root_by_scan(free_boundary, q_min, q_max)


def h1_over_h(t_hat, d_hat=1.0):
    lambda_1, lambda_2, _ = lambdas(t_hat, d_hat)
    force_factor = (
        lambda_2**2 / lambda_1 * np.tanh(lambda_1 / 2.0)
        + lambda_1**2 / lambda_2 * np.tan(lambda_2 / 2.0)
    )
    return (lambda_1**2 + lambda_2**2) / (2.0 * force_factor)


def liquid_factors(t_hat, d_hat=1.0):
    q = solve_q_liquid(t_hat, d_hat)
    lambda_1, lambda_2, tau = lambdas(t_hat, d_hat)
    m = np.sqrt(tau)
    beta_hat = 0.5 - q

    geom_factor = (
        2.0 * lambda_2 * np.sinh(lambda_1 * q)
        - 2.0 * lambda_1 * np.tan(lambda_2 * q) * np.cosh(lambda_1 * q)
    )
    moment_hat = (
        lambda_1 * np.sinh(lambda_1 * q)
        + lambda_2 * np.cosh(lambda_1 * q) * np.tan(lambda_2 * q)
        + (lambda_1**2 + lambda_2**2)
        / m
        * np.cosh(lambda_1 * q)
        * coth(m * beta_hat)
    )
    return q, lambda_1, lambda_2, m, beta_hat, geom_factor, moment_hat


def h2_over_h_liquid(t_hat, d_hat=1.0):
    q, lambda_1, lambda_2, m, beta_hat, geom_factor, moment_hat = liquid_factors(
        t_hat, d_hat
    )
    return (
        lambda_1
        * lambda_2
        / geom_factor
        * (
            -beta_hat * moment_hat
            + (lambda_1**2 + lambda_2**2) * np.cosh(lambda_1 * q) / m**2
        )
    )


def structure_profile(x_over_l, t_hat, d_hat=1.0):
    lambda_1, lambda_2, _ = lambdas(t_hat, d_hat)
    h1_h = h1_over_h(t_hat, d_hat)
    denominator = lambda_1**2 + lambda_2**2
    return h1_h * (
        lambda_2**2
        / denominator
        * np.cosh(lambda_1 * x_over_l)
        / np.cosh(lambda_1 / 2.0)
        + lambda_1**2
        / denominator
        * np.cos(lambda_2 * x_over_l)
        / np.cos(lambda_2 / 2.0)
        - 1.0
    )


def equipoise_profile(x_over_l):
    return np.pi / 2.0 * np.cos(np.pi * x_over_l)


def liquid_profile(x_over_l, t_hat, d_hat=1.0):
    q, lambda_1, lambda_2, m, beta_hat, geom_factor, moment_hat = liquid_factors(
        t_hat, d_hat
    )
    prefactor = lambda_1 * lambda_2 / geom_factor
    lambda_sum = lambda_1**2 + lambda_2**2
    cosh_q = np.cosh(lambda_1 * q)

    inner = prefactor * (
        np.cosh(lambda_1 * x_over_l)
        - cosh_q / np.cos(lambda_2 * q) * np.cos(lambda_2 * x_over_l)
        - beta_hat * moment_hat
        + lambda_sum * cosh_q / m**2
    )

    s = 0.5 - np.abs(x_over_l)
    outer = prefactor * (
        -s * moment_hat
        + lambda_sum
        * cosh_q
        / (m**2 * np.sinh(m * beta_hat))
        * np.sinh(m * s)
    )

    profile = np.where(np.abs(x_over_l) <= q, inner, outer)
    return profile, q


def solve_q_pure_bending(d_hat=1.0):
    lambda_p = kappa_for_d(d_hat) ** 0.25

    def pure_bending_equation(q):
        beta = lambda_p * q
        return (
            (beta - lambda_p / 2.0)
            * (np.sinh(beta) * np.cos(beta) - np.sin(beta) * np.cosh(beta))
            - 2.0 * np.cosh(beta) * np.cos(beta)
        )

    q = find_root_by_scan(
        pure_bending_equation,
        1e-8,
        np.pi / (2.0 * lambda_p) - 1e-8,
    )
    return q


def pure_bending_profile(x_over_l, d_hat=1.0):
    lambda_p = kappa_for_d(d_hat) ** 0.25
    q = solve_q_pure_bending(d_hat)
    beta = lambda_p * q
    boundary = 0.5 - q
    force_factor = np.sin(beta) * np.cosh(beta) - np.cos(beta) * np.sinh(beta)
    level_factor = (np.tan(beta) + np.tanh(beta)) / (
        np.tan(beta) - np.tanh(beta)
    )

    inner_constant = (
        lambda_p**2 / 2.0 * boundary * level_factor
        + lambda_p**4 / 6.0 * boundary**3
    )
    inner = (
        -lambda_p
        * np.cos(beta)
        / (2.0 * force_factor)
        * np.cosh(lambda_p * x_over_l)
        + lambda_p
        * np.cosh(beta)
        / (2.0 * force_factor)
        * np.cos(lambda_p * x_over_l)
        + inner_constant
    )

    s = 0.5 - np.abs(x_over_l)
    outer = (
        (lambda_p**2 / 2.0 * level_factor + lambda_p**4 / 4.0 * boundary**2)
        * s
        - lambda_p**4 / 12.0 * s**3
    )

    profile = np.where(np.abs(x_over_l) <= q, inner, outer)
    return profile, q


def profile_for_t_hat(x_over_l, t_hat, d_hat=1.0):
    t_hat_transition = transition_t_hat(d_hat)

    if np.isclose(t_hat, 0.0):
        profile, q = pure_bending_profile(x_over_l, d_hat)
        return profile, q, "liquid"
    if t_hat_transition >= 0.0 and t_hat < t_hat_transition:
        profile, q = liquid_profile(x_over_l, t_hat, d_hat)
        return profile, q, "liquid"
    if t_hat_transition >= 0.0 and np.isclose(t_hat, t_hat_transition):
        return equipoise_profile(x_over_l), None, "equipoise"
    return structure_profile(x_over_l, t_hat, d_hat), None, "structure"


def plot_labeled_curve(ax, x, z, t_hat, branch, label_x):
    if branch == "liquid":
        linestyle = "--"
        linewidth = 1.7
    elif branch == "equipoise":
        linestyle = "-"
        linewidth = 2.4
    else:
        linestyle = "-."
        linewidth = 1.7

    z_plot = z.copy()
    label_index = int(np.argmin(np.abs(x - label_x)))
    gap = LABEL_GAP
    z_plot[max(0, label_index - gap) : label_index + gap + 1] = np.nan

    ax.plot(x, z_plot, color=COLOR_PRESENT, lw=linewidth, ls=linestyle)
    ax.text(
        x[label_index],
        z[label_index],
        rf"${t_hat:g}$",
        color=COLOR_PRESENT,
        ha="center",
        va="center",
        fontsize=LABEL_FONT_SIZE,
    )


def legend_handles():
    return [
        Line2D(
            [0],
            [0],
            color=COLOR_PRESENT,
            lw=2.4,
            ls="-",
            label="flushfill",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=COLOR_PRESENT,
            markeredgecolor=COLOR_MARKER,
            markersize=6,
            label=r"$x/l=\pm a/l$",
        ),
    ]


def plot_profile_for_d(d_hat, output_dir=OUTPUT_DIR):
    x = np.linspace(-0.5, 0.5, 1201)
    curves = []

    for t_hat in T_HAT_VALUES:
        z, q, branch = profile_for_t_hat(x, t_hat, d_hat)
        curves.append((t_hat, z, q, branch))

    panel_z = np.concatenate([curve[1] for curve in curves])
    y_min = min(-0.08, float(np.nanmin(panel_z)) * 1.08)
    y_max = max(0.0, float(np.nanmax(panel_z)) * 1.08)

    fig, ax = plt.subplots(figsize=FIGSIZE_FIXED_KAPPA_SCAN_TAU)

    for t_hat, z, q, branch in curves:
        plot_labeled_curve(ax, x, z, t_hat, branch, label_x=0.0)

        if branch == "liquid":
            z_contact = np.interp(q, x, z)
            ax.scatter(
                [-q, q],
                [z_contact, z_contact],
                s=28,
                facecolors=COLOR_PRESENT,
                edgecolors=COLOR_MARKER,
                linewidths=0.6,
                zorder=6,
            )

    ax.axhline(0.0, color="0.3", lw=0.8)
    ax.set_title(rf"$D/D_0={d_hat:.2f}$")
    ax.set_xlabel(r"$x/l$")
    ax.set_ylabel(r"$\zeta/h$")
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(y_min, y_max)
    ax.invert_yaxis()
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.legend(
        handles=legend_handles(),
        loc="lower right",
        ncol=1,
        frameon=True,
        framealpha=0.95,
        title=r"curve labels: $T/T_0$",
        title_fontsize=plt.rcParams["legend.fontsize"],
    )
    fig.tight_layout()

    output_dir.mkdir(exist_ok=True)
    file_stem = f"fig_D_T_profile_D_over_D0_{d_hat:.2f}".replace(".", "p")
    fig.savefig(output_dir / f"{file_stem}.pdf", bbox_inches="tight")
    fig.savefig(output_dir / f"{file_stem}.png", dpi=600, bbox_inches="tight")
    return fig, ax


def make_plot():
    figures = []
    for d_hat in D_HATS:
        figures.append(plot_profile_for_d(d_hat))
    return figures


if __name__ == "__main__":
    print(f"D0 = {D0:.6e} N m")
    print(f"T0 = {T0:.6f} N/m")
    make_plot()
    if "agg" not in plt.get_backend().lower():
        plt.show()
