import numpy as np
import matplotlib.pyplot as plt
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


COLOR_EQ = "black"
COLORS_D = ["#4C78A8", "#59A14F", "#F28E2B", "#E15759", "#B07AA1"]
FIGSIZE_FIXED_KAPPA_SCAN_TAU = (5.6, 4.6)

RHO = 1000.0
G = 9.8
L = 0.03
KAPPA0 = 2.0 * np.pi**4
D0 = RHO * G * L**4 / KAPPA0
TAU0 = (KAPPA0 - np.pi**4) / np.pi**2
T0 = D0 * TAU0 / L**2

# Optical parameters for a first paraxial liquid-lens estimate.
REFRACTIVE_INDEX = 1.33
H_OVER_L = 0.01
THETA_LIMIT = np.tan(np.deg2rad(10.0))

# Plot a family of fixed-D curves on the common D0-based tension scale.
D_HATS = np.array([0.50, 0.75, 1.00, 1.25, 1.50])
T_HAT_MAX = 4.0


def coth(x):
    return 1.0 / np.tanh(x)


def kappa_for_d(d_hat):
    return KAPPA0 / d_hat


def transition_t_hat(d_hat):
    """Flush-fill tension normalized by the D0-based T0."""
    return 2.0 - d_hat


def lambdas(t_hat, d_hat=1.0):
    kappa = kappa_for_d(d_hat)
    tau = t_hat * TAU0 / d_hat
    root = np.sqrt(tau**2 + 4.0 * kappa)
    lambda_1 = np.sqrt(0.5 * (root + tau))
    lambda_2 = np.sqrt(0.5 * (root - tau))
    return lambda_1, lambda_2, tau


def find_root_by_scan(func, q_min, q_max, n=900):
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


def central_curvature_structure(t_hat, d_hat=1.0):
    lambda_1, lambda_2, _ = lambdas(t_hat, d_hat)
    h1_h = h1_over_h(t_hat, d_hat)
    return (
        h1_h
        * lambda_1**2
        * lambda_2**2
        / (lambda_1**2 + lambda_2**2)
        * (1.0 / np.cos(lambda_2 / 2.0) - 1.0 / np.cosh(lambda_1 / 2.0))
    )


def central_curvature_liquid(t_hat, d_hat=1.0):
    q = solve_q_liquid(t_hat, d_hat)
    lambda_1, lambda_2, _ = lambdas(t_hat, d_hat)
    geom_factor = (
        2.0 * lambda_2 * np.sinh(lambda_1 * q)
        - 2.0 * lambda_1 * np.tan(lambda_2 * q) * np.cosh(lambda_1 * q)
    )
    center_second_derivative = (
        lambda_1
        * lambda_2
        / geom_factor
        * (
            lambda_1**2
            + lambda_2**2
            * np.cosh(lambda_1 * q)
            / np.cos(lambda_2 * q)
        )
    )
    return abs(center_second_derivative)


def solve_q_pure_bending(d_hat=1.0):
    lambda_p = kappa_for_d(d_hat) ** 0.25

    def pure_bending_equation(q):
        beta = lambda_p * q
        return (
            (beta - lambda_p / 2.0)
            * (np.sinh(beta) * np.cos(beta) - np.sin(beta) * np.cosh(beta))
            - 2.0 * np.cosh(beta) * np.cos(beta)
        )

    return find_root_by_scan(
        pure_bending_equation,
        1e-8,
        np.pi / (2.0 * lambda_p) - 1e-8,
    )


def central_curvature_pure_bending(d_hat=1.0):
    lambda_p = kappa_for_d(d_hat) ** 0.25
    q = solve_q_pure_bending(d_hat)
    beta = lambda_p * q
    force_factor = np.sin(beta) * np.cosh(beta) - np.cos(beta) * np.sinh(beta)
    return lambda_p**3 * (np.cos(beta) + np.cosh(beta)) / (2.0 * force_factor)


def theta_max_pure_bending(d_hat=1.0):
    lambda_p = kappa_for_d(d_hat) ** 0.25
    q = solve_q_pure_bending(d_hat)
    beta = lambda_p * q
    boundary = 0.5 - q
    level_factor = (np.tan(beta) + np.tanh(beta)) / (
        np.tan(beta) - np.tanh(beta)
    )
    edge_slope = lambda_p**2 / 2.0 * level_factor + lambda_p**4 / 4.0 * boundary**2
    return H_OVER_L * abs(edge_slope)


def theta_max_liquid(t_hat, d_hat=1.0):
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
    edge_term = (
        (lambda_1**2 + lambda_2**2)
        / m
        * np.cosh(lambda_1 * q)
        / np.sinh(m * beta_hat)
    )
    edge_slope = lambda_1 * lambda_2 / geom_factor * (moment_hat - edge_term)
    return H_OVER_L * abs(edge_slope)


def theta_max_flush():
    return H_OVER_L * np.pi**2 / 2.0


def theta_max_structure(t_hat, d_hat=1.0):
    x = np.linspace(-0.5, 0.5, 1201)
    lambda_1, lambda_2, _ = lambdas(t_hat, d_hat)
    h1_h = h1_over_h(t_hat, d_hat)
    denominator = lambda_1**2 + lambda_2**2
    dz_dx_hat = h1_h * (
        lambda_2**2
        * lambda_1
        / denominator
        * np.sinh(lambda_1 * x)
        / np.cosh(lambda_1 / 2.0)
        - lambda_1**2
        * lambda_2
        / denominator
        * np.sin(lambda_2 * x)
        / np.cos(lambda_2 / 2.0)
    )
    return H_OVER_L * float(np.nanmax(np.abs(dz_dx_hat)))


def theta_max(t_hat, d_hat=1.0):
    t_hat_flush = transition_t_hat(d_hat)

    if np.isclose(t_hat, 0.0):
        if t_hat_flush > 0.0:
            return theta_max_pure_bending(d_hat)
        return theta_max_structure(t_hat, d_hat)

    if t_hat_flush >= 0.0 and t_hat < t_hat_flush:
        return theta_max_liquid(t_hat, d_hat)

    if t_hat_flush >= 0.0 and np.isclose(t_hat, t_hat_flush, atol=1e-9):
        return theta_max_flush()

    return theta_max_structure(t_hat, d_hat)


def central_curvature_hat(t_hat, d_hat=1.0):
    t_hat_eq = transition_t_hat(d_hat)

    if t_hat_eq >= 0.0 and np.isclose(t_hat, t_hat_eq, atol=1e-9):
        return np.pi**3 / 2.0

    if np.isclose(t_hat, 0.0):
        if t_hat_eq > 0.0:
            return central_curvature_pure_bending(d_hat)
        return central_curvature_structure(t_hat, d_hat)

    if t_hat_eq > 0.0 and t_hat < t_hat_eq:
        return central_curvature_liquid(t_hat, d_hat)

    return central_curvature_structure(t_hat, d_hat)


def focal_length_over_l(curvature_hat):
    return 1.0 / ((REFRACTIVE_INDEX - 1.0) * H_OVER_L * curvature_hat)


def compute_curve(d_hat, t_hat_max=T_HAT_MAX):
    t_hat_eq = transition_t_hat(d_hat)
    t_parts = []
    curvature_parts = []
    admissible_parts = []

    def append(values):
        values = np.asarray(values, dtype=float)
        t_parts.append(values)
        curvature_parts.append(
            np.array([central_curvature_hat(t_hat, d_hat) for t_hat in values])
        )
        admissible_parts.append(
            np.array([theta_max(t_hat, d_hat) <= THETA_LIMIT for t_hat in values])
        )

    if t_hat_eq > 0.0:
        append([0.0])

        if t_hat_eq > 2e-4:
            n_liquid = max(300, int(700 * t_hat_eq))
            append(np.linspace(1e-4, t_hat_eq - 1e-4, n_liquid))

        append([t_hat_eq])

        if t_hat_eq < t_hat_max:
            n_structure = max(500, int(350 * (t_hat_max - t_hat_eq)))
            append(np.linspace(t_hat_eq + 1e-4, t_hat_max, n_structure))
    else:
        append(np.linspace(0.0, t_hat_max, 1200))

    t_hat_plot = np.concatenate(t_parts)
    curvature_plot = np.concatenate(curvature_parts)
    admissible_plot = np.concatenate(admissible_parts)
    return t_hat_plot, focal_length_over_l(curvature_plot), admissible_plot


def make_plot():
    curves = []
    for d_hat in D_HATS:
        t_hat_plot, focal_plot, admissible_plot = compute_curve(d_hat)
        curves.append((d_hat, t_hat_plot, focal_plot, admissible_plot))

    all_focal = np.concatenate([curve[2] for curve in curves])

    fig, ax = plt.subplots(figsize=FIGSIZE_FIXED_KAPPA_SCAN_TAU)

    for i, (d_hat, t_hat_plot, focal_plot, admissible_plot) in enumerate(curves):
        color = COLORS_D[i % len(COLORS_D)]
        focal_admissible = np.where(admissible_plot, focal_plot, np.nan)
        focal_inadmissible = np.where(~admissible_plot, focal_plot, np.nan)
        ax.plot(
            t_hat_plot,
            focal_admissible,
            color=color,
            lw=2.0,
            label=rf"${d_hat:.2f}$",
            zorder=3,
        )
        ax.plot(
            t_hat_plot,
            focal_inadmissible,
            color=color,
            lw=2.0,
            ls="--",
            zorder=3,
        )

    legend_handles, legend_labels = ax.get_legend_handles_labels()
    legend_handles.append(
        plt.Line2D([0], [0], color="black", lw=2.0, ls="--")
    )
    legend_labels.append("Inadmissible")

    ax.set_xlabel(r"$T/T_0$")
    ax.set_ylabel(r"$f/l$")
    ax.set_xlim(0.0, T_HAT_MAX)
    ax.set_ylim(bottom=0.0, top=max(all_focal) * 1.08)
    ax.legend(
        legend_handles,
        legend_labels,
        loc="lower right",
        frameon=True,
        title=r"$D/D_0$",
        fontsize=9,
        title_fontsize=10,
        handlelength=1.3,
        borderpad=0.35,
        labelspacing=0.2,
    )
    fig.tight_layout()

    fig.savefig("fig_focal_length_D_family.pdf", bbox_inches="tight")
    fig.savefig("fig_focal_length_D_family.png", dpi=600, bbox_inches="tight")

    print(f"f/l range = {np.nanmin(all_focal):.6f} to {np.nanmax(all_focal):.6f}")
    print(
        "f range = "
        f"{np.nanmin(all_focal) * L * 1000.0:.3f} to "
        f"{np.nanmax(all_focal) * L * 1000.0:.3f} mm"
    )
    return fig, ax


if __name__ == "__main__":
    print(f"D0 = {D0:.6e} N m")
    print(f"T0 = {T0:.6f} N/m")
    make_plot()
    if "agg" not in plt.get_backend().lower():
        plt.show()
