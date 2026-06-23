import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from matplotlib.lines import Line2D

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


COLOR_PLATE_SIMPLY = "tab:orange"
COLOR_MEMBRANE = "0.30"
COLOR_EQ = "black"
COLOR_SHADE = "0.92"
COLOR_EQUIPOISE = "tab:red"
COLOR_PURE_BENDING = "tab:blue"
D_RATIOS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
H_OVER_L = 0.01
THETA_LIMIT = np.tan(np.deg2rad(10.0))

RHO = 1000.0
G = 9.8
L = 0.03
KAPPA0 = 2.0 * np.pi**4
D0 = RHO * G * L**4 / KAPPA0
TAU_EQ = (KAPPA0 - np.pi**4) / np.pi**2
T_EQ = D0 * TAU_EQ / L**2


def coth(x):
    return 1.0 / np.tanh(x)


def lambdas(t_hat, d_ratio):
    tau = t_hat * TAU_EQ / d_ratio
    kappa = KAPPA0 / d_ratio
    root = np.sqrt(tau**2 + 4.0 * kappa)
    lambda_1 = np.sqrt(0.5 * (root + tau))
    lambda_2 = np.sqrt(0.5 * (root - tau))
    return lambda_1, lambda_2, tau


def find_root_by_scan(func, q_min, q_max, n=1200):
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

    raise RuntimeError("No physical root found in the liquid-prevailing branch.")


def solve_q_liquid(t_hat, d_ratio):
    lambda_1, lambda_2, tau = lambdas(t_hat, d_ratio)
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


def h2_over_h_liquid(t_hat, d_ratio):
    q = solve_q_liquid(t_hat, d_ratio)
    lambda_1, lambda_2, tau = lambdas(t_hat, d_ratio)
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

    return (
        lambda_1
        * lambda_2
        / geom_factor
        * (
            -beta_hat * moment_hat
            + (lambda_1**2 + lambda_2**2) * np.cosh(lambda_1 * q) / m**2
        )
    )


def pure_bending_endpoint(d_ratio):
    kappa = KAPPA0 / d_ratio
    lambda_p = kappa ** 0.25

    def pure_bending_equation(q):
        beta = lambda_p * q
        return (
            (beta - lambda_p / 2.0)
            * (np.sinh(beta) * np.cos(beta) - np.sin(beta) * np.cosh(beta))
            - 2.0 * np.cosh(beta) * np.cos(beta)
        )

    q0 = find_root_by_scan(
        pure_bending_equation,
        1e-8,
        np.pi / (2.0 * lambda_p) - 1e-8,
    )
    beta0 = lambda_p * q0
    G0 = (np.tan(beta0) + np.tanh(beta0)) / (
        np.tan(beta0) - np.tanh(beta0)
    )
    h2_h0 = (
        lambda_p**2 / 2.0 * (0.5 - q0) * G0
        + lambda_p**4 / 6.0 * (0.5 - q0) ** 3
    )
    return q0, h2_h0


def pure_membrane_lambda(t_hat):
    return np.pi * np.sqrt(2.0 / t_hat)


def h2_over_h_pure_membrane(t_hat):
    lambda_m = pure_membrane_lambda(t_hat)
    return lambda_m * (lambda_m - np.pi) / 4.0


def t_hat_equipoise(d_ratio):
    return 2.0 - d_ratio


def format_ratio_label(d_ratio):
    return f"{d_ratio:g}"


def theta_max_liquid(t_hat, d_ratio, q):
    lambda_1, lambda_2, tau = lambdas(t_hat, d_ratio)
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
    return H_OVER_L * abs(lambda_1 * lambda_2 / geom_factor * (moment_hat - edge_term))


def theta_max_pure_membrane(t_hat):
    theta = np.full_like(t_hat, np.inf, dtype=float)
    positive = t_hat > 0.0
    theta[positive] = H_OVER_L * np.pi**2 / t_hat[positive]
    return theta


def admissible_mask(t_hat, d_ratio):
    if np.isclose(d_ratio, 0.0):
        theta = theta_max_pure_membrane(t_hat)
    else:
        theta = np.full_like(t_hat, np.nan, dtype=float)
        t_eq = t_hat_equipoise(d_ratio)
        internal = (t_hat > 0.0) & (t_hat < t_eq - 1e-6)
        theta[internal] = np.array(
            [
                theta_max_liquid(t_value, d_ratio, solve_q_liquid(t_value, d_ratio))
                for t_value in t_hat[internal]
            ]
        )
        theta[t_hat >= t_eq - 1e-6] = H_OVER_L * np.pi**2 / 2.0
        if np.any(internal):
            theta[t_hat <= 0.0] = theta[internal][0]
    return theta <= THETA_LIMIT


def labeled_curve(
    ax,
    x_vals,
    y_vals,
    label,
    label_x,
    color=COLOR_PLATE_SIMPLY,
    linewidth=1.8,
    admissible=None,
):
    gap = 6
    idx = np.argmin(np.abs(x_vals - label_x))
    y_plot = y_vals.copy()
    start = max(0, idx - gap)
    stop = min(len(y_plot), idx + gap + 1)
    y_plot[start:stop] = np.nan

    if admissible is None:
        admissible = np.ones_like(x_vals, dtype=bool)

    y_admissible = np.where(admissible, y_plot, np.nan)
    y_inadmissible = np.where(~admissible, y_plot, np.nan)

    ax.plot(x_vals, y_admissible, color=color, linestyle="-", lw=linewidth, zorder=3)
    ax.plot(
        x_vals,
        y_inadmissible,
        color=color,
        linestyle="--",
        lw=linewidth,
        zorder=3,
    )
    ax.text(
        x_vals[idx],
        y_vals[idx],
        label,
        color=color,
        fontsize=10,
        ha="center",
        va="center",
        zorder=5,
    )


def liquid_branch_curve(d_ratio):
    if np.isclose(d_ratio, 0.0):
        t_hat_plot = np.linspace(0.5, 2.0, 240)
        return t_hat_plot, h2_over_h_pure_membrane(t_hat_plot)

    t_eq = t_hat_equipoise(d_ratio)
    t_hat_internal = np.linspace(1e-4, t_eq - 1e-3, 200)
    h2_internal = np.array(
        [h2_over_h_liquid(t_hat, d_ratio) for t_hat in t_hat_internal]
    )
    _, h2_h0 = pure_bending_endpoint(d_ratio)
    return np.r_[0.0, t_hat_internal, t_eq], np.r_[h2_h0, h2_internal, 0.0]


def make_plot():
    fig, ax = plt.subplots(figsize=(7, 5))

    equipoise_points = []
    pure_bending_points = []
    y_max = 0.0

    for d_ratio in D_RATIOS:
        t_hat_plot, h2_plot = liquid_branch_curve(d_ratio)
        t_eq = t_hat_equipoise(d_ratio)
        label_x = 0.6 * t_eq if d_ratio < 1.0 else 0.55 * t_eq
        mask = admissible_mask(t_hat_plot, d_ratio)

        labeled_curve(
            ax,
            t_hat_plot,
            h2_plot,
            format_ratio_label(d_ratio),
            label_x=label_x,
            color=COLOR_MEMBRANE if np.isclose(d_ratio, 0.0) else COLOR_PLATE_SIMPLY,
            linewidth=2.0 if np.isclose(d_ratio, 0.0) else 1.75,
            admissible=mask,
        )
        y_max = max(y_max, np.nanmax(h2_plot))
        equipoise_points.append((t_eq, 0.0))
        if d_ratio > 0.0:
            pure_bending_points.append((0.0, h2_plot[0]))

    if pure_bending_points:
        x_bending, y_bending = np.array(pure_bending_points).T
        ax.plot(
            x_bending,
            y_bending,
            "o",
            markerfacecolor=COLOR_PURE_BENDING,
            markeredgecolor=COLOR_PURE_BENDING,
            ms=5.5,
            zorder=6,
        )

    x_eq, y_eq = np.array(equipoise_points).T
    ax.plot(
        x_eq,
        y_eq,
        "o",
        markerfacecolor=COLOR_EQUIPOISE,
        markeredgecolor=COLOR_EQUIPOISE,
        ms=6,
        zorder=6,
    )

    ax.text(
        0.08,
        0.95,
        r"labels: $D/D_0$, $h/l=0.01$",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
    )
    ax.set_xlabel(r"$T/T_{\mathrm{eq}}$")
    ax.set_ylabel(r"$h_2/h$")
    ax.set_xlim(-0.03, 2.1)
    ax.set_ylim(bottom=-0.05, top=h2_over_h_pure_membrane(0.5) * 1.08)
    ax.legend(
        handles=[
            Line2D(
                [0],
                [0],
                color=COLOR_PLATE_SIMPLY,
                lw=1.8,
                label="Liquid-prevailing branches",
            ),
            Line2D(
                [0],
                [0],
                color=COLOR_MEMBRANE,
                linestyle="-",
                lw=2.0,
                label=r"$D=0$ pure membrane ref.",
            ),
            Line2D(
                [0],
                [0],
                color="black",
                linestyle="--",
                lw=1.8,
                label=r"Outside admissible domain",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                markerfacecolor=COLOR_EQUIPOISE,
                markeredgecolor=COLOR_EQUIPOISE,
                linestyle="None",
                label="Equipoise",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                markerfacecolor=COLOR_PURE_BENDING,
                markeredgecolor=COLOR_PURE_BENDING,
                linestyle="None",
                label="Pure-bending limit",
            ),
        ],
        loc="upper right",
        frameon=True,
    )
    fig.tight_layout()

    fig.savefig("fig_fixed_D_h2_h.pdf", bbox_inches="tight")
    fig.savefig("fig_fixed_D_h2_h.png", dpi=600, bbox_inches="tight")
    return fig, ax


if __name__ == "__main__":
    print(f"D0 = {D0:.6e} N m")
    print(f"T_eq = {T_EQ:.6f} N/m")
    make_plot()
    if "agg" not in plt.get_backend().lower():
        plt.show()
