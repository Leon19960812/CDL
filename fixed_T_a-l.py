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
COLOR_PURE_BENDING = "0.30"
COLOR_EQ = "black"
COLOR_SHADE = "0.92"
COLOR_EQUIPOISE = "tab:red"
COLOR_PURE_MEMBRANE = "tab:blue"
T_RATIOS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]

RHO = 1000.0
G = 9.8
L = 0.03
KAPPA0 = 2.0 * np.pi**4
TAU0 = np.pi**2
D0 = RHO * G * L**4 / KAPPA0
T0 = TAU0 * D0 / L**2


def coth(x):
    return 1.0 / np.tanh(x)


def lambdas(d_hat, t_ratio):
    tau = t_ratio * TAU0 / d_hat
    kappa = KAPPA0 / d_hat
    root = np.sqrt(tau**2 + 4.0 * kappa)
    lambda_1 = np.sqrt(0.5 * (root + tau))
    lambda_2 = np.sqrt(0.5 * (root - tau))
    return lambda_1, lambda_2, tau, kappa


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


def solve_q_liquid(d_hat, t_ratio):
    lambda_1, lambda_2, tau, _ = lambdas(d_hat, t_ratio)
    m = np.sqrt(tau)

    def free_boundary(q):
        return (
            lambda_1**3 * np.tanh(lambda_1 * q)
            - lambda_2**3 * np.tan(lambda_2 * q)
            + m * (lambda_1**2 + lambda_2**2) * coth(m * (0.5 - q))
        )

    q_min = 1e-8
    q_max = min(0.5 - 1e-8, np.pi / (2.0 * lambda_2) - 1e-8)
    return find_root_by_scan(free_boundary, q_min, q_max)


def pure_membrane_q0(t_ratio):
    lambda_m = np.sqrt(RHO * G * L**2 / (t_ratio * T0))
    return np.pi / (2.0 * lambda_m)


def pure_bending_lambda(d_hat):
    return (KAPPA0 / d_hat) ** 0.25


def solve_q_pure_bending(d_hat):
    lambda_p = pure_bending_lambda(d_hat)

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


def d_hat_equipoise(t_ratio):
    return 2.0 - t_ratio


def format_ratio_label(t_ratio):
    return f"{t_ratio:g}"


def labeled_curve(
    ax,
    x_vals,
    y_vals,
    label,
    label_x,
    color=COLOR_PLATE_SIMPLY,
    linestyle="-",
    linewidth=1.8,
):
    gap = 6
    idx = np.argmin(np.abs(x_vals - label_x))
    y_plot = y_vals.copy()
    start = max(0, idx - gap)
    stop = min(len(y_plot), idx + gap + 1)
    y_plot[start:stop] = np.nan

    ax.plot(
        x_vals,
        y_plot,
        color=color,
        linestyle=linestyle,
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


def liquid_branch_curve(t_ratio):
    if np.isclose(t_ratio, 0.0):
        d_hat_internal = np.linspace(0.5, 1.999, 240)
        q_internal = np.array([solve_q_pure_bending(d) for d in d_hat_internal])
        return np.r_[d_hat_internal, 2.0], np.r_[q_internal, 0.5]

    d_eq = d_hat_equipoise(t_ratio)
    d_hat_internal = np.linspace(0.01, d_eq - 1e-3, 200)
    q_internal = np.array(
        [solve_q_liquid(d_hat, t_ratio) for d_hat in d_hat_internal]
    )
    q0 = pure_membrane_q0(t_ratio)
    return np.r_[0.0, d_hat_internal, d_eq], np.r_[q0, q_internal, 0.5]


def make_plot():
    fig, ax = plt.subplots(figsize=(7, 5))

    equipoise_points = []
    pure_membrane_points = []

    for t_ratio in T_RATIOS:
        d_hat_plot, q_plot = liquid_branch_curve(t_ratio)
        d_eq = d_hat_equipoise(t_ratio)
        label_x = 0.6 * d_eq if t_ratio > 0.0 else 1.25

        labeled_curve(
            ax,
            d_hat_plot,
            q_plot,
            format_ratio_label(t_ratio),
            label_x=label_x,
            color=COLOR_PURE_BENDING if np.isclose(t_ratio, 0.0) else COLOR_PLATE_SIMPLY,
            linestyle="--" if np.isclose(t_ratio, 0.0) else "-",
            linewidth=2.0 if np.isclose(t_ratio, 0.0) else 1.75,
        )
        equipoise_points.append((d_eq, 0.5))
        if t_ratio > 0.0:
            pure_membrane_points.append((0.0, q_plot[0]))

    if pure_membrane_points:
        x_membrane, y_membrane = np.array(pure_membrane_points).T
        ax.plot(
            x_membrane,
            y_membrane,
            "o",
            markerfacecolor=COLOR_PURE_MEMBRANE,
            markeredgecolor=COLOR_PURE_MEMBRANE,
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
        0.03,
        0.95,
        r"labels: $T/T_0$",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
    )
    ax.set_xlabel(r"$D/D_0$")
    ax.set_ylabel(r"$a/l$")
    ax.set_xlim(-0.03, 2.1)
    ax.set_ylim(0.0, 0.525)
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
                color=COLOR_PURE_BENDING,
                linestyle="--",
                lw=2.0,
                label=r"$T=0$ pure bending ref.",
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
                markerfacecolor=COLOR_PURE_MEMBRANE,
                markeredgecolor=COLOR_PURE_MEMBRANE,
                linestyle="None",
                label="Pure-membrane limit",
            ),
        ],
        loc="lower right",
        frameon=True,
    )
    fig.tight_layout()

    fig.savefig("fig_fixed_T_a_l.pdf", bbox_inches="tight")
    fig.savefig("fig_fixed_T_a_l.png", dpi=600, bbox_inches="tight")
    return fig, ax


if __name__ == "__main__":
    print(f"D0 = {D0:.6e} N m")
    print(f"T0 = {T0:.6f} N/m")
    make_plot()
    if "agg" not in plt.get_backend().lower():
        plt.show()
