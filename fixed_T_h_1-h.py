import numpy as np
import matplotlib.pyplot as plt
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
T_RATIOS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
D_HAT_MAX = 4.0

RHO = 1000.0
G = 9.8
L = 0.03
KAPPA0 = 2.0 * np.pi**4
TAU0 = np.pi**2
D0 = RHO * G * L**4 / KAPPA0
T0 = TAU0 * D0 / L**2


def lambdas(d_hat, t_ratio):
    tau = t_ratio * TAU0 / d_hat
    kappa = KAPPA0 / d_hat
    root = np.sqrt(tau**2 + 4.0 * kappa)
    lambda_1 = np.sqrt(0.5 * (root + tau))
    lambda_2 = np.sqrt(0.5 * (root - tau))
    return lambda_1, lambda_2


def h1_over_h(d_hat, t_ratio):
    lambda_1, lambda_2 = lambdas(d_hat, t_ratio)
    force_factor = (
        lambda_2**2 / lambda_1 * np.tanh(lambda_1 / 2.0)
        + lambda_1**2 / lambda_2 * np.tan(lambda_2 / 2.0)
    )
    return (lambda_1**2 + lambda_2**2) / (2.0 * force_factor)


def pure_bending_lambda(d_hat):
    return (KAPPA0 / d_hat) ** 0.25


def h1_over_h_pure_bending(d_hat):
    lambda_p = pure_bending_lambda(d_hat)
    return lambda_p / (np.tan(lambda_p / 2.0) + np.tanh(lambda_p / 2.0))


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


def structure_branch_curve(t_ratio):
    if np.isclose(t_ratio, 0.0):
        d_hat_internal = np.linspace(2.001, D_HAT_MAX, 200)
        h1_internal = h1_over_h_pure_bending(d_hat_internal)
        return np.r_[2.0, d_hat_internal], np.r_[0.0, h1_internal]

    d_eq = d_hat_equipoise(t_ratio)
    d_hat_internal = np.linspace(d_eq + 1e-3, D_HAT_MAX, 200)
    h1_internal = h1_over_h(d_hat_internal, t_ratio)
    return np.r_[d_eq, d_hat_internal], np.r_[0.0, h1_internal]


def make_plot():
    fig, ax = plt.subplots(figsize=(7, 5))

    equipoise_points = []

    for t_ratio in T_RATIOS:
        d_hat_plot, h1_plot = structure_branch_curve(t_ratio)
        d_eq = d_hat_equipoise(t_ratio)
        label_x = d_eq + 0.45 * (D_HAT_MAX - d_eq)

        labeled_curve(
            ax,
            d_hat_plot,
            h1_plot,
            format_ratio_label(t_ratio),
            label_x=label_x,
            color=COLOR_PURE_BENDING if np.isclose(t_ratio, 0.0) else COLOR_PLATE_SIMPLY,
            linestyle="--" if np.isclose(t_ratio, 0.0) else "-",
            linewidth=2.0 if np.isclose(t_ratio, 0.0) else 1.75,
        )
        equipoise_points.append((d_eq, 0.0))

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
    ax.set_ylabel(r"$h_1/h$")
    ax.set_xlim(0.0, D_HAT_MAX)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(
        handles=[
            Line2D(
                [0],
                [0],
                color=COLOR_PLATE_SIMPLY,
                lw=1.8,
                label="Structure-prevailing branches",
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
        ],
        loc="upper right",
        frameon=True,
    )
    fig.tight_layout()

    fig.savefig("fig_fixed_T_h1_h.pdf", bbox_inches="tight")
    fig.savefig("fig_fixed_T_h1_h.png", dpi=600, bbox_inches="tight")
    return fig, ax


if __name__ == "__main__":
    print(f"D0 = {D0:.6e} N m")
    print(f"T0 = {T0:.6f} N/m")
    make_plot()
    if "agg" not in plt.get_backend().lower():
        plt.show()
