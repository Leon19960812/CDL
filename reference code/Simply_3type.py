import numpy as np
import matplotlib.pyplot as plt
import scienceplots
from scipy.optimize import brentq
from matplotlib.ticker import FormatStrFormatter
from matplotlib.lines import Line2D
plt.style.use(['science', 'notebook', 'grid'])

# 这个脚本用来绘制简支第三种情况下挠度曲线（liquid prevails）
# 对比present model 和 classical model
# 2026.1.15 by Leon

# 定义参数
l = 1
h = 0.02
V = h * l
x = np.linspace(-l / 2, l / 2, 100)

# lambda > pi 的情形
lambda_vals = [np.pi+0.001, 3.5, 4.0, 4.5, 5.0]


# -------------------------------
# present model
# -------------------------------
def determinant(beta, lam):
    """系数矩阵行列式，求解 beta = lambda * a / l"""
    d1 = (beta - lam / 2) * np.sinh(beta) - np.cosh(beta)
    d2 = (beta - lam / 2) * np.sin(beta) + np.cos(beta)
    return d1 * np.cos(beta) - d2 * np.cosh(beta)


def solve_a(lam):
    """给定 lambda 求 a"""

    f = lambda b: determinant(b, lam)
    low, high = 1e-6, lam / 2 - 1e-4

    try:
        beta = brentq(f, low, high)
    except ValueError:
        # 如果默认区间没有符号变化，扫描寻找括区
        scan = np.linspace(low, high, 400)
        vals = determinant(scan, lam)
        sign = np.sign(vals)
        idx = np.where(np.diff(sign) != 0)[0]
        if len(idx) == 0:
            raise
        beta = brentq(f, scan[idx[0]], scan[idx[0] + 1])

    return beta * l / lam


def zeta_present(x, lam, a):
    """分段挠度函数"""
    beta = lam * a / l
    F = np.sin(beta) * np.cosh(beta) - np.cos(beta) * np.sinh(beta)
    G = (np.tan(beta) + np.tanh(beta)) / (np.tan(beta) - np.tanh(beta))
    delta = l / 2 - a

    const_inner = h * lam**2 / (2 * l) * delta * G + h * lam**4 / (6 * l**3) * delta**3
    inner = (
        -h * lam * np.cos(beta) / (2 * F) * np.cosh(lam * x / l)
        + h * lam * np.cosh(beta) / (2 * F) * np.cos(lam * x / l)
        + const_inner
    )

    s = l / 2 - np.abs(x)  # 同时满足两端简支边界条件
    outer = (
        (V * lam**2 / (2 * l**2) * G + V * lam**4 / (4 * l**4) * delta**2) * s
        - V * lam**4 / (12 * l**4) * s**3
    )

    is_inner = np.abs(x) <= a
    return np.where(is_inner, inner, outer)


# -------------------------------
# classical model
# -------------------------------
def zeta_classical(x, lam):
    return (h * lam**4 / l**2) * (x**4 / (24 * l**2) - x**2 / 16 + 5 * l**2 / 384)


# -------------------------------
# 画图
# -------------------------------
fig, ax = plt.subplots(figsize=(7,5))

color_present = 'tab:orange'
color_classical = 'tab:red'

gap = 2  # 用于在中间留空显示断点

for lam in lambda_vals:
    # 判断线形
    if np.isclose(lam, np.pi + 0.001, atol=1e-12):
        ls = '-'
        lw = 2.0
    else:
        ls = '--'
        lw = 1.5

    # 计算 a
    a_val = solve_a(lam)

    idx = len(x) // 2

    # ---------- present model ----------
    x0 = 0.0
    idx_p = np.argmin(np.abs(x - x0))

    z = zeta_present(x, lam, a_val)/h

    z_plot = z.copy()
    z_plot[idx_p - gap:idx_p + gap + 1] = np.nan  # 在中间留空以显示断点
    plt.plot(
        x,
        z_plot,
        linestyle=ls,
        linewidth=lw,
        color=color_present
    )

    plt.text(
        x[idx_p],
        z[idx_p],
        rf'${lam:.3g}$',
        fontsize=14,
        color=color_present,
        ha='center',
        va='center'
    )

    # ---------- classical model ----------
    x1 = 0.1
    idx_c = np.argmin(np.abs(x - x1))
    z_c = zeta_classical(x, lam)/h

    zc_plot = z_c.copy()
    zc_plot[idx_c - gap:idx_c + gap + 1] = np.nan  # 在中间留空以显示断点
    plt.plot(
        x,
        zc_plot,
        linestyle=ls,
        linewidth=lw,
        color=color_classical
    )

    plt.text(
        x[idx_c],
        z_c[idx_c],
        rf'${lam:.3g}$',
        fontsize=14,
        color=color_classical,
        ha='center',
        va='center'
    )

    # ---------- marker points at x = +/-a ----------
    z_pp = zeta_present(np.array([a_val]), lam, a_val)[0] / h
    z_pn = zeta_present(np.array([-a_val]), lam, a_val)[0] / h

    plt.scatter(
        [a_val, -a_val],
        [z_pp, z_pn],
        color=color_present,
        s=30,
        zorder=6
    )


plt.xlabel(r'$x/l$')
plt.ylabel(r'$\zeta/h$')
plt.gca().invert_yaxis()

legend_handles = [
    Line2D([0], [0], color=color_present, lw=2, linestyle='-', label='Present model'),
    Line2D([0], [0], color=color_classical, lw=2, linestyle='-', label='Classical model'),
    Line2D([0], [0], color='black', lw=2, linestyle='-', label=r'$\lambda=\pi$'),
    Line2D([0], [0], color='black', lw=2, linestyle='--', label=r'$\lambda>\pi$')
]
ax.legend(handles=legend_handles, loc='best', fontsize=11, frameon=True)
ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
plt.show()
