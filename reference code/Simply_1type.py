import numpy as np
import matplotlib.pyplot as plt
import scienceplots
from scipy.optimize import brentq
from matplotlib.ticker import FormatStrFormatter
plt.style.use(['science', 'notebook', 'grid'])

# 这个脚本用来绘制简支第一种情况下在不同lambda下的挠度曲线
# 对比present model 和 classical model
# 2026.1.14 by Leon

# 定义参数
l = 1
h = 0.02
V = h * l
x = np.linspace(-1/2, 1/2, 100)

lambda_vals = [1.0, 1.5, 2.0, 2.5, np.pi]


# -------------------------------
# present model
# ------------------------------
def h_1(lam):
    return (V*lam)/(l*(np.tan(lam/2)+np.tanh(lam/2)))

def A(lam):
    h_1_val = h_1(lam)
    return h_1_val/(2*np.cosh(lam/2))

def C(lam):
    h_1_val = h_1(lam)
    return h_1_val/(2*np.cos(lam/2))

def zeta(x, lam):
    h_1_val = h_1(lam)
    A_val = A(lam)
    C_val = C(lam)
    return A_val * np.cosh(lam*x/l) + C_val * np.cos(lam*x/l) - h_1_val


# -------------------------------
# classical model
# ------------------------------
def zeta_classical(x, lam):
    return (h*lam**4/l**2)*(x**4/(24*l**2) - x**2/16 + 5*l**2/384)


# -------------------------------
# 画图
# ------------------------------
fig, ax = plt.subplots(figsize=(7,5))

color_present = 'tab:orange'
color_classical = 'tab:red'

gap = 2 # 用于在中间留空显示断点

for lam in lambda_vals:
    # 判断线形
    if np.isclose(lam, np.pi):
        ls = '-'
        lw = 2.0
    else:
        ls = '--'
        lw = 1.5
    
    idx = len(x)//2
# ----------present model---------
    x0 = 0.0
    idx_p = np.argmin(np.abs(x - x0))

    x_offset = 0.05
    z = zeta(x, lam)/h
    
    z_plot = z.copy()
    z_plot[idx_p-gap:idx_p+gap+1] = np.nan  # 在中间留空以显示断点
    plt.plot(
        x, z_plot,
        linestyle=ls,
        linewidth=lw,
        color=color_present
    )

    
    plt.text(
        x[idx_p], z[idx_p],
        rf'${lam:.3g}$',
        fontsize=14,
        color=color_present,
        ha='center',
        va='center'
    )

# ---------classical model--------
    x1 = 0.1
    idx_c = np.argmin(np.abs(x - x1))
    z_c = zeta_classical(x, lam)/h
    
    zc_plot = z_c.copy()
    zc_plot[idx_c-gap:idx_c+gap+1] = np.nan  # 在中间留空以显示断点
    plt.plot(
        x, zc_plot,
        linestyle=ls,
        linewidth=lw,
        color=color_classical
    )
    
    plt.text(
        x[idx_c], z_c[idx_c],
        rf'${lam:.3g}$',
        fontsize=14,
        color=color_classical,
        ha='center',
        va='center'
    )



plt.xlabel(r'$x/l$')
plt.ylabel(r'$\zeta/h$')
plt.gca().invert_yaxis() 
plt.legend()

ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
plt.show()


