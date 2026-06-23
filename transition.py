import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif']
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['axes.unicode_minus'] = False


# -----------------------------
# Length criterion figure
# -----------------------------

# normalized span
s = np.linspace(0, 3.5, 500)  # s = l / l_c

# contribution ratios
R_D = 1 / (1 + s**2)
R_T = s**2 / (1 + s**2)

# create figure
fig, ax = plt.subplots(figsize=(7.2, 4.8), dpi=300)

# plot curves
ax.plot(
    s, R_D,
    linewidth=2.5,
    label=r'$R_D=\dfrac{C_D}{C_D+C_T}=\dfrac{1}{1+(l/l_c)^2}$'
)

ax.plot(
    s, R_T,
    linewidth=2.5,
    linestyle='--',
    label=r'$R_T=\dfrac{C_T}{C_D+C_T}=\dfrac{(l/l_c)^2}{1+(l/l_c)^2}$'
)

# transition line
ax.axvline(
    x=1,
    color='black',
    linestyle=':',
    linewidth=1.8
)

# transition point
ax.plot(1, 0.5, 'ko', markersize=5)
ax.text(
    1.06, 0.52,
    r'$R_D=R_T=0.5$',
    fontsize=11
)

# label for critical length
ax.text(
    1.0, 1.035,
    r'$l=l_c$',
    ha='center',
    va='bottom',
    fontsize=12
)

# region annotations
ax.text(
    0.42, 0.16,
    r'$l/l_c<1$' + '\n' + 'bending-dominated',
    ha='center',
    va='center',
    fontsize=11
)

ax.text(
    1.65, 0.16,
    r'$l/l_c>1$' + '\n' + 'tension-dominated',
    ha='center',
    va='center',
    fontsize=11
)

# formula box
formula_text = (
    r'$\dfrac{C_D}{C_T}=\left(\dfrac{l_c}{l}\right)^2$'
    '\n'
    r'$l_c=\pi\sqrt{\dfrac{D}{T}}$'
)

ax.text(
    2.22, 0.78,
    formula_text,
    fontsize=13,
    bbox=dict(
        boxstyle='round,pad=0.35',
        facecolor='white',
        edgecolor='black',
        linewidth=1.0
    )
)

# axis labels
ax.set_xlabel(r'Normalized span, $l/l_c$', fontsize=13)
ax.set_ylabel('Contribution ratio', fontsize=13)

# limits and ticks
ax.set_xlim(0, 3.5)
ax.set_ylim(0, 1.08)

ax.set_xticks(np.arange(0, 3.6, 0.5))
ax.set_yticks(np.arange(0, 1.1, 0.2))

# legend
ax.legend(
    loc='center right',
    frameon=False,
    fontsize=10
)

# visual polish
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.tick_params(direction='in', length=4, width=1)
ax.grid(False)

plt.tight_layout()

# save figure
plt.savefig('length_criterion_bending_tension.png', dpi=600, bbox_inches='tight')
plt.savefig('length_criterion_bending_tension.pdf', bbox_inches='tight')

plt.show()