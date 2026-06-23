# Codex task: Plot tension-tunable CDL response at fixed bending stiffness \(D\)

## Goal

Generate a publication-quality figure for the unified bending--tension CDL model.

The physical idea is:

> Fix the bending stiffness \(D=D_0\), plate span \(l\), liquid density \(\rho\), and gravity \(g\). Then tune only the imposed pre-tension \(T\). The system should move continuously from the liquid-prevailing regime to the structure-prevailing regime through the equipoise state.

The target figure should reproduce the style of the previous CDL papers: curves for

\[
\frac{h_1}{h}, \qquad \frac{h_2}{h}, \qquad \frac{2a}{l},
\]

but now plotted against the normalized tension

\[
\hat{T}=\frac{T}{T_{\mathrm{eq}}}.
\]

Use Python with `numpy`, `scipy`, and `matplotlib`.

---

## Recommended parameter choice

Use a theoretically clean and physically reasonable fixed bending stiffness by prescribing

\[
\kappa_0 = 2\pi^4,
\]

where

\[
\kappa_0 = \frac{\rho g l^4}{D_0}.
\]

Thus

\[
D_0 = \frac{\rho g l^4}{2\pi^4}.
\]

Use the following dimensional values:

\[
\rho = 1000~\mathrm{kg/m^3}, \qquad
 g = 9.8~\mathrm{m/s^2}, \qquad
 l = 0.03~\mathrm{m}.
\]

Then

\[
D_0 \approx 4.07\times 10^{-5}~\mathrm{N\,m}.
\]

This value is in the order of magnitude of sub-millimeter PDMS membranes used in tunable liquid lens/prism systems.

The corresponding equipoise tension is

\[
T_{\mathrm{eq}}
=
\frac{D_0}{l^2}
\left(
\frac{\kappa_0-\pi^4}{\pi^2}
\right).
\]

For \(\kappa_0=2\pi^4\), this simplifies to

\[
T_{\mathrm{eq}}
=
\frac{\pi^2D_0}{l^2}.
\]

Numerically,

\[
T_{\mathrm{eq}}\approx 0.446~\mathrm{N/m}.
\]

---

## Dimensionless parameters

For each imposed tension \(T\), define

\[
\tau = \frac{T l^2}{D_0}.
\]

Since \(D=D_0\) is fixed,

\[
\kappa = \kappa_0 = 2\pi^4
\]

is constant.

Define

\[
\lambda_1^2
=
\frac{1}{2}
\left[
\sqrt{\tau^2+4\kappa_0}
+
\tau
\right],
\]

\[
\lambda_2^2
=
\frac{1}{2}
\left[
\sqrt{\tau^2+4\kappa_0}
-
\tau
\right].
\]

Then

\[
\lambda_1 = \sqrt{\lambda_1^2}, \qquad
\lambda_2 = \sqrt{\lambda_2^2}.
\]

The liquid--structure equipoise condition is

\[
\lambda_2=\pi,
\]

which is equivalent to

\[
\kappa_0 = \pi^4+\pi^2\tau.
\]

Therefore

\[
\tau_{\mathrm{eq}}
=
\frac{\kappa_0-\pi^4}{\pi^2}.
\]

For the recommended choice \(\kappa_0=2\pi^4\),

\[
\tau_{\mathrm{eq}} = \pi^2.
\]

---

## Regime classification

Use the following classification:

\[
T<T_{\mathrm{eq}}
\quad \Leftrightarrow \quad
\tau<\tau_{\mathrm{eq}}
\quad \Leftrightarrow \quad
\lambda_2>\pi:
\quad \text{liquid-prevailing}.
\]

\[
T=T_{\mathrm{eq}}
\quad \Leftrightarrow \quad
\lambda_2=\pi:
\quad \text{equipoise}.
\]

\[
T>T_{\mathrm{eq}}
\quad \Leftrightarrow \quad
\tau>\tau_{\mathrm{eq}}
\quad \Leftrightarrow \quad
\lambda_2<\pi:
\quad \text{structure-prevailing}.
\]

---

## Important endpoint: \(T=0\) must recover the pure bending plate

At \(T=0\),

\[
\tau=0.
\]

Then

\[
\lambda_1^2=\lambda_2^2=\sqrt{\kappa_0},
\]

and therefore

\[
\lambda_1=\lambda_2=\kappa_0^{1/4}=\lambda_p.
\]

This is exactly the pure-bending plate limit.

For \(\kappa_0=2\pi^4\),

\[
\lambda_p = 2^{1/4}\pi > \pi,
\]

so the \(T=0\) endpoint lies in the liquid-prevailing regime.

In the plot, mark the \(T=0\) point as:

\[
T=0: \quad \text{pure bending plate}.
\]

Do **not** directly substitute \(T=0\) into the general liquid-prevailing formula because it contains \(\mu^2=T/D_0\) and may be numerically singular. Use the pure-bending limiting formulas given below.

---

# Figure design

Create one figure with three horizontal panels.

## Panel (a): structure-prevailing liquid level

Plot

\[
\frac{h_1}{h}
\]

against

\[
\hat{T}=\frac{T}{T_{\mathrm{eq}}}.
\]

Only plot the structure-prevailing branch:

\[
\hat{T}\geq 1.
\]

Expected trend:

\[
\frac{h_1}{h}=0
\quad \text{at} \quad
\hat{T}=1,
\]

and

\[
\frac{h_1}{h}\to 1
\quad \text{as} \quad
\hat{T}\to \infty.
\]

## Panel (b): liquid-prevailing deformed liquid level

Plot

\[
\frac{h_2}{h}
\]

against

\[
\hat{T}=\frac{T}{T_{\mathrm{eq}}}.
\]

Only plot the liquid-prevailing branch:

\[
0\leq \hat{T}\leq 1.
\]

Expected trend:

\[
\frac{h_2}{h}>0
\quad \text{for} \quad
\hat{T}<1,
\]

and

\[
\frac{h_2}{h}=0
\quad \text{at} \quad
\hat{T}=1.
\]

## Panel (c): normalized interaction length

Plot the normalized interaction length

\[
\frac{2a}{l}
\]

against

\[
\hat{T}=\frac{T}{T_{\mathrm{eq}}}.
\]

Only plot the liquid-prevailing branch:

\[
0\leq \hat{T}\leq 1.
\]

Here \(a\) is the interaction half-width, since

\[
\Sigma=\{-a<x<a\}.
\]

Therefore \(2a/l\) is the full normalized liquid--structure interaction length.

Expected trend:

\[
0<\frac{2a}{l}<1
\quad \text{for} \quad
\hat{T}<1,
\]

and

\[
\frac{2a}{l}=1
\quad \text{at} \quad
\hat{T}=1.
\]

---

## Add reference labels

In all three panels, add a vertical dashed line at

\[
\hat{T}=1.
\]

Label it as

\[
T=T_{\mathrm{eq}}.
\]

Also mark or annotate the left endpoint:

\[
\hat{T}=0:
\quad \text{pure bending plate}.
\]

---

# Structure-prevailing branch: \(h_1/h\)

For

\[
\tau>\tau_{\mathrm{eq}},
\]

use

\[
\mathcal{F}
=
\frac{\lambda_2^2}{\lambda_1}
\tanh\left(\frac{\lambda_1}{2}\right)
+
\frac{\lambda_1^2}{\lambda_2}
\tan\left(\frac{\lambda_2}{2}\right).
\]

Then

\[
\frac{h_1}{h}
=
\frac{\lambda_1^2+\lambda_2^2}{2\mathcal{F}}.
\]

At the equipoise point, set

\[
h_1/h = 0.
\]

---

# Liquid-prevailing branch: free boundary \(a/l\)

For

\[
0<\tau<\tau_{\mathrm{eq}},
\]

first solve for

\[
q=\frac{a}{l},
\qquad
0<q<\frac{1}{2}.
\]

Define

\[
m = \mu l = \sqrt{\tau}.
\]

The free-boundary equation is

\[
f(q)=0,
\]

where

\[
f(q)
=
\lambda_1^3\sinh(\lambda_1 q)
-
\lambda_2^3\cosh(\lambda_1 q)\tan(\lambda_2 q)
+
m(\lambda_1^2+\lambda_2^2)
\coth\left[m\left(\frac{1}{2}-q\right)\right]
\cosh(\lambda_1 q).
\]

The physical root is the one satisfying

\[
0<q<\frac{1}{2}.
\]

In practice, the root lies before the tangent singularity, so use the bracket

\[
q\in \left(10^{-8},\ \min\left(\frac{1}{2}-10^{-8},\ \frac{\pi}{2\lambda_2}-10^{-8}\right)\right).
\]

Use `scipy.optimize.brentq`.

At the equipoise point, set

\[
q=\frac{1}{2},
\qquad
\frac{2a}{l}=1.
\]

---

# Liquid-prevailing branch: \(h_2/h\)

After solving for

\[
q=\frac{a}{l},
\]

define

\[
\hat{\beta}=\frac{1}{2}-q.
\]

Define

\[
\mathcal{G}
=
2\lambda_2\sinh(\lambda_1 q)
-
2\lambda_1\tan(\lambda_2 q)\cosh(\lambda_1 q).
\]

Define the nondimensional quantity

\[
\widehat{\mathcal{M}}
=
\lambda_1\sinh(\lambda_1 q)
+
\lambda_2\cosh(\lambda_1 q)\tan(\lambda_2 q)
+
\frac{\lambda_1^2+\lambda_2^2}{m}
\cosh(\lambda_1 q)
\coth(m\hat{\beta}).
\]

Then use

\[
\frac{h_2}{h}
=
\frac{\lambda_1\lambda_2}{\mathcal{G}}
\left[
-\hat{\beta}\widehat{\mathcal{M}}
+
\frac{(\lambda_1^2+\lambda_2^2)\cosh(\lambda_1 q)}{m^2}
\right].
\]

This formula should be evaluated only for \(m>0\). Do not use it at exactly \(T=0\).

At the equipoise point, set

\[
h_2/h = 0.
\]

---

# Pure-bending endpoint at \(T=0\)

At \(T=0\),

\[
\lambda=\lambda_p=\kappa_0^{1/4}.
\]

For \(\kappa_0=2\pi^4\),

\[
\lambda=2^{1/4}\pi.
\]

Solve the pure-bending liquid-prevailing free-boundary equation for

\[
q_0=\frac{a_0}{l}.
\]

Let

\[
b=\lambda q_0.
\]

The equation is

\[
\left(b-\frac{\lambda}{2}\right)
\left[
\sinh b \cos b
-
\sin b \cosh b
\right]
-
2\cosh b\cos b
=0.
\]

Use the bracket

\[
q_0\in\left(10^{-8},\frac{\pi}{2\lambda}-10^{-8}\right).
\]

Then

\[
\frac{2a_0}{l}=2q_0.
\]

Define

\[
\mathcal{G}_0
=
\frac{\tan b+\tanh b}{\tan b-\tanh b}.
\]

The pure-bending endpoint for \(h_2/h\) is

\[
\frac{h_2}{h}
=
\frac{\lambda^2}{2}
\left(\frac{1}{2}-q_0\right)
\mathcal{G}_0
+
\frac{\lambda^4}{6}
\left(\frac{1}{2}-q_0\right)^3.
\]

Add this endpoint as a marker at

\[
\hat{T}=0.
\]

---

# Numerical sampling

Use

```python
n_liquid = 200
n_structure = 200
T_hat_liquid_internal = np.linspace(1e-4, 0.999, n_liquid)
T_hat_structure_internal = np.linspace(1.001, 4.0, n_structure)
```

For the liquid branch, avoid evaluating the general formula exactly at \(T=0\). Insert the pure-bending endpoint separately at \(\hat{T}=0\), and insert the equipoise endpoint at \(\hat{T}=1\).

For the structure branch, insert the equipoise endpoint \(h_1/h=0\) at \(\hat{T}=1\).

---

# Implementation notes

Implement a numerically stable `coth` function:

```python
def coth(x):
    return 1.0 / np.tanh(x)
```

For root solving, if `brentq` fails due to sign issues near tangent singularities, scan the bracket into many subintervals and choose the first sign-changing interval. The physical root should be continuous from the pure-bending endpoint at \(\hat{T}=0\) to the equipoise limit at \(\hat{T}=1\).

A robust approach:

```python
def find_root_by_scan(f, q_min, q_max, n=1000):
    qs = np.linspace(q_min, q_max, n)
    vals = np.array([f(q) for q in qs])
    finite = np.isfinite(vals)
    qs = qs[finite]
    vals = vals[finite]
    for i in range(len(qs)-1):
        if vals[i] == 0:
            return qs[i]
        if vals[i] * vals[i+1] < 0:
            return brentq(f, qs[i], qs[i+1])
    raise RuntimeError("No physical root found")
```

---

# Plotting style

Use a clean journal style.

Suggested settings:

```python
import matplotlib.pyplot as plt
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "mathtext.fontset": "dejavuserif",
})
```

Create three panels in one row:

```python
fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2), constrained_layout=True)
```

Use:

- solid line for the theoretical curve,
- dashed vertical line for \(T/T_{\mathrm{eq}}=1\),
- circular marker at \(T/T_{\mathrm{eq}}=0\) for the pure-bending endpoint,
- circular marker at \(T/T_{\mathrm{eq}}=1\) for the equipoise point.

Label axes:

Panel (a):

```python
axes[0].set_xlabel(r"$T/T_{\mathrm{eq}}$")
axes[0].set_ylabel(r"$h_1/h$")
```

Panel (b):

```python
axes[1].set_xlabel(r"$T/T_{\mathrm{eq}}$")
axes[1].set_ylabel(r"$h_2/h$")
```

Panel (c):

```python
axes[2].set_xlabel(r"$T/T_{\mathrm{eq}}$")
axes[2].set_ylabel(r"$2a/l$")
```

Use x-limits:

```python
axes[0].set_xlim(0, 4)
axes[1].set_xlim(0, 1.05)
axes[2].set_xlim(0, 1.05)
```

For panel (a), only the structure branch is physically active, but keep x-axis from 0 to 4 and lightly shade \(T/T_{\mathrm{eq}}<1\) as the liquid-prevailing region.

For panels (b) and (c), only the liquid branch is physically active.

Add text labels:

- `liquid-prevailing` for \(T/T_{\mathrm{eq}}<1\),
- `structure-prevailing` for \(T/T_{\mathrm{eq}}>1\),
- `equipoise` at \(T/T_{\mathrm{eq}}=1\),
- `pure bending plate` near \(T/T_{\mathrm{eq}}=0\).

Do not over-crowd the figure.

---

# Output files

Save the figure as:

```python
fig_fixed_D_tension_tuning.pdf
fig_fixed_D_tension_tuning.png
```

Use at least 600 dpi for PNG:

```python
fig.savefig("fig_fixed_D_tension_tuning.pdf", bbox_inches="tight")
fig.savefig("fig_fixed_D_tension_tuning.png", dpi=600, bbox_inches="tight")
```

---

# Optional caption draft

**Figure X. Tension-controlled liquid--structure configuration for a plate with fixed bending stiffness.** The bending stiffness is fixed as \(D=D_0\), so that \(\kappa=\rho g l^4/D_0=2\pi^4\) remains constant. Increasing the imposed tension \(T\) increases \(\tau=Tl^2/D_0\) and decreases the effective CDL parameter \(\lambda_2\), driving the system from the liquid-prevailing regime to the structure-prevailing regime through the equipoise state at \(T=T_{\mathrm{eq}}\). The point \(T=0\) recovers the pure-bending plate limit, while \(T=T_{\mathrm{eq}}\) corresponds to the coupled equipoise condition.
