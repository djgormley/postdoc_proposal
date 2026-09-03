"""
Simulated delay-offset coherence map.

Demonstrates the estimator of proposal Eq. (3) on synthetic data built from
Eq. (1): a delay-limited foreground, a 21 cm signal, white noise, and a
multiplicative gain.

Two cases, side by side:
  left   discrete delays: the visibility g_i g_j* for two feeds whose
         reflections share a delay but differ in amplitude
         -> conjugate ridge pairs at Delta = +/- t1, +/- t2
  right  smooth chromaticity  g = 1 + dg(nu), dg with delay content spread
         smoothly to a few hundred ns  -> no ridges, a smear near Delta = 0

The contrast provides the preliminary test in Section 4. Real data may contain
both isolated ridges and broad chromatic structure, requiring a hybrid model.

Run:  python3 sim_coherence.py
Out:  fig_coherence_sim.pdf / .png
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from figstyle import use, TEXTWIDTH

use()

rng = np.random.default_rng(20260901)

# ---------------------------------------------------------------- band
NCHAN = 1024                      # CHIME's channel count
NU0, NU1 = 400e6, 800e6           # full band
dnu = (NU1 - NU0) / NCHAN         # 390.625 kHz
nu = NU0 + dnu * np.arange(NCHAN)
tau = np.fft.fftshift(np.fft.fftfreq(NCHAN, d=dnu))      # s
tau_ns = tau * 1e9                                       # 2.5 ns resolution

NREAL = 600                       # integrations in the ensemble (one declination)

# Blackman-Harris taper along frequency before the delay transform. Real
# reflection delays are not on the FFT grid, so a rectangular window leaks
# each copy across the whole delay axis at ~1/(pi n) and buries weak ridges.
# Standard practice in 21 cm delay spectroscopy, and required here.
_n = np.arange(NCHAN) / (NCHAN - 1)
WIN = (0.35875 - 0.48829 * np.cos(2 * np.pi * _n)
       + 0.14128 * np.cos(4 * np.pi * _n) - 0.01168 * np.cos(6 * np.pi * _n))

# ---------------------------------------------------------------- amplitudes
A_FG = 1.0e4                      # foreground, delay-limited
A_21 = 1.0                        # 21 cm signal, broadband in delay
A_N = 10.0                        # thermal noise per channel
TAU_F = 15.0                      # foreground delay width, ns (1 sigma)

# reflections
T1, E1 = 33.0, 0.010              # feed-reflector standing wave
T2, E2 = 175.0, 0.003             # a long cable reflection


def foreground(n):
    """Delay-limited bright foreground: Gaussian envelope in delay space."""
    env = np.exp(-0.5 * (tau_ns / TAU_F) ** 2)
    z = (rng.normal(size=(n, NCHAN)) + 1j * rng.normal(size=(n, NCHAN))) / np.sqrt(2)
    d = z * env                                  # delay domain
    return A_FG * np.fft.ifft(np.fft.ifftshift(d, axes=-1), axis=-1) * NCHAN


def signal(n):
    z = (rng.normal(size=(n, NCHAN)) + 1j * rng.normal(size=(n, NCHAN))) / np.sqrt(2)
    return A_21 * z


def noise(n):
    z = (rng.normal(size=(n, NCHAN)) + 1j * rng.normal(size=(n, NCHAN))) / np.sqrt(2)
    return A_N * z


def gain_discrete():
    """Visibility gain g_i g_j*: a reflection in feed i lands at +tau, one in
    feed j at -tau, so real ridges come in conjugate pairs."""
    def feed(scale, phase):
        return (1.0 + scale * E1 * np.exp(2j * np.pi * (nu * T1 * 1e-9 + phase))
                    + scale * E2 * np.exp(2j * np.pi * (nu * T2 * 1e-9 + phase)))
    return feed(1.0, 0.0) * np.conj(feed(0.6, 0.17))


def gain_smooth(rms=0.011, scale_ns=120.0):
    """Chromaticity with delay content spread smoothly, no discrete lines."""
    env = np.exp(-np.abs(tau_ns) / scale_ns)
    z = (rng.normal(size=NCHAN) + 1j * rng.normal(size=NCHAN)) / np.sqrt(2)
    dg = np.fft.ifft(np.fft.ifftshift(z * env)) * NCHAN
    dg = dg / np.std(dg) * rms
    return 1.0 + dg


def coherence_map(g, dmax_ns=260.0, tmax_ns=340.0):
    """gamma(tau, Delta) averaged over the ensemble at fixed declination."""
    d = g[None, :] * (foreground(NREAL) + signal(NREAL)) + noise(NREAL)
    D = np.fft.fftshift(np.fft.fft(d * WIN[None, :], axis=-1), axes=-1)
    p_full = np.mean(np.abs(D) ** 2, axis=0)              # <|D(tau)|^2>

    # Shift on the FULL delay grid, then window. Rolling a truncated array
    # wraps its far tail onto its near edge and manufactures correlations.
    keep = np.abs(tau_ns) <= tmax_ns
    t = tau_ns[keep]
    step = tau_ns[1] - tau_ns[0]
    shifts = np.arange(-int(dmax_ns / step), int(dmax_ns / step) + 1)

    G = np.zeros((len(shifts), keep.sum()))
    for i, s in enumerate(shifts):
        Ds = np.roll(D, s, axis=-1)                       # D(tau - Delta)
        C = np.mean(D * np.conj(Ds), axis=0)[keep]
        norm = np.sqrt(p_full * np.roll(p_full, s))[keep]
        G[i] = np.abs(C) / np.where(norm > 0, norm, np.inf)
    return t, shifts * step, G


t, dl, G_disc = coherence_map(gain_discrete())
_, _, G_smooth = coherence_map(gain_smooth())

# ---------------------------------------------------------------- plot
# Top: the two coherence maps. Bottom: a quantitative discriminator, the
# ridge contrast max_tau|gamma| / median_tau|gamma| at each Delta. Discrete
# delays give isolated peaks over a flat floor near 1; smooth chromaticity
# raises the whole plane, so its contrast stays low and featureless.
def contrast(G):
    return G.max(axis=1) / np.median(G, axis=1)

fig = plt.figure(figsize=(TEXTWIDTH, 5.4), dpi=200)
gs = fig.add_gridspec(2, 2, height_ratios=[3.0, 1.35], hspace=0.42, wspace=0.12)
axes = [fig.add_subplot(gs[0, 0]), None]
axes[1] = fig.add_subplot(gs[0, 1], sharey=axes[0])
axc = fig.add_subplot(gs[1, :])

kw = dict(origin="lower", aspect="auto", cmap="magma", vmin=0, vmax=1,
          extent=[t[0], t[-1], dl[0], dl[-1]])
for ax, G, title in zip(
        axes, [G_disc, G_smooth],
        ["discrete delays (33, 175 ns)", "smooth chromaticity, comparable RMS"]):
    im = ax.imshow(G, **kw)
    ax.set_xlabel(r"delay $\tau$ [ns]")
    ax.set_title(title)
axes[1].tick_params(labelleft=False)

for d, lab, col in [(T1, "33", "cyan"), (-T1, None, "cyan"),
                    (T2, "175", "cyan"), (-T2, None, "cyan"),
                    (T2 - T1, "142", "0.7"), (-(T2 - T1), None, "0.7"),
                    (T2 + T1, "208", "0.7"), (-(T2 + T1), None, "0.7")]:
    axes[0].annotate("", xy=(t[0] + 26, d), xytext=(t[0] - 6, d),
                     arrowprops=dict(arrowstyle="->", color=col, lw=0.9))
    if lab:
        axes[0].text(t[0] + 30, d + 9, lab, color=col, fontsize=7.5)
axes[0].text(0.98, 0.03, "cyan: reflections\ngray: sum/difference\n(not reflections)",
             transform=axes[0].transAxes, ha="right", va="bottom", fontsize=7, color="w")
axes[0].set_ylabel(r"delay offset $\Delta$ [ns]")
cb = fig.colorbar(im, ax=axes, fraction=0.03, pad=0.02)
cb.set_label(r"coherence $|\gamma(\tau,\Delta)|$")
cb.ax.tick_params(labelsize=8)

c_disc, c_smooth = contrast(G_disc), contrast(G_smooth)
axc.plot(dl, c_disc, color="C1", lw=1.3, label="discrete delays")
axc.plot(dl, c_smooth, color="0.45", lw=1.3, label="smooth chromaticity")
for d in (T1, -T1, T2, -T2):
    axc.axvline(d, color="cyan", lw=0.6, alpha=0.6)
for d in (T2 - T1, -(T2 - T1), T2 + T1, -(T2 + T1)):
    axc.axvline(d, color="0.7", lw=0.6, alpha=0.6, ls="--")
axc.set_xlim(dl[0], dl[-1])
axc.set_yscale("log")
axc.set_ylim(0.8, 90)
axc.set_xlabel(r"delay offset $\Delta$ [ns]")
axc.set_ylabel(r"ridge contrast  $\max_\tau|\gamma| \,/\, \mathrm{med}_\tau|\gamma|$")
axc.legend(loc="upper center", frameon=False, ncol=2)
axc.grid(alpha=0.25, lw=0.5)

fig.savefig("fig_coherence_sim.pdf", bbox_inches="tight")
fig.savefig("fig_coherence_sim.png", bbox_inches="tight")

# ---------------------------------------------------------------- report
def peak(G, d):
    return G[np.argmin(np.abs(dl - d))].max()

empty = np.array([peak(G_disc, d) for d in (60, 90, 110, 240, 255)])
print(f"N = {NREAL}; off-ridge floor {empty.mean():.3f} (1/sqrt(N) = {1/np.sqrt(NREAL):.3f})")
for lab, c in [("discrete", c_disc), ("smooth  ", c_smooth)]:
    on = [c[np.argmin(np.abs(dl - d))] for d in (T1, T2)]
    off = np.median(c[np.abs(dl) > 5])
    print(f"  contrast {lab}: at 33/175 ns = {on[0]:.1f}/{on[1]:.1f}; median off-ridge = {off:.2f}")
for lab, G in [("discrete", G_disc), ("smooth  ", G_smooth)]:
    print(f"  {lab}  +33 {peak(G,T1):.3f}  -33 {peak(G,-T1):.3f}  "
          f"+175 {peak(G,T2):.3f}  -175 {peak(G,-T2):.3f}  "
          f"142 {peak(G,T2-T1):.3f}  208 {peak(G,T1+T2):.3f}")
