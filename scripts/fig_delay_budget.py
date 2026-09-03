"""
Figure 1: the delay budget of CHIME's 21 cm auto-spectrum analysis.

Schematic, not a measurement. The BAO curve is sin(k r_d) exp(-k^2 Sigma^2/2)
with r_d = 100 h^-1 Mpc and Sigma = 7 h^-1 Mpc -- the right wiggle spacing and
damping, arbitrary amplitude. Everything else on the figure (the 200 ns cut,
the 280 ns retained floor, the 33 ns standing wave, and the scenario-dependent
5 sigma target band) is
sourced: Amiri et al. 2025 (arXiv:2511.19620) and the RFIsher sweep.

Delay <-> k_par conversion at z = 1.16:
    k_par = 2 pi nu_21 H0 E(z) tau / [c (1+z)^2]  =  1.245 h/Mpc per microsecond
Check: tau = 280 ns -> 0.35 h/Mpc, the published analysis floor.

Run:  python3 fig_delay_budget.py
Out:  fig_delay_budget.pdf / .png
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from figstyle import use, TEXTWIDTH

use()

KPER_US = 1.245          # h/Mpc per microsecond of delay, at z = 1.16
RD = 100.0               # sound horizon, h^-1 Mpc
SIGMA_NL = 7.0           # BAO damping, h^-1 Mpc

TAU_CUT = 200.0          # DAYENU high-pass cutoff, ns
TAU_MASK = 280.0         # everything below this is discarded, ns
TAU_FG = 45.0            # illustrative short-baseline foreground extent, ns
TAU_SW = 33.4            # feed-reflector standing wave (~5 m cavity), ns
REQ_LO, REQ_HI = 100.0, 150.0   # scenario-dependent 5 sigma target range, ns

tau = np.linspace(0, 420, 2000)
k = KPER_US * tau / 1000.0
wiggle = np.sin(k * RD) * np.exp(-0.5 * (k * SIGMA_NL) ** 2)

box = dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85)

fig, ax = plt.subplots(figsize=(TEXTWIDTH, 3.1), dpi=200)

# regions
ax.axvspan(0, TAU_FG, color="0.75", alpha=0.9, lw=0)
ax.text(TAU_FG / 2, 0.79, "intrinsic\nforegrounds\n(schematic)", ha="center", va="center",
        fontsize=6.8, bbox=box)
ax.axvspan(TAU_CUT, TAU_MASK, color="#f4b6c2", alpha=0.8, lw=0)
ax.text((TAU_CUT + TAU_MASK) / 2, 0.80, "filter\ntransition", ha="center",
        va="center", fontsize=7.5, bbox=box)
ax.axvspan(TAU_MASK, 420, color="#b7e1b0", alpha=0.8, lw=0)
ax.text(350, 0.80, "kept in CHIME 2025\n$k_\\parallel > 0.35$", ha="center",
        va="center", fontsize=7.5, bbox=box)

# 5 sigma requirement
ax.axvspan(REQ_LO, REQ_HI, color="#ffd966", alpha=0.55, lw=0)
ax.text((REQ_LO + REQ_HI) / 2, -0.72, "scenario-dependent 5$\\sigma$ target\n$\\tau_{\\rm cut}=100$--$150$ ns",
        ha="center", va="center", fontsize=7.5, bbox=box)

# discarded span
ax.annotate("", xy=(0, -0.95), xytext=(TAU_MASK, -0.95),
            arrowprops=dict(arrowstyle="<->", lw=1.1))
ax.text(140, -1.02, "DAYENU cut: 200 ns; retained floor: 280 ns",
        ha="center", va="top", fontsize=7.5)

# standing wave
ax.axvline(TAU_SW, color="C3", lw=1.4)
ax.text(36, -0.40, "33 ns standing wave\n(validation case)", color="C3",
        ha="left", va="center", fontsize=7.5, bbox=box)

# BAO curve, faded where foregrounds make it irrecoverable
m = tau > TAU_FG
ax.plot(tau[~m], wiggle[~m], color="C0", lw=1.2, alpha=0.35)
ax.plot(tau[m], wiggle[m], color="C0", lw=1.6)
ax.text(140, 0.55, "BAO wiggles: $k \\simeq 0.05$ to $0.3\\,h\\,$Mpc$^{-1}$",
        color="C0", ha="center", fontsize=8, bbox=box)

# cut-limiting population
ax.annotate("", xy=(300, 0.22), xytext=(REQ_LO, 0.22),
            arrowprops=dict(arrowstyle="<->", lw=1.0, color="0.3"))
ax.text(205, 0.30,
        "cut-limiting population: cross-talk paths (to ~260 ns),\n"
        "cable reflections; censused first",
        ha="center", va="bottom", fontsize=7.2, style="italic", bbox=box)

ax.axvline(TAU_CUT, color="k", lw=1.0)
ax.text(TAU_CUT - 2, -0.10, "200 ns cut", fontsize=7.5, ha="right",
        va="center", bbox=box)

ax.set_xlim(0, 420)
ax.set_ylim(-1.15, 1.02)
ax.set_xlabel("delay $\\tau$ [ns]")
ax.set_ylabel("BAO wiggle along $k_\\parallel$ (schematic)")
ax.set_yticks([])
sec = ax.secondary_xaxis("top", functions=(lambda t: KPER_US * t / 1000.0,
                                           lambda kk: 1000.0 * kk / KPER_US))
sec.set_xlabel("$k_\\parallel$ [$h\\,$Mpc$^{-1}$] at $z = 1.16$")
for side in ("top", "right"):
    ax.spines[side].set_visible(False)

fig.tight_layout()
fig.savefig("fig_delay_budget.pdf")
fig.savefig("fig_delay_budget.png")
print(f"tau = {TAU_MASK:.0f} ns -> k_par = {KPER_US * TAU_MASK / 1000:.3f} h/Mpc "
      f"(published floor 0.35)")
