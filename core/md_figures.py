"""
BioScaff Studio - publication-quality Matplotlib figures.

Each builder takes the computed results and returns a Figure. The app saves
them as PNG (chosen dpi), SVG and PDF (vector, editable text).
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from core.md import PPII_BOX, RELEASE_MODELS  # noqa: E402

# Okabe-Ito: colour-blind safe, prints well in greyscale order.
PALETTE = ["#0072B2", "#E69F00", "#009E73", "#D55E00",
           "#CC79A7", "#56B4E9", "#F0E442", "#000000"]

_CFG = {"w": 5.0, "h": 3.6, "font": 9, "solute": "Solute"}


def configure(width, height, font, solute_name="Solute"):
    _CFG.update(w=width, h=height, font=font, solute=solute_name or "Solute")
    f = font
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
        "font.size": f, "axes.labelsize": f, "axes.titlesize": f + 1,
        "xtick.labelsize": f - 1, "ytick.labelsize": f - 1,
        "legend.fontsize": f - 1, "legend.frameon": False,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.8, "xtick.major.width": 0.8, "ytick.major.width": 0.8,
        "xtick.direction": "out", "ytick.direction": "out",
        "lines.linewidth": 1.5, "lines.markersize": 4,
        "axes.prop_cycle": plt.cycler(color=PALETTE),
        "savefig.bbox": "tight", "savefig.pad_inches": 0.05,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "figure.dpi": 100,
    })


def color(i):
    return PALETTE[i % len(PALETTE)]


def _fig(ncols=1, nrows=1, w=None, h=None):
    fig, axes = plt.subplots(nrows, ncols, squeeze=False, constrained_layout=True,
                             figsize=(w or _CFG["w"], h or _CFG["h"]))
    return fig, axes


def _panel(ax, letter):
    ax.text(-0.02, 1.02, letter, transform=ax.transAxes, fontweight="bold",
            fontsize=_CFG["font"] + 2, ha="right", va="bottom")


def _mean_sd(series):
    s = np.asarray(series, float)
    return s.mean(), (s.std(ddof=1) if len(s) > 1 else 0.0)


def _ns(df):
    return df["time_ps"].to_numpy() / 1000.0


def _bar_style():
    return {"edgecolor": "black", "linewidth": 0.6, "capsize": 3,
            "error_kw": {"elinewidth": 0.8}}


def _note_sd(ax, systems):
    n = [len(d["table"]) for d in systems.values()]
    if max(n) > 1:
        ax.text(1.0, 1.0, "mean ± SD over frames", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=_CFG["font"] - 2, color="0.35")


# --------------------------------------------------------------------------

def composition(rows):
    """rows: list of (system label, {category: atom count})."""
    cats = []
    for _, d in rows:
        cats += [c for c in d if c not in cats]
    labels = [r[0] for r in rows]
    fig, ax = _fig(h=max(1.4, 0.55 * len(rows) + 1.0))
    ax = ax[0, 0]
    left = np.zeros(len(rows))
    for k, c in enumerate(cats):
        vals = np.array([d.get(c, 0) for _, d in rows], float)
        ax.barh(labels, vals, left=left, color=color(k), edgecolor="black",
                linewidth=0.5, label=c, height=0.6)
        left += vals
    ax.invert_yaxis()
    ax.set_xlabel("Number of atoms")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))
    return fig


def sasa_bars(systems):
    labels = list(systems)
    x = np.arange(len(labels))
    w = 0.38
    fig, ax = _fig()
    ax = ax[0, 0]
    tops = np.zeros(len(labels))
    for j, (col, name) in enumerate([("polar_A2", "Polar (N, O and their H)"),
                                     ("nonpolar_A2", "Nonpolar")]):
        stats = [_mean_sd(systems[l]["table"][col] / 100.0) for l in labels]
        m, sd = np.array(stats).T
        ax.bar(x + (j - 0.5) * w, m, w, yerr=sd, color=color(j), label=name, **_bar_style())
        tops = np.maximum(tops, m + sd)
    for i, l in enumerate(labels):
        pf = systems[l]["table"]["polar_fraction"].mean()
        ax.text(x[i], tops[i] * 1.03, f"{pf:.0%} polar", ha="center", va="bottom",
                fontsize=_CFG["font"] - 1)
    ax.set_ylim(0, tops.max() * 1.18)
    ax.set_xticks(x, labels)
    ax.set_ylabel("SASA (nm²)")
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.12), ncols=2)
    _note_sd(ax, systems)
    return fig


def timeseries(systems, col, ylabel, scale=1.0, ref=None):
    fig, ax = _fig()
    ax = ax[0, 0]
    for i, (l, d) in enumerate(systems.items()):
        t = d["table"]
        if col in t:
            ax.plot(_ns(t), t[col] * scale, color=color(i), label=l)
    if ref is not None:
        ax.axhline(ref, color="0.5", lw=0.8, ls=":")
    ax.set_xlabel("Time (ns)")
    ax.set_ylabel(ylabel)
    ax.legend()
    return fig


def component_bars(systems, prefix, suffix, ylabel, scale=1.0):
    """Grouped bars: one group per component, one bar per system."""
    comps = []
    for d in systems.values():
        for c in d["table"].columns:
            if c.startswith(prefix) and c.endswith(suffix):
                name = c[len(prefix):len(c) - len(suffix)]
                if name and name not in ("solute", "total", "polar", "nonpolar",
                                         "total_per_heavy_atom") and name not in comps:
                    comps.append(name)
    if not comps:
        return None
    labels = list(systems)
    x = np.arange(len(comps))
    w = 0.8 / len(labels)
    fig, ax = _fig()
    ax = ax[0, 0]
    for i, l in enumerate(labels):
        t = systems[l]["table"]
        stats = [(_mean_sd(t[f"{prefix}{c}{suffix}"] * scale) if f"{prefix}{c}{suffix}" in t
                  else (np.nan, 0.0)) for c in comps]
        m, sd = np.array(stats).T
        ax.bar(x + (i - (len(labels) - 1) / 2) * w, m, w, yerr=sd, color=color(i),
               label=l, **_bar_style())
    ax.set_xticks(x, comps)
    ax.set_ylabel(ylabel)
    ax.legend()
    _note_sd(ax, systems)
    return fig


def hbond_bars(systems):
    labels = list(systems)
    x = np.arange(len(labels))
    fig, axes = _fig(ncols=2)
    specs = [("solute_water_per_polar_atom",
              f"{_CFG['solute']}–water H-bonds\nper polar atom (N, O)"),
             ("solute_solute", f"{_CFG['solute']}–{_CFG['solute'].lower()} H-bonds")]
    for k, (col, ylabel) in enumerate(specs):
        ax = axes[0, k]
        stats = [_mean_sd(systems[l]["table"][col]) for l in labels]
        m, sd = np.array(stats).T
        ax.bar(x, m, 0.6, yerr=sd, color=[color(i) for i in range(len(labels))],
               **_bar_style())
        ax.set_xticks(x, labels)
        ax.set_ylabel(ylabel)
        _panel(ax, "ab"[k])
    _note_sd(axes[0, 1], systems)
    return fig


def hbond_matrix(systems):
    items = [(l, d["matrix"]) for l, d in systems.items() if d.get("matrix") is not None]
    if not items:
        return None
    n = len(items)
    side = min(_CFG["w"] / n, 3.6)
    fig, axes = _fig(ncols=n, w=side * n + 0.6, h=side + 0.4)
    vmax = max(float(m.to_numpy().max()) for _, m in items) or 1.0
    for k, (l, m) in enumerate(items):
        ax = axes[0, k]
        im = ax.imshow(m.to_numpy(), cmap="Blues", vmin=0, vmax=vmax)
        ax.set_xticks(range(len(m)), m.columns, rotation=45, ha="right")
        ax.set_yticks(range(len(m)), m.index)
        for i in range(len(m)):
            for j in range(len(m)):
                v = m.iat[i, j]
                ax.text(j, i, f"{v:.1f}", ha="center", va="center",
                        fontsize=_CFG["font"] - 1,
                        color="white" if v > 0.6 * vmax else "black")
        ax.set_title(l)
        for s in ("top", "right"):
            ax.spines[s].set_visible(True)
    cb = fig.colorbar(im, ax=axes[0, :].tolist(), shrink=0.8)
    cb.set_label("H-bonds per frame")
    return fig


def free_volume(systems):
    labels = list(systems)
    fig, axes = _fig(ncols=2)
    ax = axes[0, 0]
    x = np.arange(len(labels))
    w = 0.38
    for j, (key, name) in enumerate([("free_volume_fraction", "Outside all atoms"),
                                     ("probe_accessible_fraction", "Accessible to water probe")]):
        vals = [systems[l]["summary"][key] * 100 for l in labels]
        ax.bar(x + (j - 0.5) * w, vals, w, color=color(j), label=name, **_bar_style())
    ax.set_xticks(x, labels)
    ax.set_ylabel("Fraction of volume (%)")
    ax.set_ylim(0, 100)
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.12))
    _panel(ax, "a")
    ax = axes[0, 1]
    for i, l in enumerate(labels):
        h = systems[l]["hist"]
        ax.step(h["clearance_A"], h["fraction_of_free_volume"] * 100, where="mid",
                color=color(i), label=l)
    ax.set_xlabel("Local pore radius, clearance (Å)")
    ax.set_ylabel("Share of free volume (%)")
    ax.legend()
    _panel(ax, "b")
    return fig


def rg_bars(systems):
    labels = list(systems)
    x = np.arange(len(labels))
    fig, ax = _fig(w=min(_CFG["w"], 1.2 + 1.1 * len(labels)))
    ax = ax[0, 0]
    stats = [_mean_sd(systems[l]["table"]["Rg_solute_A"] / 10.0) for l in labels]
    m, sd = np.array(stats).T
    ax.bar(x, m, 0.6, yerr=sd, color=[color(i) for i in range(len(labels))], **_bar_style())
    ax.set_xticks(x, labels)
    ax.set_ylabel("Radius of gyration (nm)")
    _note_sd(ax, systems)
    return fig


def msd(systems):
    fig, ax = _fig()
    ax = ax[0, 0]
    for i, (l, d) in enumerate(systems.items()):
        t, info = d["table"], d["info"]
        lag_ns = t["lag_ps"] / 1000.0
        ax.plot(lag_ns, t["MSD_A2"] / 100.0, color=color(i),
                label=f"{l}: D = {info['D_cm2_per_s']:.2e} cm²/s")
        sel = (t["lag_ps"] >= info["fit_start_ps"]) & (t["lag_ps"] <= info["fit_end_ps"])
        fit = info["slope"] * t["lag_ps"][sel] + info["intercept"]
        ax.plot(lag_ns[sel], fit / 100.0, color="black", ls="--", lw=0.9)
    ax.plot([], [], color="black", ls="--", lw=0.9, label="Linear fit region")
    ax.set_xlabel("Lag time (ns)")
    ax.set_ylabel("MSD (nm²)")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.legend()
    return fig


def rdf(systems):
    fig, axes = _fig(ncols=2)
    for i, (l, d) in enumerate(systems.items()):
        t, s = d["table"], d["summary"]
        axes[0, 0].plot(t["r_A"], t["g_r"], color=color(i), label=l)
        axes[0, 1].plot(t["r_A"], t["n_r"], color=color(i), label=l)
        if "first_min_r_A" in s:
            axes[0, 1].plot(s["first_min_r_A"], s["coordination_number"], "o",
                            color=color(i), mfc="white")
    axes[0, 0].axhline(1.0, color="0.5", lw=0.8, ls=":")
    axes[0, 0].set_xlabel("r (Å)")
    axes[0, 0].set_ylabel("g(r)")
    axes[0, 0].legend()
    axes[0, 1].set_xlabel("r (Å)")
    axes[0, 1].set_ylabel("Cumulative coordination n(r)")
    _panel(axes[0, 0], "a")
    _panel(axes[0, 1], "b")
    return fig


def ramachandran(systems):
    n = len(systems)
    side = min(_CFG["w"] / n, 3.4)
    fig, axes = _fig(ncols=n, w=side * n, h=side + 0.3)
    p0, p1 = PPII_BOX["phi"], PPII_BOX["psi"]
    for k, (l, d) in enumerate(systems.items()):
        ax = axes[0, k]
        t = d["table"]
        alpha = float(np.clip(300 / max(len(t), 1), 0.05, 0.8))
        ax.scatter(t["phi_deg"], t["psi_deg"], s=3, alpha=alpha, color=color(k),
                   linewidths=0, rasterized=True)
        ax.add_patch(plt.Rectangle((p0[0], p1[0]), p0[1] - p0[0], p1[1] - p1[0],
                                   fill=False, ls="--", lw=0.9, ec="black"))
        frac = d["per_frame"]["PPII_fraction"].mean()
        ax.text(p0[1] + 4, p1[1] - 4, f"PPII\n{frac:.0%}", va="top",
                fontsize=_CFG["font"] - 1)
        ax.axhline(0, color="0.8", lw=0.5, zorder=0)
        ax.axvline(0, color="0.8", lw=0.5, zorder=0)
        ax.set_xlim(-180, 180)
        ax.set_ylim(-180, 180)
        ax.set_xticks([-180, -90, 0, 90, 180])
        ax.set_yticks([-180, -90, 0, 90, 180])
        ax.set_aspect("equal")
        ax.set_xlabel("φ (°)")
        if k == 0:
            ax.set_ylabel("ψ (°)")
        ax.set_title(l)
        for s in ("top", "right"):
            ax.spines[s].set_visible(True)
    return fig


def release(groups, model, time_label):
    """groups: {label: {"t": array, "q": array, "sd": array|None, "fits": dict}}"""
    fig, ax = _fig()
    ax = ax[0, 0]
    f = RELEASE_MODELS[model]["f"]
    for i, (l, g) in enumerate(groups.items()):
        ax.errorbar(g["t"], g["q"], yerr=g.get("sd"), fmt="o", color=color(i),
                    mfc="white", mew=1.0, capsize=2, elinewidth=0.8, label=l)
        if model in g["fits"]:
            tmax = g["t"].max()
            if model == "Korsmeyer–Peppas":
                tmax = g["t"][g["q"] <= 60].max() if (g["q"] <= 60).any() else tmax
            tt = np.linspace(max(g["t"].min(), 0), tmax, 200)
            ax.plot(tt, f(tt, *g["fits"][model]), color=color(i), lw=1.2)
    ax.plot([], [], color="0.3", lw=1.2, label=f"{model} fit")
    ax.set_xlabel(time_label)
    ax.set_ylabel("Cumulative release (%)")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.legend()
    return fig
