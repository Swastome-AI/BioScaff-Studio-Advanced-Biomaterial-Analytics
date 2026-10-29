"""Publication-quality figures for the condition pipeline."""
from __future__ import annotations

import textwrap

import numpy as np
import matplotlib.pyplot as plt

from core import md_figures as base
from core.models import METRICS

color = base.color


def configure(width, height, font):
    base.configure(width, height, font, "Scaffold")


def _fig(ncols=1, nrows=1, w=None, h=None):
    return base._fig(ncols=ncols, nrows=nrows, w=w, h=h)


def _wrap(name, width=14):
    return "\n".join(textwrap.wrap(str(name), width)) or str(name)


def _ticks(ax, names):
    ax.set_xticks(range(len(names)), [_wrap(n) for n in names])
    if max(len(str(n)) for n in names) > 14 or len(names) > 5:
        for t in ax.get_xticklabels():
            t.set_rotation(30)
            t.set_ha("right")


def time_unit(hours_max):
    """Readable unit for a time axis given its largest value in hours."""
    if hours_max < 1 / 60:
        return "s", 3600.0
    if hours_max < 2:
        return "min", 60.0
    if hours_max < 72:
        return "h", 1.0
    return "days", 1 / 24


def _ci_note(ax):
    ax.text(1.0, 1.01, "median and 95 % interval", transform=ax.transAxes, ha="right",
            va="bottom", fontsize=base._CFG["font"] - 2, color="0.4")


def composition(rows):
    """rows: list of (condition name, {component: % w/v})."""
    comps = []
    for _, d in rows:
        comps += [c for c in d if c not in comps]
    names = [r[0] for r in rows]
    fig, ax = _fig(w=max(base._CFG["w"], 0.8 * len(rows) + 2.5))
    ax = ax[0, 0]
    bottom = np.zeros(len(rows))
    for k, c in enumerate(comps):
        vals = np.array([d.get(c, 0.0) for _, d in rows])
        ax.bar(range(len(rows)), vals, 0.62, bottom=bottom, color=color(k), label=c,
               edgecolor="black", linewidth=0.5)
        bottom += vals
    for i, b in enumerate(bottom):
        if b == 0:
            ax.text(i, 0.02 * max(bottom.max(), 1), "water\nonly", ha="center", va="bottom",
                    fontsize=base._CFG["font"] - 2, color="0.4")
    _ticks(ax, names)
    ax.set_ylabel("Concentration (% w/v)")
    if comps:
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))
    return fig


def metric_bars(items, key, ylabel=None, scale=1.0, ax=None, log=False, ref=None):
    """items: list of (condition name, colour index, summary dict)."""
    rows = [(n, ci, s[key]) for n, ci, s in items if key in s and np.isfinite(s[key]["median"])]
    if not rows:
        return None
    own = ax is None
    if own:
        fig, ax = _fig(w=max(base._CFG["w"] * 0.7, 0.75 * len(rows) + 1.6))
        ax = ax[0, 0]
    label, unit, _ = METRICS[key]
    if key.endswith("_h"):                       # times: pick a readable unit
        unit, scale = time_unit(max(r[2]["hi"] for r in rows))
        if ylabel is None:
            ylabel = f"{label} ({unit})"
    med = np.array([r[2]["median"] for r in rows]) * scale
    lo = np.array([r[2]["lo"] for r in rows]) * scale
    hi = np.array([r[2]["hi"] for r in rows]) * scale
    yerr = np.vstack([np.abs(med - lo), np.abs(hi - med)])
    ax.bar(range(len(rows)), med, 0.62, color=[color(r[1]) for r in rows], yerr=yerr,
           edgecolor="black", linewidth=0.6, capsize=3, error_kw={"elinewidth": 0.8})
    if ref is not None:
        ax.axhline(ref, color="0.4", lw=0.8, ls=":")
    if (med < 0).any():
        ax.axhline(0, color="black", lw=0.6)
    if log and hi.max() / max(lo.min(), 1e-12) > 20:
        ax.set_yscale("log")
    _ticks(ax, [r[0] for r in rows])
    ax.set_ylabel(ylabel or (f"{label} ({unit})" if unit else label))
    if own:
        _ci_note(ax)
        return fig
    return ax


def panel(items, keys, ncols=2):
    keys = [k for k in keys if any(k in s and np.isfinite(s[k]["median"]) for _, _, s in items)]
    if not keys:
        return None
    nrows = int(np.ceil(len(keys) / ncols))
    w = base._CFG["w"] * (1.0 if ncols == 1 else 1.45)
    fig, axes = _fig(ncols=ncols, nrows=nrows, w=w, h=base._CFG["h"] * 0.85 * nrows)
    for i, k in enumerate(keys):
        ax = axes.flat[i]
        metric_bars(items, k, ax=ax)
        base._panel(ax, "abcdefgh"[i])
    for ax in list(axes.flat)[len(keys):]:
        ax.axis("off")
    _ci_note(axes.flat[0])
    return fig


def charge_curves(curve_items, pH):
    """curve_items: list of (name, colour index, (grid, med, lo, hi))."""
    if not curve_items:
        return None
    fig, ax = _fig()
    ax = ax[0, 0]
    for name, ci, (g, med, lo, hi) in curve_items:
        ax.fill_between(g, lo, hi, color=color(ci), alpha=0.18, linewidth=0)
        ax.plot(g, med, color=color(ci), label=name)
    ax.axhline(0, color="black", lw=0.6)
    ax.axvline(pH, color="0.4", lw=0.8, ls="--")
    ax.text(pH, ax.get_ylim()[1], f" pH {pH:g}", va="top", fontsize=base._CFG["font"] - 1,
            color="0.3")
    ax.set_xlabel("pH")
    ax.set_ylabel("Net charge (e per kDa of scaffold)")
    ax.set_xlim(2, 12)
    ax.legend(loc="upper right")
    return fig


def sieving(curve_items, drug_markers):
    """D/D0 versus solute radius, with drug radii marked."""
    if not curve_items:
        return None
    fig, ax = _fig()
    ax = ax[0, 0]
    for name, ci, (r, med, lo, hi) in curve_items:
        ax.fill_between(r, lo, hi, color=color(ci), alpha=0.18, linewidth=0)
        ax.plot(r, med, color=color(ci), label=name)
    for dname, rad in drug_markers:
        ax.axvline(rad, color="0.35", lw=0.8, ls="--")
        ax.text(rad, 0.02, f" {dname}", rotation=90, va="bottom", ha="right",
                fontsize=base._CFG["font"] - 2, color="0.3")
    ax.set_xscale("log")
    ax.set_xlabel("Solute hydrodynamic radius (nm)")
    ax.set_ylabel("Relative diffusivity D/D₀")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower left")
    return fig


def release(curve_items):
    if not curve_items:
        return None
    fig, ax = _fig()
    ax = ax[0, 0]
    t90 = []
    for _, _, (t, med, _, _) in curve_items:
        t90.append(t[np.searchsorted(med, 90, side="left").clip(0, len(t) - 1)])
    unit, sc = time_unit(max(t90))
    for name, ci, (t, med, lo, hi) in curve_items:
        ax.fill_between(t * sc, lo, hi, color=color(ci), alpha=0.18, linewidth=0)
        ax.plot(t * sc, med, color=color(ci), label=name)
    ax.axhline(50, color="0.6", lw=0.7, ls=":")
    ax.set_xscale("log")
    ax.set_xlabel(f"Time ({unit})")
    ax.set_ylabel("Cumulative drug release (%)")
    ax.set_ylim(0, 102)
    ax.legend(loc="upper left")
    return fig


def pores(curve_items, widths):
    """Pore-size distributions with bacterium widths marked."""
    if not curve_items:
        return None
    fig, ax = _fig()
    ax = ax[0, 0]
    for name, ci, (d, pdf) in curve_items:
        dens = pdf * d                      # per log-diameter, comparable shapes
        ax.plot(d, dens / dens.max(), color=color(ci), label=name)
        ax.fill_between(d, 0, dens / dens.max(), color=color(ci), alpha=0.12, linewidth=0)
    for bname, w in widths:
        ax.axvline(w, color="black", lw=0.9, ls="--")
        ax.text(w, 1.02, f" {bname}", rotation=90, va="bottom", ha="center",
                fontsize=base._CFG["font"] - 2)
    ax.set_xscale("log")
    ax.set_xlabel("Pore diameter (µm)")
    ax.set_ylabel("Relative frequency (log scale)")
    ax.set_ylim(0, 1.25)
    ax.legend(loc="upper right")
    return fig


def molecule_properties(entries):
    """Side-by-side bars of key molecular descriptors."""
    if not entries:
        return None
    names = [e["name"] for e in entries]
    specs = [("mw", "Molecular weight (kDa)", 1e-3),
             ("polar_sasa_fraction", "Polar surface fraction", 1),
             ("N_plus_O_per_kDa", "Polar atoms per kDa", 1),
             ("hydrodynamic_radius_nm", "Size: equivalent radius (nm)", 1)]
    fig, axes = _fig(ncols=2, nrows=2, w=base._CFG["w"] * 1.45, h=base._CFG["h"] * 1.7)
    for i, (key, label, sc) in enumerate(specs):
        ax = axes.flat[i]
        vals = []
        for e in entries:
            if key == "mw":
                vals.append(e["mw"] * sc)
            elif key == "N_plus_O_per_kDa":
                vals.append(e["props"]["N_plus_O"] / (e["mw"] / 1000))
            elif key == "hydrodynamic_radius_nm":
                vals.append(e["hydrodynamic_radius_nm"])
            else:
                vals.append(e["props"][key])
        ax.bar(range(len(vals)), vals, 0.62, color=[color(k) for k in range(len(vals))],
               edgecolor="black", linewidth=0.6)
        _ticks(ax, names)
        ax.set_ylabel(label)
        if key == "mw" and max(vals) / max(min(vals), 1e-9) > 50:
            ax.set_yscale("log")
        base._panel(ax, "abcd"[i])
    return fig
