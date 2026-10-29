"""
Condition-level models (CPU, seconds).

These are screening estimates from established physical models, computed from
the molecules' structures and the composition of each condition. They are not
molecular dynamics. Uncertain inputs are propagated by Monte Carlo sampling,
so every result comes with a 95 % interval.

Models
------
* Hydrophilicity: mass-weighted polar fraction of the solvent-accessible
  surface and polar (N + O) atoms per kDa.
* Charge: Henderson–Hasselbalch over ionizable groups → net, cationic and
  anionic charge per kDa of scaffold at the chosen pH (cationic charge of
  chitosan-type amines is a recognised proxy for contact antibacterial
  activity).
* Molecular sieving: Ogston (1973) obstruction model for diffusion through a
  random fibre network, D/D0 = exp(−(rs + rf)/rf · φ^½).
* Drug release: Fickian release from a slab releasing from both faces
  (Crank 1975), with D from Stokes–Einstein × Ogston.
* Bacterial passage (optional): fraction of pores wider than the bacterium,
  for a log-normal pore-size distribution the user supplies.
"""
from __future__ import annotations

import math

import numpy as np

from core.molecules import charge_at_pH

K_B = 1.380649e-23
BACTERIA = {   # typical cell dimensions (µm): (width, length)
    "Staphylococcus aureus (coccus)": (0.8, 0.8),
    "Escherichia coli (rod)": (0.5, 2.0),
    "Pseudomonas aeruginosa (rod)": (0.6, 2.0),
    "Streptococcus pyogenes (coccus)": (0.8, 0.8),
    "Custom": (1.0, 1.0),
}

DEFAULT_SETTINGS = {
    "pH": 7.4, "T_C": 37.0, "thickness_mm": 1.0,
    "n_samples": 2000, "seed": 42,
    "sd_pKa": 0.3, "rel_conc": 0.10, "rel_density": 0.05, "rel_chain": 0.20,
    "rel_drug": 0.15, "rel_thickness": 0.10, "rel_pore": 0.20,
}

METRICS = {   # key: (label, unit, description)
    "conc_total": ("Total scaffold concentration", "% w/v", "Sum of component concentrations."),
    "phi": ("Polymer volume fraction φ", "", "Volume occupied by scaffold chains."),
    "polar_sasa_fraction": ("Polar surface fraction", "", "Share of exposed surface that is N/O."),
    "polar_per_kDa": ("Polar atoms per kDa", "per kDa", "H-bonding N + O atoms per mass."),
    "net_charge_per_kDa": ("Net charge", "e/kDa", "Net charge density at the chosen pH."),
    "cationic_per_kDa": ("Cationic charge", "e/kDa", "Positive charge density (antibacterial proxy)."),
    "anionic_per_kDa": ("Anionic charge", "e/kDa", "Magnitude of negative charge density."),
    "D_rel": ("Drug relative diffusivity D/D0", "", "Hindrance of the drug by the network."),
    "D_cm2s": ("Drug diffusivity in scaffold", "cm²/s", "Stokes–Einstein × Ogston."),
    "t50_h": ("Time to 50 % release", "h", "Fickian slab release."),
    "t90_h": ("Time to 90 % release", "h", "Fickian slab release."),
    "pore_pass_fraction": ("Pores wider than the bacterium", "", "Size-exclusion estimate."),
}


# ---------------------------------------------------------------- physics

def water_viscosity(T_K):
    """Pa·s; Vogel equation, valid ~273–373 K."""
    return 2.414e-5 * 10 ** (247.8 / (T_K - 140.0))


def stokes_einstein(r_nm, T_K):
    """m²/s for a sphere of hydrodynamic radius r_nm in water."""
    return K_B * T_K / (6 * np.pi * water_viscosity(T_K) * np.asarray(r_nm) * 1e-9)


def ogston(r_s_nm, r_f_nm, phi):
    phi = np.clip(phi, 0.0, 0.99)
    return np.exp(-((r_s_nm + r_f_nm) / r_f_nm) * np.sqrt(phi))


def slab_release_fraction(tau):
    """Mt/M∞ for a slab of thickness L releasing from both faces; tau = D·t/L²."""
    tau = np.asarray(tau, float)
    out = np.empty_like(tau)
    early = tau < 0.05
    out[early] = 4 * np.sqrt(tau[early] / np.pi)
    if (~early).any():
        k = (2 * np.arange(60) + 1).reshape((-1,) + (1,) * tau.ndim)
        series = (8 / (k ** 2 * np.pi ** 2) * np.exp(-(k ** 2) * np.pi ** 2 * tau[None])).sum(0)
        out[~early] = 1 - series[~early]
    return np.clip(out, 0, 1)


def _tau_for(fraction):
    grid = np.logspace(-6, 1, 20000)
    return float(np.interp(fraction, slab_release_fraction(grid), grid))


TAU50, TAU90 = _tau_for(0.5), _tau_for(0.9)


def lognormal_pass_fraction(width_um, mean_um, cv):
    """P(pore diameter > width) for a log-normal distribution (mean, CV)."""
    sigma = np.sqrt(np.log(1 + cv ** 2))
    mu = np.log(mean_um) - sigma ** 2 / 2
    z = (np.log(width_um) - mu) / (sigma * np.sqrt(2))
    return 0.5 * np.vectorize(math.erfc)(z)


def lognormal_pdf(d, mean_um, cv):
    sigma = np.sqrt(np.log(1 + cv ** 2))
    mu = np.log(mean_um) - sigma ** 2 / 2
    return np.exp(-(np.log(d) - mu) ** 2 / (2 * sigma ** 2)) / (d * sigma * np.sqrt(2 * np.pi))


# ---------------------------------------------------------------- Monte Carlo

def _mult(rng, rel, size):
    """Multiplicative log-normal factor with mean 1 and relative SD `rel`."""
    if rel <= 0:
        return np.ones(size)
    s = np.sqrt(np.log(1 + rel ** 2))
    return np.exp(rng.normal(-s ** 2 / 2, s, size))


def summarise(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if not x.size:
        return {"median": np.nan, "lo": np.nan, "hi": np.nan}
    lo, med, hi = np.percentile(x, [2.5, 50, 97.5])
    return {"median": float(med), "lo": float(lo), "hi": float(hi)}


def _band(a):
    return np.percentile(a, [50, 2.5, 97.5], axis=-1)


def run_conditions(molecules: dict, conditions: list, settings: dict, progress=None):
    """Evaluate every condition. Returns {condition id: result dict}.

    Parameters that belong to a molecule (pKa, density, chain radius, drug
    radius) and the slab thickness are shared across conditions within each
    Monte Carlo draw, so differences between conditions are not blurred by
    re-drawing the same unknown. Concentrations and pore sizes are drawn
    independently per condition (preparation variability).
    """
    s = {**DEFAULT_SETTINGS, **settings}
    n = int(s["n_samples"])
    rng = np.random.default_rng(int(s["seed"]))
    T = s["T_C"] + 273.15
    pH_grid = np.round(np.arange(2.0, 12.01, 0.1), 2)
    r_grid = np.logspace(-1, 1, 80)
    L = s["thickness_mm"] * 1e-3 * _mult(rng, s["rel_thickness"], n)

    P = {}
    for mid, m in molecules.items():
        shift = rng.normal(0, s["sd_pKa"], (len(m["groups"]), n))
        q, qp, qn = charge_at_pH(m["groups"], s["pH"], m["permanent"], shift)
        qg, _, _ = charge_at_pH(m["groups"], pH_grid[:, None], m["permanent"],
                                shift[:, None, :] if len(m["groups"]) else None)
        P[mid] = {"rho": m["density"] * _mult(rng, s["rel_density"], n),
                  "rf": m["chain_radius_nm"] * _mult(rng, s["rel_chain"], n),
                  "rs": m["hydrodynamic_radius_nm"] * _mult(rng, s["rel_drug"], n),
                  "q": q * np.ones(n), "qp": qp * np.ones(n), "qn": qn * np.ones(n),
                  "q_grid": qg * np.ones((len(pH_grid), n))}

    results = {}
    for ci, cond in enumerate(conditions):
        comps = [c for c in cond["components"]
                 if c["mol_id"] in molecules and molecules[c["mol_id"]]["role"] != "Solvent"
                 and c["conc"] > 0]
        out = {"name": cond["name"], "role": cond["role"], "metrics": {}, "curves": {},
               "notes": [], "has_scaffold": bool(comps), "challenge": cond["challenge"]}
        M = out["metrics"]
        if comps:
            conc = {c["mol_id"]: c["conc"] * _mult(rng, s["rel_conc"], n) for c in comps}
            total = sum(conc.values())
            phi_i = {k: (v / 100.0) / P[k]["rho"] for k, v in conc.items()}
            phi = sum(phi_i.values())
            rf = sum(phi_i[k] * P[k]["rf"] for k in conc) / phi
            w = {k: v / total for k, v in conc.items()}
            mw = {k: molecules[k]["mw"] / 1000.0 for k in conc}
            M["conc_total"] = total
            M["phi"] = phi
            M["polar_sasa_fraction"] = sum(w[k] * molecules[k]["props"]["polar_sasa_fraction"]
                                           for k in conc)
            M["polar_per_kDa"] = sum(w[k] * molecules[k]["props"]["N_plus_O"] / mw[k]
                                     for k in conc)
            M["net_charge_per_kDa"] = sum(w[k] * P[k]["q"] / mw[k] for k in conc)
            M["cationic_per_kDa"] = sum(w[k] * P[k]["qp"] / mw[k] for k in conc)
            M["anionic_per_kDa"] = -sum(w[k] * P[k]["qn"] / mw[k] for k in conc)
            qg = sum(w[k] * P[k]["q_grid"] / mw[k] for k in conc)
            out["curves"]["charge_pH"] = (pH_grid, *_band(qg))
            out["curves"]["sieving"] = (r_grid, *_band(ogston(r_grid[:, None], rf, phi)))
            if np.median(phi) > 0.5:
                out["notes"].append("Polymer volume fraction above 0.5: the Ogston model "
                                    "is meant for dilute–semidilute networks.")
        else:
            phi, rf = np.zeros(n), np.ones(n)
            out["notes"].append("No scaffold components: values describe free solution.")
            out["curves"]["sieving"] = (r_grid, np.ones_like(r_grid), np.ones_like(r_grid),
                                        np.ones_like(r_grid))

        if cond["challenge"] == "drug" and cond.get("drug_id") in molecules:
            rs = P[cond["drug_id"]]["rs"]
            D0 = stokes_einstein(rs, T)
            rel = ogston(rs, rf, phi) if comps else np.ones(n)
            D = D0 * rel
            M["D_rel"] = rel
            M["D_cm2s"] = D * 1e4
            M["t50_h"] = TAU50 * L ** 2 / D / 3600
            M["t90_h"] = TAU90 * L ** 2 / D / 3600
            t_med = np.median(M["t50_h"])
            t_grid = np.logspace(np.log10(t_med) - 3, np.log10(np.median(M["t90_h"])) + 0.8, 160)
            frac = slab_release_fraction(D[None, :] * (t_grid[:, None] * 3600) / L[None, :] ** 2)
            out["curves"]["release"] = (t_grid, *_band(frac * 100))
            out["drug"] = molecules[cond["drug_id"]]["name"]
            out["drug_radius_nm"] = molecules[cond["drug_id"]]["hydrodynamic_radius_nm"]

        if cond["challenge"] == "bacterium":
            b = cond.get("bacterium") or {}
            width = float(b.get("width_um", 1.0))
            out["bacterium"] = {**b, "width_um": width}
            pores = cond.get("pores")
            if not comps:
                M["pore_pass_fraction"] = np.ones(n)
                out["notes"].append("No scaffold: nothing blocks the bacterium (positive "
                                    "control).")
            elif pores and pores.get("mean_um", 0) > 0:
                mean = pores["mean_um"] * _mult(rng, s["rel_pore"], n)
                cv = max(pores.get("cv", 0.5), 0.01)
                M["pore_pass_fraction"] = lognormal_pass_fraction(width, mean, cv)
                d = np.logspace(np.log10(max(width, pores["mean_um"]) / 100),
                                np.log10(max(width, pores["mean_um"]) * 10), 300)
                out["curves"]["pores"] = (d, lognormal_pdf(d, pores["mean_um"], cv))
                out["pores"] = pores
            else:
                out["notes"].append("Bacterial passage not estimated: no pore-size input. "
                                    "The cationic charge density is reported as an "
                                    "antibacterial proxy instead.")

        out["samples"] = M
        out["summary"] = {k: summarise(v) for k, v in M.items()}
        results[cond["id"]] = out
        if progress:
            progress((ci + 1) / len(conditions), cond["name"])
    return results


def prob_greater(a, b):
    """Share of Monte Carlo draws in which a > b (aligned draws)."""
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        return np.nan
    ok = np.isfinite(a) & np.isfinite(b)
    return float((a[ok] > b[ok]).mean()) if ok.any() else np.nan
