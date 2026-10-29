"""
BioScaff Studio - analysis core.

Plain Python, no Streamlit, so every function can be tested, reused in a
notebook, or described in a methods section.

Units follow MDAnalysis: lengths in Å, areas in Å², volumes in Å³, time in ps.
Column names say the unit.

Every number here is computed from the uploaded coordinates. Nothing is
simulated, estimated from atom counts, or drawn from a template curve.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.spatial import cKDTree

# --------------------------------------------------------------------------
# Atom properties
# --------------------------------------------------------------------------

# Bondi (1964) van der Waals radii, Å; Na/K/Mg/Ca from Mantina et al. (2009).
BONDI = {
    "H": 1.20, "C": 1.70, "N": 1.55, "O": 1.52, "F": 1.47, "P": 1.80,
    "S": 1.80, "CL": 1.75, "BR": 1.85, "I": 1.98, "SI": 2.10,
    "NA": 2.27, "K": 2.75, "MG": 1.73, "CA": 2.31, "ZN": 1.39, "FE": 1.94,
}
DEFAULT_RADIUS = 1.80
POLAR_ELEMENTS = ("N", "O")


def guess_elements(names, resnames, elements=None) -> np.ndarray:
    """Upper-case element symbols.

    Uses the element column when the file has one. Otherwise guesses from the
    atom name. A name such as "CA" is carbon (protein alpha carbon) unless the
    whole residue is a single ion called CA.
    """
    if elements is not None:
        el = np.asarray(elements, dtype=object)
        if len(el) and all(isinstance(e, str) and e.strip() for e in el):
            return np.array([e.strip().upper() for e in el])
    out = []
    for name, res in zip(names, resnames):
        letters = re.sub(r"[^A-Za-z]", "", str(name)).upper()
        if not letters:
            out.append("X")
        elif letters == str(res).strip().upper() and letters in BONDI:
            out.append(letters)          # monatomic ion: NA, CL, K, CA ...
        elif letters[:2] in ("CL", "BR"):
            out.append(letters[:2])
        else:
            out.append(letters[0])
    return np.array(out)


def atomgroup_elements(ag) -> np.ndarray:
    try:
        el = ag.elements
    except Exception:
        el = None
    return guess_elements(ag.names, ag.resnames, el)


def vdw_radii(elements) -> np.ndarray:
    return np.array([BONDI.get(e, DEFAULT_RADIUS) for e in elements], dtype=float)


def polar_mask(coords, elements) -> np.ndarray:
    """N and O are polar; each H takes the class of its nearest heavy atom."""
    elements = np.asarray(elements)
    polar = np.isin(elements, POLAR_ELEMENTS)
    is_h = elements == "H"
    if is_h.any() and (~is_h).any():
        heavy = np.where(~is_h)[0]
        _, j = cKDTree(coords[heavy]).query(coords[is_h])
        polar[is_h] = polar[heavy[j]]
    return polar


def frame_indices(n_frames, start=0, stop=None, step=1, max_frames=None):
    idx = np.arange(n_frames)[start:stop:max(1, step)]
    if max_frames and len(idx) > max_frames:
        idx = idx[np.linspace(0, len(idx) - 1, max_frames).round().astype(int)]
    return idx


def parse_components(text: str) -> dict:
    """'Label = MDAnalysis selection', one per line. '#' starts a comment."""
    comps = {}
    for raw in (text or "").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if "=" not in line:
            raise ValueError(f"Component line needs 'Label = selection': {raw!r}")
        label, sel = (p.strip() for p in line.split("=", 1))
        if not label or not sel:
            raise ValueError(f"Component line needs 'Label = selection': {raw!r}")
        comps[label] = sel
    return comps


def residue_table(atoms) -> pd.DataFrame:
    df = pd.DataFrame({"resname": atoms.resnames, "res": atoms.resindices})
    return (df.groupby("resname")
              .agg(atoms=("res", "size"), residues=("res", "nunique"))
              .sort_values("atoms", ascending=False)
              .reset_index())


# --------------------------------------------------------------------------
# SASA (Shrake-Rupley)
# --------------------------------------------------------------------------

def _sphere_points(n: int) -> np.ndarray:
    """Near-uniform points on the unit sphere (golden spiral)."""
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    theta = np.pi * (1 + 5 ** 0.5) * i
    return np.column_stack([np.cos(theta) * np.sin(phi),
                            np.sin(theta) * np.sin(phi),
                            np.cos(phi)])


def sasa_per_atom(coords, radii, probe=1.4, n_points=100) -> np.ndarray:
    """Shrake-Rupley solvent-accessible surface area per atom (Å²).

    Pass solute atoms only: water and ions should be removed first, otherwise
    the explicit water hides the surface. Periodic images are not considered,
    so a solute that crosses the box edge should be made whole first.
    """
    coords = np.asarray(coords, dtype=float)
    r = np.asarray(radii, dtype=float) + probe
    sphere = _sphere_points(n_points)
    tree = cKDTree(coords)
    neighbours = tree.query_ball_point(coords, r + r.max())
    out = np.empty(len(coords))
    for i, nb in enumerate(neighbours):
        nb = np.asarray(nb, dtype=int)
        nb = nb[nb != i]
        exposed = n_points
        if nb.size:
            pts = coords[i] + r[i] * sphere
            d2 = ((pts[:, None, :] - coords[nb][None, :, :]) ** 2).sum(-1)
            exposed = n_points - int((d2 < (r[nb] ** 2)[None, :]).any(1).sum())
        out[i] = 4.0 * np.pi * r[i] ** 2 * exposed / n_points
    return out


def run_sasa(universe, solute, components, frames, probe=1.4, n_points=100):
    el = atomgroup_elements(solute)
    radii = vdw_radii(el)
    heavy = int((el != "H").sum())
    masks = {k: np.isin(solute.indices, ag.indices) for k, ag in components.items()}
    polar = None
    rows = []
    for ts in universe.trajectory[list(frames)]:
        pos = solute.positions
        if polar is None:
            polar = polar_mask(pos, el)
        s = sasa_per_atom(pos, radii, probe, n_points)
        row = {"frame": ts.frame, "time_ps": ts.time,
               "total_A2": s.sum(), "polar_A2": s[polar].sum(),
               "nonpolar_A2": s[~polar].sum()}
        row["total_per_heavy_atom_A2"] = row["total_A2"] / max(heavy, 1)
        for k, m in masks.items():
            row[f"{k}_A2"] = s[m].sum()
        rows.append(row)
    df = pd.DataFrame(rows)
    df["polar_fraction"] = df["polar_A2"] / df["total_A2"]
    return df


# --------------------------------------------------------------------------
# Hydrogen bonds
# --------------------------------------------------------------------------

def run_hbonds(universe, solute_sel, water_sel, components, start, stop, step,
               d_a_cutoff=3.0, angle_cutoff=150.0, shell=3.5):
    """Solute-water and solute-solute hydrogen bonds per frame.

    Geometric criteria (MDAnalysis HydrogenBondAnalysis): donor-acceptor
    distance <= d_a_cutoff, donor-H-acceptor angle >= angle_cutoff. Donors
    and acceptors are N and O atoms; donor-H pairs are found by distance
    (1.2 Å), so no bond records are needed. Only water within `shell` Å of
    the solute is searched, which keeps large boxes fast.
    """
    from MDAnalysis.analysis.hydrogenbonds import HydrogenBondAnalysis

    solute = universe.select_atoms(solute_sel)
    if (atomgroup_elements(solute) == "H").sum() == 0:
        raise ValueError("The solute has no hydrogen atoms. Hydrogen bonds need "
                         "explicit hydrogens (add them before solvating).")
    shell_w = f"(byres (({water_sel}) and around {shell} ({solute_sel})))"
    both = f"(({solute_sel}) or {shell_w})"
    hba = HydrogenBondAnalysis(
        universe,
        donors_sel=f"{both} and (name N* or name O*)",
        hydrogens_sel=f"{both} and name H*",
        acceptors_sel=f"{both} and (name N* or name O*)",
        d_h_cutoff=1.2, d_a_cutoff=d_a_cutoff,
        d_h_a_angle_cutoff=angle_cutoff, update_selections=True,
    )
    hba.run(start=start, stop=stop, step=step)
    hb = np.asarray(hba.results.hbonds)
    frames, times = np.asarray(hba.frames), np.asarray(hba.times)

    n_polar = int(np.isin(atomgroup_elements(solute), POLAR_ELEMENTS).sum())
    if len(hb):
        f = hb[:, 0].astype(int)
        d = hb[:, 1].astype(int)
        a = hb[:, 3].astype(int)
        d_sol = np.isin(d, solute.indices)
        a_sol = np.isin(a, solute.indices)
        sw, ss = d_sol ^ a_sol, d_sol & a_sol
    else:
        f = d = a = np.array([], dtype=int)
        sw = ss = np.array([], dtype=bool)

    rows = []
    for fr, t in zip(frames, times):
        here = f == fr
        rows.append({"frame": int(fr), "time_ps": float(t),
                     "solute_water": int((here & sw).sum()),
                     "solute_solute": int((here & ss).sum())})
    df = pd.DataFrame(rows)
    df["solute_water_per_polar_atom"] = df["solute_water"] / max(n_polar, 1)

    matrix = None
    if components:
        labels = list(components)
        lab = np.full(universe.atoms.n_atoms, -1)
        for k, ag in enumerate(components.values()):
            lab[ag.indices] = k
        m = np.zeros((len(labels), len(labels)))
        if ss.any():
            li, lj = lab[d[ss]], lab[a[ss]]
            ok = (li >= 0) & (lj >= 0)
            np.add.at(m, (li[ok], lj[ok]), 1)
        m = m + m.T - np.diag(np.diag(m))          # count i-j and j-i together
        matrix = pd.DataFrame(m / max(len(frames), 1), index=labels, columns=labels)
    return df, matrix


# --------------------------------------------------------------------------
# Size and shape
# --------------------------------------------------------------------------

def run_rg(universe, solute, components, frames):
    rows = []
    for ts in universe.trajectory[list(frames)]:
        row = {"frame": ts.frame, "time_ps": ts.time,
               "Rg_solute_A": solute.radius_of_gyration(),
               "asphericity": solute.asphericity()}
        for k, ag in components.items():
            row[f"Rg_{k}_A"] = ag.radius_of_gyration()
        rows.append(row)
    return pd.DataFrame(rows)


def run_rmsd(universe, solute_sel, start, stop, step):
    """Heavy-atom RMSD after optimal superposition on the first analysed frame."""
    from MDAnalysis.analysis import rms
    sel = f"({solute_sel}) and not name H*"
    r = rms.RMSD(universe, universe, select=sel, ref_frame=start)
    r.run(start=start, stop=stop, step=step)
    a = np.asarray(r.results.rmsd)
    return pd.DataFrame({"frame": a[:, 0].astype(int), "time_ps": a[:, 1],
                         "RMSD_A": a[:, 2]})


# --------------------------------------------------------------------------
# Free volume and pore size
# --------------------------------------------------------------------------

MAX_GRID_POINTS = 8_000_000


def free_volume(coords, radii, box=None, spacing=1.0, probe=1.4, chunk=200_000):
    """Grid-based free volume and pore-size distribution of the solute.

    For each grid point the "clearance" is the radius of the largest empty
    sphere centred there: distance to the nearest atom centre minus that
    atom's van der Waals radius. Clearance > 0 means the point is outside
    every atom (free volume). Clearance > probe means a probe of that radius
    (1.4 Å = water) can sit there.

    With an orthorhombic box the grid fills the box and distances use periodic
    images. Without one, the grid fills the solute's bounding box.
    """
    pos = np.asarray(coords, dtype=float)
    radii = np.asarray(radii, dtype=float)
    periodic = (box is not None and np.all(np.asarray(box[:3]) > 0)
                and np.allclose(box[3:6], 90.0))
    if periodic:
        L = np.asarray(box[:3], dtype=float)
        pos = np.mod(pos, L)
        pos = np.where(pos >= L, pos - L, pos)
        origin, extent = np.zeros(3), L
        tree = cKDTree(pos, boxsize=L)
    else:
        origin = pos.min(0)
        extent = np.maximum(pos.max(0) - origin, spacing)
        tree = cKDTree(pos)

    n = np.maximum(np.round(extent / spacing).astype(int), 1)
    if np.prod(n) > MAX_GRID_POINTS:
        raise ValueError(f"Grid would have {int(np.prod(n)):,} points. "
                         "Increase the grid spacing.")
    axes = [origin[i] + (np.arange(n[i]) + 0.5) * extent[i] / n[i] for i in range(3)]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)

    k = min(16, len(pos))
    clearance = np.empty(len(grid))
    for s in range(0, len(grid), chunk):
        d, j = tree.query(grid[s:s + chunk], k=k)
        if k == 1:
            d, j = d[:, None], j[:, None]
        clearance[s:s + chunk] = (d - radii[j]).min(1)

    volume = float(np.prod(extent))
    free = clearance > 0
    pores = clearance[free]
    summary = {
        "region": "periodic box" if periodic else "solute bounding box",
        "volume_A3": volume,
        "grid_spacing_A": float(spacing),
        "grid_points": int(len(grid)),
        "free_volume_fraction": float(free.mean()),
        "occupied_volume_fraction": float(1 - free.mean()),
        "probe_radius_A": float(probe),
        "probe_accessible_fraction": float((clearance > probe).mean()),
        "mean_clearance_A": float(pores.mean()) if pores.size else 0.0,
        "max_clearance_A": float(pores.max()) if pores.size else 0.0,
    }
    top = max(np.ceil(pores.max() * 4) / 4 if pores.size else 1.0, 0.5)
    counts, edges = np.histogram(pores, bins=np.arange(0, top + 0.25, 0.25))
    hist = pd.DataFrame({
        "clearance_A": 0.5 * (edges[1:] + edges[:-1]),
        "volume_A3": counts * volume / len(grid),
        "fraction_of_free_volume": counts / max(pores.size, 1),
    })
    return summary, hist


def run_free_volume(universe, solute, frame, spacing, probe, use_box=True):
    universe.trajectory[frame]
    el = atomgroup_elements(solute)
    box = universe.dimensions if use_box else None
    summary, hist = free_volume(solute.positions, vdw_radii(el), box, spacing, probe)
    summary["frame"] = int(universe.trajectory.ts.frame)
    return summary, hist


# --------------------------------------------------------------------------
# Diffusion (MSD)
# --------------------------------------------------------------------------

def fit_diffusion(lag_ps, msd_A2, fit_range=(0.1, 0.5), dim=3):
    n = len(msd_A2)
    i0 = int(fit_range[0] * n)
    i1 = max(i0 + 3, int(fit_range[1] * n))
    if i1 > n:
        raise ValueError("Too few frames to fit a diffusion coefficient.")
    slope, intercept = np.polyfit(lag_ps[i0:i1], msd_A2[i0:i1], 1)
    d = slope / (2 * dim)                         # Å²/ps
    return {"D_A2_per_ps": d, "D_cm2_per_s": d * 1e-4,
            "fit_start_ps": float(lag_ps[i0]), "fit_end_ps": float(lag_ps[i1 - 1]),
            "slope": slope, "intercept": intercept}


def run_msd(universe, sel, start, stop, step, fit_range=(0.1, 0.5)):
    """Einstein MSD. Coordinates must be unwrapped (no jumps across the box)."""
    from MDAnalysis.analysis.msd import EinsteinMSD
    try:
        import tidynamics  # noqa: F401
        fft = True
    except ImportError:
        fft = False
    m = EinsteinMSD(universe, select=sel, msd_type="xyz", fft=fft)
    m.run(start=start, stop=stop, step=step)
    msd = np.asarray(m.results.timeseries)
    lag = np.arange(len(msd)) * universe.trajectory.dt * max(step, 1)
    info = fit_diffusion(lag, msd, fit_range)
    info["n_atoms"] = len(universe.select_atoms(sel))
    df = pd.DataFrame({"lag_ps": lag, "MSD_A2": msd})
    return df, info


# --------------------------------------------------------------------------
# Radial distribution function
# --------------------------------------------------------------------------

def rdf_summary(r, g, rho):
    dr = r[1] - r[0]
    n_r = np.cumsum(4 * np.pi * rho * g * r ** 2 * dr)
    peak = int(np.argmax(g))
    gs = np.convolve(g, np.ones(3) / 3, mode="same")
    first_min = None
    for i in range(peak + 1, len(gs) - 1):
        if gs[i] <= gs[i - 1] and gs[i] < gs[i + 1]:
            first_min = i
            break
    out = {"first_peak_r_A": float(r[peak]), "first_peak_g": float(g[peak])}
    if first_min is not None:
        out.update({"first_min_r_A": float(r[first_min]),
                    "coordination_number": float(n_r[first_min])})
    return n_r, out


def run_rdf(universe, sel1, sel2, start, stop, step, rmax=10.0, nbins=100):
    from MDAnalysis.analysis.rdf import InterRDF
    if universe.dimensions is None:
        raise ValueError("RDF needs box dimensions (CRYST1 line in the PDB).")
    g1, g2 = universe.select_atoms(sel1), universe.select_atoms(sel2)
    if not len(g1) or not len(g2):
        raise ValueError("One of the RDF selections is empty.")
    if np.intersect1d(g1.indices, g2.indices).size:
        raise ValueError("The two RDF selections overlap; pick distinct groups.")
    rdf = InterRDF(g1, g2, nbins=nbins, range=(0.0, rmax))
    rdf.run(start=start, stop=stop, step=step)
    r, g = np.asarray(rdf.results.bins), np.asarray(rdf.results.rdf)
    vols = [ts.volume for ts in universe.trajectory[start:stop:step]]
    rho = len(g2) / float(np.mean(vols))
    n_r, summary = rdf_summary(r, g, rho)
    summary.update({"n_group1": len(g1), "n_group2": len(g2)})
    return pd.DataFrame({"r_A": r, "g_r": g, "n_r": n_r}), summary


# --------------------------------------------------------------------------
# Backbone dihedrals
# --------------------------------------------------------------------------

PPII_BOX = {"phi": (-110.0, -40.0), "psi": (100.0, 180.0)}


def ppii_fraction(phi, psi):
    phi, psi = np.asarray(phi), np.asarray(psi)
    inside = ((phi >= PPII_BOX["phi"][0]) & (phi <= PPII_BOX["phi"][1])
              & (psi >= PPII_BOX["psi"][0]) & (psi <= PPII_BOX["psi"][1]))
    return float(inside.mean()) if inside.size else np.nan


def run_ramachandran(universe, sel, start, stop, step):
    from MDAnalysis.analysis.dihedrals import Ramachandran
    ag = universe.select_atoms(sel)
    if not len(ag):
        raise ValueError("The protein selection is empty.")
    rama = Ramachandran(ag, check_protein=False)
    rama.run(start=start, stop=stop, step=step)
    ang = np.asarray(rama.results.angles)        # (frames, residues, 2)
    frames = np.asarray(rama.frames)
    rows = []
    for fi, fr in enumerate(frames):
        for ri in range(ang.shape[1]):
            rows.append((int(fr), ri, ang[fi, ri, 0], ang[fi, ri, 1]))
    df = pd.DataFrame(rows, columns=["frame", "residue_index", "phi_deg", "psi_deg"])
    per_frame = pd.DataFrame(
        [(fr, ppii_fraction(g.phi_deg, g.psi_deg)) for fr, g in df.groupby("frame")],
        columns=["frame", "PPII_fraction"])
    return df, per_frame


# --------------------------------------------------------------------------
# Release kinetics from experimental data
# --------------------------------------------------------------------------

RELEASE_MODELS = {
    "Zero-order": {
        "f": lambda t, k0: k0 * t,
        "params": ["k0"], "eq": "Q = k0·t",
        "p0": lambda t, q: [q.max() / max(t.max(), 1e-12)]},
    "First-order": {
        "f": lambda t, qmax, k1: qmax * (1 - np.exp(-k1 * t)),
        "params": ["Qmax", "k1"], "eq": "Q = Qmax·(1 − exp(−k1·t))",
        "p0": lambda t, q: [max(q.max(), 1.0), 1.0 / max(t.mean(), 1e-12)]},
    "Higuchi": {
        "f": lambda t, kh: kh * np.sqrt(t),
        "params": ["kH"], "eq": "Q = kH·√t",
        "p0": lambda t, q: [q.max() / max(np.sqrt(t.max()), 1e-12)]},
    "Korsmeyer–Peppas": {
        "f": lambda t, k, n: k * np.power(t, n),
        "params": ["kKP", "n"], "eq": "Q = kKP·tⁿ  (fit to Q ≤ 60 %)",
        "p0": lambda t, q: [q.max() / max(np.sqrt(t.max()), 1e-12), 0.5]},
}


def _peppas_mechanism(n):
    # Thresholds for a thin film / slab (Ritger & Peppas 1987).
    if n <= 0.5:
        return "Fickian diffusion"
    if n < 1.0:
        return "Anomalous (diffusion + swelling/erosion)"
    return "Case II (relaxation/erosion-controlled)"


def fit_release(t, q):
    """Fit cumulative release (% of loaded drug) against time."""
    t, q = np.asarray(t, float), np.asarray(q, float)
    ok = np.isfinite(t) & np.isfinite(q)
    t, q = t[ok], q[ok]
    rows, fits = [], {}
    for name, m in RELEASE_MODELS.items():
        mask = np.ones_like(t, bool)
        if name == "Korsmeyer–Peppas":
            mask = (t > 0) & (q > 0) & (q <= 60)
        tt, qq = t[mask], q[mask]
        npar = len(m["params"])
        row = {"model": name, "equation": m["eq"], "n_points": int(len(tt))}
        if len(tt) < npar + 2:
            row["note"] = "too few points"
            rows.append(row)
            continue
        try:
            popt, _ = curve_fit(m["f"], tt, qq, p0=m["p0"](tt, qq), maxfev=20000)
        except Exception as e:                    # noqa: BLE001
            row["note"] = f"fit failed: {e}"
            rows.append(row)
            continue
        qhat = m["f"](tt, *popt)
        rss = float(((qq - qhat) ** 2).sum())
        tss = float(((qq - qq.mean()) ** 2).sum())
        row["R2"] = 1 - rss / tss if tss > 0 else np.nan
        row["RMSE"] = float(np.sqrt(rss / len(tt)))
        row.update(dict(zip(m["params"], popt)))
        if name == "Korsmeyer–Peppas":
            row["mechanism"] = _peppas_mechanism(popt[1])
        rows.append(row)
        fits[name] = popt
    return pd.DataFrame(rows), fits
