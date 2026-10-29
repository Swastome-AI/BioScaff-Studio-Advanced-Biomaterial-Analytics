"""
BioScaff Studio - Streamlit front end.

Computation lives in analysis.py, figures in figures.py.
Run locally:  streamlit run app.py
"""
import datetime as dt
import io
import json
import os
import re
import tempfile
import zipfile

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import streamlit as st
import MDAnalysis as mda

from core import md as an
from core import md_figures as F

st.set_page_config(page_title="MD trajectory analysis", page_icon="🧬", layout="wide")

WATER_DEFAULT = "resname HOH WAT SOL TIP3 TIP3P TIP4 TIP4P SPC SPCE T3P"
IONS_DEFAULT = "resname NA CL K SOD CLA POT MG CAL ZN"
FIG_SIZES = {
    "Single column (3.5 in)": (3.5, 2.8),
    "1.5 column (5 in)": (5.0, 3.6),
    "Double column (7 in)": (7.0, 4.2),
    "Poster panel (10 in)": (10.0, 6.0),
}

ss = st.session_state
ss.setdefault("md_results", {})   # analysis name -> result record
ss.setdefault("md_figcache", {})  # figure name -> {"key", "files", "preview"}
ss.setdefault("md_tmpdir", tempfile.mkdtemp(prefix="bioscaff_md_"))


# ==========================================================================
# Sidebar
# ==========================================================================
with st.sidebar:
    st.title("🧬 MD trajectory analysis")
    st.caption("Advanced: for systems simulated elsewhere (GPU).")

    with st.expander("Project & labels"):
        PROJECT = st.text_input("Project name", "MD trajectory analysis")
        SOLUTE_NAME = st.text_input(
            "Word for the solute", "Scaffold",
            help="Used in axis labels: e.g. Scaffold, Hydrogel, Polymer, Protein.")

    st.subheader("1 · Systems")
    n_sys = st.radio("Systems to compare", [1, 2], index=1, horizontal=True,
                     help="Upload two systems to compare them on every analysis.")
    uploads = []
    for i in range(n_sys):
        with st.expander(f"System {i + 1}", expanded=True):
            label = st.text_input("Label", f"System {'AB'[i]}", key=f"label{i}")
            top = st.file_uploader("Structure (.pdb or .gro)", type=["pdb", "gro"],
                                   key=f"top{i}")
            trj = st.file_uploader("Trajectory, optional (.xtc, .dcd, .trr)",
                                   type=["xtc", "dcd", "trr"], key=f"trj{i}")
            uploads.append((label.strip() or f"System {i + 1}", top, trj))

    st.subheader("2 · Atom selections")
    WATER_SEL = st.text_input("Water", WATER_DEFAULT)
    SOLUTE_SEL = st.text_input("Solute", f"not ({WATER_DEFAULT}) and not ({IONS_DEFAULT})",
                               help="MDAnalysis selection language.")
    COMP_TEXT = st.text_area(
        "Components (optional), one per line: Label = selection",
        placeholder="Collagen = protein\nPCL = resname PCL\nChitosan = resname <names>\n"
                    "Hyaluronic acid = resname <names>",
        help="Use the residue names listed under System overview. Components give "
             "per-component SASA, Rg and an H-bond contact matrix.")

    st.subheader("3 · Trajectory frames")
    c1, c2, c3 = st.columns(3)
    START = c1.number_input("Start", 0, value=0, step=1)
    STOP = c2.number_input("Stop", 0, value=0, step=1, help="0 = last frame")
    STEP = c3.number_input("Step", 1, value=1, step=1)

    st.subheader("4 · Figures")
    size_name = st.selectbox("Size", list(FIG_SIZES), index=1)
    DPI = st.select_slider("PNG resolution (dpi)", [300, 600, 1200], value=600)
    FONT = st.slider("Font size (pt)", 6, 16, 9)

FIG_W, FIG_H = FIG_SIZES[size_name]
F.configure(FIG_W, FIG_H, FONT, SOLUTE_NAME)
SLUG = re.sub(r"[^A-Za-z0-9]+", "_", PROJECT).strip("_") or "bioscaff"


# ==========================================================================
# Loading systems
# ==========================================================================
def _file_id(up):
    return getattr(up, "file_id", None) or f"{up.name}_{up.size}"


def save_upload(up, sub):
    d = os.path.join(ss.md_tmpdir, re.sub(r"[^A-Za-z0-9_.-]", "_", sub))
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, os.path.basename(up.name))
    if not os.path.exists(path):
        with open(path, "wb") as fh:
            fh.write(up.getbuffer())
    return path


@st.cache_resource(show_spinner="Loading system…", max_entries=6)
def load_universe(top_path, trj_path):
    return mda.Universe(top_path, trj_path) if trj_path else mda.Universe(top_path)


try:
    COMP_DEFS = an.parse_components(COMP_TEXT)
    comp_error = None
except ValueError as e:
    COMP_DEFS, comp_error = {}, str(e)

systems = []
for i, (label, top, trj) in enumerate(uploads):
    if top is None:
        continue
    if any(s["label"] == label for s in systems):
        label = f"{label} ({i + 1})"
    try:
        tp = save_upload(top, f"sys{i}_{_file_id(top)}")
        jp = save_upload(trj, f"sys{i}_{_file_id(trj)}") if trj else None
        u = load_universe(tp, jp)
    except Exception as e:  # noqa: BLE001
        st.sidebar.error(f"{label}: could not load files. {e}")
        continue
    systems.append({"label": label, "u": u,
                    "files": [top.name] + ([trj.name] if trj else [])})

if comp_error:
    st.sidebar.error(comp_error)


def frame_range(u):
    n = u.trajectory.n_frames
    stop = n if STOP == 0 else min(int(STOP), n)
    start = min(int(START), max(stop - 1, 0))
    return start, stop, int(STEP)


def components_for(u):
    out = {}
    for name, sel in COMP_DEFS.items():
        ag = u.select_atoms(sel)
        if len(ag):
            out[name] = ag
    return out


def any_trajectory():
    return any(s["u"].trajectory.n_frames > 1 for s in systems)


# ==========================================================================
# Running, storing and showing results
# ==========================================================================
def run_analysis(name, fn, params, need_traj=False):
    """Run fn(system) for every loaded system; store results in session state."""
    out, errors = {}, []
    bar = st.progress(0.0, text=f"Running {name}…")
    for i, s in enumerate(systems):
        if need_traj and s["u"].trajectory.n_frames < 2:
            errors.append(f"{s['label']}: needs a trajectory file.")
            continue
        try:
            out[s["label"]] = fn(s)
        except Exception as e:  # noqa: BLE001
            errors.append(f"{s['label']}: {e}")
        bar.progress((i + 1) / len(systems), text=f"Running {name}… {s['label']} done")
    bar.empty()
    ss.md_results[name] = {"run_id": dt.datetime.now().isoformat(timespec="seconds"),
                        "params": params, "systems": out, "errors": errors,
                        "selections": {"solute": SOLUTE_SEL, "water": WATER_SEL,
                                       "components": COMP_DEFS}}


def show_errors(name):
    for e in ss.md_results.get(name, {}).get("errors", []):
        st.error(e)


def figure_block(name, builder, version):
    """Render once per settings change; offer PNG/SVG/PDF downloads."""
    key = (version, FIG_W, FIG_H, DPI, FONT, SOLUTE_NAME)
    cached = ss.md_figcache.get(name)
    if not cached or cached["key"] != key:
        F.configure(FIG_W, FIG_H, FONT, SOLUTE_NAME)
        fig = builder()
        if fig is None:
            return
        files = {}
        for fmt in ("png", "svg", "pdf"):
            buf = io.BytesIO()
            fig.savefig(buf, format=fmt, dpi=DPI)
            files[fmt] = buf.getvalue()
        prev = io.BytesIO()
        fig.savefig(prev, format="png", dpi=200)
        width_px = int(fig.get_figwidth() * 110)
        plt.close(fig)
        cached = {"key": key, "files": files, "preview": prev.getvalue(), "width": width_px}
        ss.md_figcache[name] = cached
    st.image(cached["preview"], width=min(cached["width"], 1100))
    mime = {"png": "image/png", "svg": "image/svg+xml", "pdf": "application/pdf"}
    cols = st.columns([1, 1, 1, 3])
    for col, fmt in zip(cols, ("png", "svg", "pdf")):
        col.download_button(
            f"⬇ {fmt.upper()}" + (f" · {DPI} dpi" if fmt == "png" else " · vector"),
            cached["files"][fmt], file_name=f"{SLUG}_{name}.{fmt}", mime=mime[fmt],
            key=f"dl_{name}_{fmt}")


def table_block(df, name):
    st.dataframe(df, hide_index=True)
    st.download_button("⬇ CSV", df.to_csv(index=False).encode(), f"{SLUG}_{name}.csv",
                       "text/csv", key=f"csv_{name}")


def needs_systems():
    if not systems:
        st.info("Upload a structure file in the sidebar to begin.")
        return True
    return False


def single_frame_note():
    if systems and not any_trajectory():
        st.caption("Single structure loaded: results describe one snapshot and have no "
                   "error bars. Upload a trajectory to average over time.")


def summarise(name, rows_fn):
    res = ss.md_results[name]
    rows = [{"System": l, **rows_fn(d)} for l, d in res["systems"].items()]
    return pd.DataFrame(rows)


def ms(series, digits=2):
    s = np.asarray(series, float)
    if len(s) > 1:
        return f"{s.mean():.{digits}f} ± {s.std(ddof=1):.{digits}f}"
    return f"{s.mean():.{digits}f}"


# ==========================================================================
# Page
# ==========================================================================
st.title(PROJECT)
st.caption("Structural and dynamic analysis of solvated biomaterial systems from PDB/GRO "
           "structures and MD trajectories. Every value is computed from your coordinates.")

tabs = st.tabs(["📊 Overview", "💧 SASA", "🔗 H-bonds", "🕳️ Free volume",
                "📏 Size & stability", "🚶 Diffusion (MSD)", "📈 RDF",
                "🧬 Backbone (Ramachandran)", "💊 Release kinetics", "📥 Export"])

# ---------------------------------------------------------------- Overview
with tabs[0]:
    st.header("System overview")
    if not needs_systems():
        comp_rows = []
        for s in systems:
            u = s["u"]
            st.subheader(s["label"])
            try:
                sol = u.select_atoms(SOLUTE_SEL)
                wat = u.select_atoms(WATER_SEL)
                comps = components_for(u)
            except Exception as e:  # noqa: BLE001
                st.error(f"Selection error: {e}")
                continue
            box = u.dimensions
            c = st.columns(5)
            c[0].metric("Total atoms", f"{u.atoms.n_atoms:,}")
            c[1].metric(f"{SOLUTE_NAME} atoms", f"{len(sol):,}")
            c[2].metric("Water molecules", f"{wat.n_residues:,}")
            c[3].metric("Frames", f"{u.trajectory.n_frames:,}")
            c[4].metric("Box (nm)", "—" if box is None else
                        " × ".join(f"{v / 10:.1f}" for v in box[:3]))
            if u.trajectory.n_frames > 1:
                st.caption(f"Frame interval {u.trajectory.dt:g} ps · total "
                           f"{u.trajectory.totaltime / 1000:g} ns")
            el = an.atomgroup_elements(sol) if len(sol) else np.array([])
            if len(sol) and not (el == "H").any():
                st.warning("The solute has no hydrogens. H-bond analysis needs explicit H.")
            if len(sol) == 0:
                st.warning("The solute selection matched no atoms. Check the residue names below.")
            with st.expander("Residue names in this file (use these in selections)"):
                st.dataframe(an.residue_table(u.atoms), hide_index=True)
            counts, assigned = {}, np.zeros(u.atoms.n_atoms, bool)
            for k, ag in comps.items():
                counts[k] = len(ag)
                assigned[ag.indices] = True
            rest = np.setdiff1d(sol.indices, np.where(assigned)[0])
            if len(rest):
                counts[f"Other {SOLUTE_NAME.lower()}" if comps else SOLUTE_NAME] = len(rest)
            counts["Water"] = len(wat)
            other = u.atoms.n_atoms - len(sol) - len(wat)
            if other > 0:
                counts["Ions / other"] = other
            comp_rows.append((s["label"], counts))
        if comp_rows:
            st.subheader("Composition")
            figure_block("composition", lambda: F.composition(comp_rows),
                         json.dumps(comp_rows, default=str))

# ---------------------------------------------------------------- SASA
with tabs[1]:
    st.header("Solvent-accessible surface area")
    st.markdown(
        "Shrake–Rupley SASA of the solute with Bondi radii, after removing water and ions. "
        "The polar share (N, O and their hydrogens) indicates how hydrophilic the exposed "
        "surface is. Compare systems by polar fraction or per-atom SASA, since total SASA "
        "grows with system size.")
    if not needs_systems():
        single_frame_note()
        c = st.columns(3)
        probe = c[0].number_input("Probe radius (Å)", 0.5, 3.0, 1.4, 0.1)
        npts = c[1].number_input("Points per atom", 30, 500, 100, 10)
        maxf = c[2].number_input("Max frames", 1, 1000, 25,
                                 help="SASA is slow; frames are spread evenly over the range.")
        if st.button("▶ Run SASA", type="primary"):
            def job(s):
                u = s["u"]
                a, b, c_ = frame_range(u)
                fr = an.frame_indices(u.trajectory.n_frames, a, b, c_, int(maxf))
                return {"table": an.run_sasa(u, u.select_atoms(SOLUTE_SEL),
                                             components_for(u), fr, probe, int(npts))}
            run_analysis("SASA", job, {"probe_A": probe, "points_per_atom": int(npts),
                                       "max_frames": int(maxf)})
        if "SASA" in ss.md_results:
            show_errors("SASA")
            res = ss.md_results["SASA"]
            sysres = res["systems"]
            if sysres:
                st.dataframe(summarise("SASA", lambda d: {
                    "Total (nm²)": ms(d["table"]["total_A2"] / 100),
                    "Polar (nm²)": ms(d["table"]["polar_A2"] / 100),
                    "Nonpolar (nm²)": ms(d["table"]["nonpolar_A2"] / 100),
                    "Polar fraction": ms(d["table"]["polar_fraction"], 3),
                    "Per heavy atom (Å²)": ms(d["table"]["total_per_heavy_atom_A2"]),
                    "Frames": len(d["table"])}), hide_index=True)
                figure_block("sasa_polar_nonpolar", lambda: F.sasa_bars(sysres), res["run_id"])
                if any(len(d["table"]) > 1 for d in sysres.values()):
                    figure_block("sasa_timeseries", lambda: F.timeseries(
                        sysres, "total_A2", "Total SASA (nm²)", 0.01), res["run_id"])
                if COMP_DEFS:
                    figure_block("sasa_components", lambda: F.component_bars(
                        sysres, "", "_A2", "SASA (nm²)", 0.01), res["run_id"])

# ---------------------------------------------------------------- H-bonds
with tabs[2]:
    st.header("Hydrogen bonds")
    st.markdown(
        "Counts actual hydrogen bonds from geometry (donor–acceptor distance and "
        "donor–H–acceptor angle). **Solute–water** bonds measure hydration; "
        "**solute–solute** bonds measure how strongly the components hold together. "
        "Requires explicit hydrogens.")
    if not needs_systems():
        single_frame_note()
        c = st.columns(3)
        dcut = c[0].number_input("Donor–acceptor cutoff (Å)", 2.5, 4.0, 3.0, 0.1)
        acut = c[1].number_input("Minimum D–H···A angle (°)", 90.0, 180.0, 150.0, 5.0)
        shell = c[2].number_input("Water search shell (Å)", 3.0, 8.0, 3.5, 0.5)
        if st.button("▶ Run H-bond analysis", type="primary"):
            def job(s):
                u = s["u"]
                a, b, c_ = frame_range(u)
                df, m = an.run_hbonds(u, SOLUTE_SEL, WATER_SEL, components_for(u),
                                      a, b, c_, dcut, acut, shell)
                return {"table": df, "matrix": m}
            run_analysis("H-bonds", job, {"d_a_cutoff_A": dcut, "angle_cutoff_deg": acut,
                                          "water_shell_A": shell})
        if "H-bonds" in ss.md_results:
            show_errors("H-bonds")
            res = ss.md_results["H-bonds"]
            sysres = res["systems"]
            if sysres:
                st.dataframe(summarise("H-bonds", lambda d: {
                    f"{SOLUTE_NAME}–water": ms(d["table"]["solute_water"], 1),
                    "…per polar atom": ms(d["table"]["solute_water_per_polar_atom"], 3),
                    f"{SOLUTE_NAME}–{SOLUTE_NAME.lower()}": ms(d["table"]["solute_solute"], 1),
                    "Frames": len(d["table"])}), hide_index=True)
                figure_block("hbonds_summary", lambda: F.hbond_bars(sysres), res["run_id"])
                if any(len(d["table"]) > 1 for d in sysres.values()):
                    figure_block("hbonds_timeseries", lambda: F.timeseries(
                        sysres, "solute_water", f"{SOLUTE_NAME}–water H-bonds"), res["run_id"])
                if any(d.get("matrix") is not None for d in sysres.values()):
                    st.subheader("Hydrogen bonds between components")
                    st.caption("Mean H-bonds per frame. Off-diagonal cells show how much "
                               "components bond to each other, e.g. PCL to the biopolymers.")
                    figure_block("hbonds_component_matrix",
                                 lambda: F.hbond_matrix(sysres), res["run_id"])

# ---------------------------------------------------------------- Free volume
with tabs[3]:
    st.header("Free volume and pore size")
    st.markdown(
        "A grid is laid over the box. At each point the *clearance* is the radius of the "
        "largest empty sphere that fits there without touching a solute atom. This gives "
        "the free-volume fraction and a distribution of local pore radii.")
    st.info(
        "**How to read this.** These are molecular-scale voids (Å to nm), not the "
        "10–300 µm pores of a real scaffold, and a bacterium (~1 µm) is about a thousand "
        "times larger than a typical simulation box. Use the result to compare how densely "
        "the polymer chains pack, not as a direct prediction of bacterial penetration. In a "
        "dilute solvated box the free volume mostly reflects polymer concentration; it is "
        "most meaningful for densely packed or dried systems.")
    if not needs_systems():
        c = st.columns(4)
        spacing = c[0].number_input("Grid spacing (Å)", 0.25, 3.0, 1.0, 0.25)
        probe_fv = c[1].number_input("Probe radius (Å)", 0.5, 5.0, 1.4, 0.1)
        frame_fv = c[2].number_input("Frame (−1 = last)", -1, value=-1, step=1)
        use_box = c[3].toggle("Use periodic box", True,
                              help="Off: grid covers only the solute's bounding box.")
        if st.button("▶ Run free-volume analysis", type="primary"):
            def job(s):
                u = s["u"]
                fr = u.trajectory.n_frames - 1 if frame_fv < 0 else min(int(frame_fv),
                                                                        u.trajectory.n_frames - 1)
                summ, hist = an.run_free_volume(u, u.select_atoms(SOLUTE_SEL), fr,
                                                spacing, probe_fv, use_box)
                return {"summary": summ, "hist": hist}
            run_analysis("Free volume", job, {"grid_spacing_A": spacing,
                                              "probe_A": probe_fv, "frame": int(frame_fv),
                                              "periodic_box": bool(use_box)})
        if "Free volume" in ss.md_results:
            show_errors("Free volume")
            res = ss.md_results["Free volume"]
            sysres = res["systems"]
            if sysres:
                st.dataframe(summarise("Free volume", lambda d: {
                    "Region": d["summary"]["region"],
                    "Volume (nm³)": f"{d['summary']['volume_A3'] / 1000:.1f}",
                    "Free volume (%)": f"{d['summary']['free_volume_fraction'] * 100:.1f}",
                    "Probe-accessible (%)": f"{d['summary']['probe_accessible_fraction'] * 100:.1f}",
                    "Mean clearance (Å)": f"{d['summary']['mean_clearance_A']:.2f}",
                    "Max clearance (Å)": f"{d['summary']['max_clearance_A']:.2f}"}),
                    hide_index=True)
                figure_block("free_volume", lambda: F.free_volume(sysres), res["run_id"])

# ---------------------------------------------------------------- Rg / RMSD
with tabs[4]:
    st.header("Size and structural stability")
    st.markdown(
        "**Radius of gyration** measures compactness. **RMSD** (heavy atoms, after "
        "superposition on the first frame) measures drift from the starting structure; a "
        "plateau suggests the system has equilibrated. Trajectories should be made whole "
        "(e.g. `gmx trjconv -pbc mol`) so molecules are not split across the box edge.")
    if not needs_systems():
        single_frame_note()
        if st.button("▶ Run size & stability", type="primary"):
            def job(s):
                u = s["u"]
                a, b, c_ = frame_range(u)
                fr = an.frame_indices(u.trajectory.n_frames, a, b, c_)
                out = {"table": an.run_rg(u, u.select_atoms(SOLUTE_SEL),
                                          components_for(u), fr)}
                if u.trajectory.n_frames > 1:
                    out["rmsd"] = an.run_rmsd(u, SOLUTE_SEL, a, b, c_)
                return out
            run_analysis("Size & stability", job, {})
        if "Size & stability" in ss.md_results:
            show_errors("Size & stability")
            res = ss.md_results["Size & stability"]
            sysres = res["systems"]
            if sysres:
                st.dataframe(summarise("Size & stability", lambda d: {
                    "Rg (nm)": ms(d["table"]["Rg_solute_A"] / 10, 3),
                    "Asphericity": ms(d["table"]["asphericity"], 3),
                    "Final RMSD (Å)": (f"{d['rmsd']['RMSD_A'].iloc[-1]:.2f}"
                                       if "rmsd" in d else "—"),
                    "Frames": len(d["table"])}), hide_index=True)
                multi = any(len(d["table"]) > 1 for d in sysres.values())
                if multi:
                    figure_block("rg_timeseries", lambda: F.timeseries(
                        sysres, "Rg_solute_A", "Radius of gyration (nm)", 0.1), res["run_id"])
                else:
                    figure_block("rg", lambda: F.rg_bars(sysres), res["run_id"])
                rmsd_sys = {l: {"table": d["rmsd"]} for l, d in sysres.items() if "rmsd" in d}
                if rmsd_sys:
                    figure_block("rmsd_timeseries", lambda: F.timeseries(
                        rmsd_sys, "RMSD_A", "Heavy-atom RMSD (Å)"), res["run_id"])
                if COMP_DEFS:
                    figure_block("rg_components", lambda: F.component_bars(
                        sysres, "Rg_", "_A", "Radius of gyration (nm)", 0.1), res["run_id"])

# ---------------------------------------------------------------- MSD
with tabs[5]:
    st.header("Diffusion: mean squared displacement")
    st.markdown(
        "MSD of a chosen group (e.g. a drug molecule, or water oxygens) over lag time. The "
        "slope of the linear region gives the self-diffusion coefficient, D = slope / 6. "
        "Slower diffusion of a drug or water inside one scaffold than another is a "
        "molecular-level indicator of more hindered transport. Needs a trajectory with "
        "**unwrapped** coordinates (`gmx trjconv -pbc nojump`).")
    st.caption("For reference, bulk water diffuses at about 2.3 × 10⁻⁵ cm²/s at 25 °C; "
               "the TIP3P model overestimates this by roughly 2×.")
    if not needs_systems():
        msd_sel = st.text_input("Atoms to track", f"({WATER_SEL}) and name O*",
                                help="Example for a drug: resname DRG")
        fit = st.slider("Linear fit window (fraction of lag times)", 0.0, 1.0, (0.1, 0.5), 0.05)
        if st.button("▶ Run MSD", type="primary", disabled=not any_trajectory()):
            def job(s):
                u = s["u"]
                a, b, c_ = frame_range(u)
                df, info = an.run_msd(u, msd_sel, a, b, c_, fit)
                return {"table": df, "info": info}
            run_analysis("MSD", job, {"selection": msd_sel, "fit_window": list(fit)},
                         need_traj=True)
        if not any_trajectory():
            st.caption("Upload a trajectory to enable this analysis.")
        if "MSD" in ss.md_results:
            show_errors("MSD")
            res = ss.md_results["MSD"]
            sysres = res["systems"]
            if sysres:
                st.dataframe(summarise("MSD", lambda d: {
                    "D (cm²/s)": f"{d['info']['D_cm2_per_s']:.3e}",
                    "Fit window (ps)": f"{d['info']['fit_start_ps']:.0f}–{d['info']['fit_end_ps']:.0f}",
                    "Atoms": d["info"]["n_atoms"]}), hide_index=True)
                figure_block("msd", lambda: F.msd(sysres), res["run_id"])

# ---------------------------------------------------------------- RDF
with tabs[6]:
    st.header("Radial distribution function")
    st.markdown(
        "g(r) of group 2 around group 1, e.g. water oxygens around chitosan amine "
        "nitrogens. The first peak locates the first hydration shell; the cumulative "
        "coordination n(r) at the first minimum is the number of neighbours in that shell.")
    if not needs_systems():
        single_frame_note()
        c = st.columns(2)
        rdf1 = c[0].text_input("Group 1 (centre)", f"({SOLUTE_SEL}) and name N*")
        rdf2 = c[1].text_input("Group 2 (neighbours)", f"({WATER_SEL}) and name O*")
        c = st.columns(2)
        rmax = c[0].number_input("Max distance (Å)", 4.0, 20.0, 10.0, 1.0)
        nbins = c[1].number_input("Bins", 20, 400, 100, 10)
        if st.button("▶ Run RDF", type="primary"):
            def job(s):
                u = s["u"]
                a, b, c_ = frame_range(u)
                df, summ = an.run_rdf(u, rdf1, rdf2, a, b, c_, rmax, int(nbins))
                return {"table": df, "summary": summ}
            run_analysis("RDF", job, {"group1": rdf1, "group2": rdf2, "rmax_A": rmax,
                                      "bins": int(nbins)})
        if "RDF" in ss.md_results:
            show_errors("RDF")
            res = ss.md_results["RDF"]
            sysres = res["systems"]
            if sysres:
                st.dataframe(summarise("RDF", lambda d: {
                    "First peak r (Å)": f"{d['summary']['first_peak_r_A']:.2f}",
                    "g(peak)": f"{d['summary']['first_peak_g']:.2f}",
                    "First minimum r (Å)": f"{d['summary'].get('first_min_r_A', np.nan):.2f}",
                    "Coordination number": f"{d['summary'].get('coordination_number', np.nan):.2f}",
                    "Group 1 atoms": d["summary"]["n_group1"],
                    "Group 2 atoms": d["summary"]["n_group2"]}), hide_index=True)
                figure_block("rdf", lambda: F.rdf(sysres), res["run_id"])

# ---------------------------------------------------------------- Ramachandran
with tabs[7]:
    st.header("Backbone conformation (Ramachandran)")
    st.markdown(
        "φ/ψ backbone dihedrals of a protein component. Collagen chains sit in the "
        "polyproline-II (PPII) region (φ ≈ −75°, ψ ≈ +145°). The share of residues "
        "staying in the dashed PPII box over a trajectory indicates whether the collagen "
        "keeps its native conformation inside the scaffold. A single structure only shows "
        "the starting model.")
    if not needs_systems():
        single_frame_note()
        rama_sel = st.text_input("Protein selection", "protein",
                                 help="Must contain backbone atoms named N, CA, C.")
        if st.button("▶ Run Ramachandran", type="primary"):
            def job(s):
                u = s["u"]
                a, b, c_ = frame_range(u)
                df, per = an.run_ramachandran(u, rama_sel, a, b, c_)
                return {"table": df, "per_frame": per}
            run_analysis("Ramachandran", job, {"selection": rama_sel,
                                               "PPII_box": an.PPII_BOX})
        if "Ramachandran" in ss.md_results:
            show_errors("Ramachandran")
            res = ss.md_results["Ramachandran"]
            sysres = res["systems"]
            if sysres:
                st.dataframe(summarise("Ramachandran", lambda d: {
                    "Residues": d["table"]["residue_index"].nunique(),
                    "Frames": d["per_frame"].shape[0],
                    "PPII fraction": ms(d["per_frame"]["PPII_fraction"], 3)}),
                    hide_index=True)
                figure_block("ramachandran", lambda: F.ramachandran(sysres), res["run_id"])

# ---------------------------------------------------------------- Release
with tabs[8]:
    st.header("Release kinetics (experimental data)")
    st.markdown(
        "Release over hours to days is far beyond what MD can simulate, so this tab fits "
        "**your measured release data** (e.g. UV-Vis of the release medium) to standard "
        "models. Upload a CSV with a time column, a cumulative % released column, and "
        "optionally a group column (scaffold) and an SD column.")
    template = pd.DataFrame({
        "scaffold": ["A"] * 3 + ["B"] * 3,
        "time_h": [1, 4, 24] * 2,
        "released_pct": [np.nan] * 6,
        "sd_pct": [np.nan] * 6})
    st.download_button("⬇ Blank CSV template", template.to_csv(index=False).encode(),
                       "release_template.csv", "text/csv")
    rel_file = st.file_uploader("Release data (.csv)", type=["csv"], key="release_csv")
    if rel_file is not None:
        try:
            raw = pd.read_csv(rel_file)
        except Exception as e:  # noqa: BLE001
            st.error(f"Could not read CSV: {e}")
            raw = None
        if raw is not None and len(raw.columns) >= 2:
            cols = list(raw.columns)
            num = [c for c in cols if pd.api.types.is_numeric_dtype(raw[c])]
            c = st.columns(4)
            tcol = c[0].selectbox("Time column", num or cols, index=0)
            qcol = c[1].selectbox("% released column", num or cols,
                                  index=min(1, len(num or cols) - 1))
            gcol = c[2].selectbox("Group column", ["(none)"] + cols)
            sdcol = c[3].selectbox("SD column", ["(none)"] + num)
            tlabel = st.text_input("Time axis label", f"Time ({tcol})")
            groups, tables = {}, []
            grouped = [("All", raw)] if gcol == "(none)" else list(raw.groupby(gcol))
            for g, d in grouped:
                d = d.dropna(subset=[tcol, qcol]).sort_values(tcol)
                tab, fits = an.fit_release(d[tcol], d[qcol])
                tab.insert(0, "group", str(g))
                tables.append(tab)
                groups[str(g)] = {"t": d[tcol].to_numpy(float), "q": d[qcol].to_numpy(float),
                                  "sd": d[sdcol].to_numpy(float) if sdcol != "(none)" else None,
                                  "fits": fits}
            fit_table = pd.concat(tables, ignore_index=True)
            st.dataframe(fit_table.round(4), hide_index=True)
            st.caption("Korsmeyer–Peppas is fitted only to points ≤ 60 % release, so its "
                       "R² is not directly comparable with the other models. The mechanism "
                       "label uses thin-film thresholds (n ≤ 0.5 Fickian, n ≥ 1 case II).")
            model = st.selectbox("Model to plot", list(an.RELEASE_MODELS), index=3)
            version = f"{_file_id(rel_file)}|{tcol}|{qcol}|{gcol}|{sdcol}|{model}|{tlabel}"
            figure_block("release_kinetics", lambda: F.release(groups, model, tlabel), version)
            ss.md_results["Release kinetics"] = {
                "run_id": version, "params": {"file": rel_file.name, "time": tcol,
                                              "release": qcol, "group": gcol},
                "systems": {"fits": {"table": fit_table}}, "errors": []}

# ---------------------------------------------------------------- Export
METHOD_NOTES = {
    "SASA": "Shrake–Rupley solvent-accessible surface area of solute atoms (water and ions "
            "removed) with Bondi van der Waals radii; H atoms assigned polar/nonpolar class "
            "of nearest heavy atom; periodic images not considered.",
    "H-bonds": "Geometric hydrogen bonds with MDAnalysis HydrogenBondAnalysis; donors and "
               "acceptors N/O, donor–H pairs within 1.2 Å; water within the search shell of "
               "the solute considered.",
    "Free volume": "Grid-based clearance (distance to nearest atom centre minus its Bondi "
                   "radius) over the periodic box or solute bounding box.",
    "Size & stability": "Mass-weighted radius of gyration and asphericity per frame; "
                        "heavy-atom RMSD after optimal superposition (MDAnalysis RMSD).",
    "MSD": "Einstein MSD (MDAnalysis EinsteinMSD, xyz); D from linear fit, D = slope/6.",
    "RDF": "MDAnalysis InterRDF; coordination number from integrating 4πρg(r)r² to the "
           "first minimum.",
    "Ramachandran": "Backbone φ/ψ dihedrals (MDAnalysis Ramachandran); PPII region "
                    "φ −110° to −40°, ψ 100° to 180°.",
    "Release kinetics": "Non-linear least-squares fits (SciPy curve_fit) of zero-order, "
                        "first-order, Higuchi and Korsmeyer–Peppas models.",
}


def methods_text():
    lines = [f"{PROJECT} — analysis log", f"Generated {dt.datetime.now():%Y-%m-%d %H:%M}",
             f"Software: MDAnalysis {mda.__version__}, NumPy {np.__version__}, "
             f"SciPy {scipy.__version__}, pandas {pd.__version__}, "
             f"Matplotlib {matplotlib.__version__}, Streamlit {st.__version__}", "",
             "Systems:"]
    for s in systems:
        u = s["u"]
        lines.append(f"  {s['label']}: {', '.join(s['files'])}; {u.atoms.n_atoms} atoms; "
                     f"{u.trajectory.n_frames} frames; dt = {u.trajectory.dt:g} ps")
    lines += ["", f"Frames used: start {START}, stop {STOP or 'last'}, step {STEP}", ""]
    for name, res in ss.md_results.items():
        lines.append(f"[{name}]  run {res['run_id']}")
        lines.append(f"  Method: {METHOD_NOTES.get(name, '')}")
        if "selections" in res:
            lines.append(f"  Selections: {json.dumps(res['selections'])}")
        lines.append(f"  Parameters: {json.dumps(res['params'], default=str)}")
        for e in res.get("errors", []):
            lines.append(f"  Error: {e}")
        lines.append("")
    return "\n".join(lines)


def all_tables():
    out = {}
    for name, res in ss.md_results.items():
        base = re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")
        for label, d in res["systems"].items():
            lab = re.sub(r"[^A-Za-z0-9]+", "_", label).strip("_")
            for key in ("table", "hist", "rmsd", "per_frame"):
                if isinstance(d.get(key), pd.DataFrame):
                    suffix = "" if key == "table" else f"_{key}"
                    out[f"{base}_{lab}{suffix}.csv"] = d[key]
            if isinstance(d.get("matrix"), pd.DataFrame):
                out[f"{base}_{lab}_component_matrix.csv"] = d["matrix"].reset_index(
                    names="component")
            if isinstance(d.get("info"), dict):
                out[f"{base}_{lab}_fit.csv"] = pd.DataFrame([d["info"]])
            if isinstance(d.get("summary"), dict):
                out[f"{base}_{lab}_summary.csv"] = pd.DataFrame([d["summary"]])
    return out


with tabs[9]:
    st.header("Export")
    if not ss.md_results:
        st.info("Run an analysis first. Results, figures and a methods log collect here.")
    else:
        tables = all_tables()
        st.markdown(f"**{len(tables)} data tables** and **{len(ss.md_figcache)} figures** "
                    "(PNG, SVG, PDF each) ready.")
        zbuf = io.BytesIO()
        with zipfile.ZipFile(zbuf, "w", zipfile.ZIP_DEFLATED) as z:
            for fname, df in tables.items():
                z.writestr(f"data/{fname}", df.to_csv(index=False))
            for name, c in ss.md_figcache.items():
                for fmt, data in c["files"].items():
                    z.writestr(f"figures/{SLUG}_{name}.{fmt}", data)
            z.writestr("methods.txt", methods_text())
        st.download_button("⬇ Download everything (.zip)", zbuf.getvalue(),
                           f"{SLUG}_results.zip", "application/zip", type="primary")
        with st.expander("Methods log (paste into your notebook or paper)"):
            st.code(methods_text(), language=None)
        st.subheader("Individual tables")
        for fname, df in tables.items():
            with st.expander(fname):
                st.dataframe(df.head(200), hide_index=True)
                st.download_button("⬇ CSV", df.to_csv(index=False).encode(), fname,
                                   "text/csv", key=f"exp_{fname}")
