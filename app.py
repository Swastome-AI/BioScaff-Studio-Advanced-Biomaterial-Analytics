"""
BioScaff Studio — scaffold screening pipeline.

Molecule library → Conditions → Run → Results → Report.
Runs on CPU in seconds. Computation lives in core/.
Run locally:  streamlit run app.py
"""
import copy
import datetime as dt
import hashlib
import io
import json
import re
import uuid
import zipfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402
import streamlit.components.v1 as components  # noqa: E402

from core import models as MOD  # noqa: E402
from core import molecules as MOL  # noqa: E402
from core import plots as PL  # noqa: E402
from core import report as REP  # noqa: E402
from core.viewer import viewer_html  # noqa: E402

st.set_page_config(page_title="BioScaff Studio", page_icon="🧬", layout="wide",
                   initial_sidebar_state="expanded")

st.markdown("""
<style>
.block-container{padding-top:1.6rem;max-width:1400px}
.hero{background:linear-gradient(135deg,#0b3c5d 0%,#0072B2 70%,#56B4E9 100%);color:#fff;
      padding:22px 28px;border-radius:16px;margin-bottom:14px}
.hero h1{color:#fff;font-size:2.0rem;margin:0 0 4px 0;font-weight:700}
.hero p{color:#e3f0fb;margin:0;font-size:1.0rem}
.steps{display:flex;gap:10px;flex-wrap:wrap;margin:6px 0 4px 0}
.step{flex:1;min-width:150px;border:1px solid #d9e2ec;border-radius:12px;padding:10px 14px;
      background:#f8fafc}
.step b{display:block;font-size:.95rem;color:#102a43}
.step span{font-size:.82rem;color:#627d98}
.step.done{border-color:#009E73;background:#effaf5}
.step.warn{border-color:#E69F00;background:#fff8e6}
.pill{display:inline-block;padding:2px 9px;border-radius:999px;font-size:.78rem;
      background:#e6f0fa;color:#0b3c5d;margin-right:4px}
.check-ok{color:#007a5a}.check-warn{color:#b26b00}.check-info{color:#486581}
.small{font-size:.85rem;color:#627d98}
</style>""", unsafe_allow_html=True)

ss = st.session_state
ss.setdefault("lib", {})              # molecule id -> entry
ss.setdefault("conds", [])            # ordered condition ids
ss.setdefault("cdata", {})            # condition id -> condition dict
ss.setdefault("settings", dict(MOD.DEFAULT_SETTINGS))
ss.setdefault("run", None)            # last run record
ss.setdefault("figs", {})             # figure name -> cached files
ss.setdefault("upl_key", 0)

SETTING_KEYS = ["pH", "T_C", "thickness_mm", "n_samples", "seed", "sd_pKa", "rel_conc",
                "rel_density", "rel_chain", "rel_drug", "rel_thickness", "rel_pore"]
if "_pending" in ss:                          # applied before any widget is created
    pend = ss.pop("_pending")
    for k, v in pend.items():
        ss[k] = v
    for k in SETTING_KEYS:
        ss[f"set_{k}"] = ss.settings[k]
for k in SETTING_KEYS:
    ss.setdefault(f"set_{k}", ss.settings[k])

ROLES_COND = ["Test", "Positive control", "Negative control", "Baseline"]
CHALLENGES = {"none": "None", "drug": "Drug / payload", "bacterium": "Bacterium"}
FIG_SIZES = {"Single column (3.5 in)": (3.5, 2.8), "1.5 column (5 in)": (5.0, 3.6),
             "Double column (7 in)": (7.0, 4.2), "Poster panel (10 in)": (10.0, 6.0)}


# ==========================================================================
# Sidebar
# ==========================================================================
def project_json() -> str:
    data = {"app": "BioScaff Studio", "version": 1,
            "project": ss.get("project", ""), "author": ss.get("author", ""),
            "settings": ss.settings, "molecules": ss.lib,
            "conditions": [ss.cdata[c] for c in ss.conds]}
    return json.dumps(data, default=lambda o: o.item() if hasattr(o, "item") else str(o),
                      indent=1)


def load_project(raw: bytes):
    data = json.loads(raw.decode("utf-8"))
    if data.get("app") != "BioScaff Studio":
        raise ValueError("This is not a BioScaff Studio project file.")
    for k in list(ss.keys()):
        if re.match(r"^(m[nrdfsvgt]|c[a-z]+|focus)_", str(k)) or k == "inspect":
            del ss[k]
    ss.lib = data["molecules"]
    for e in ss.lib.values():
        e["checks"] = [tuple(c) for c in e.get("checks", [])]
    ss.cdata = {c["id"]: c for c in data["conditions"]}
    ss.conds = [c["id"] for c in data["conditions"]]
    ss.settings = {**MOD.DEFAULT_SETTINGS, **data.get("settings", {})}
    ss._pending = {"project": data.get("project") or "Scaffold comparison",
                   "author": data.get("author", "")}
    ss.run, ss.figs = None, {}


with st.sidebar:
    st.markdown("### 🧬 BioScaff Studio")
    st.caption("Scaffold screening pipeline · CPU · no simulation needed")
    st.text_input("Project name", "Scaffold comparison", key="project")
    st.text_input("Author / team", "", key="author")
    st.divider()
    st.markdown("**Figures**")
    size_name = st.selectbox("Size", list(FIG_SIZES), index=1)
    DPI = st.select_slider("PNG resolution (dpi)", [300, 600, 1200], value=600)
    FONT = st.slider("Font size (pt)", 6, 16, 9)
    st.divider()
    st.markdown("**Project file**")
    st.download_button("⬇ Save project (.json)", project_json().encode(),
                       f"{re.sub(r'[^A-Za-z0-9]+', '_', ss.project) or 'project'}.bioscaff.json",
                       "application/json")
    up = st.file_uploader("Open project", type=["json"], key="proj_upload",
                          label_visibility="collapsed")
    if up is not None and st.button("Load this project"):
        try:
            load_project(up.getvalue())
            loaded = True
        except Exception as e:  # noqa: BLE001
            loaded = False
            st.error(f"Could not load: {e}")
        if loaded:
            st.rerun()
    st.caption("Sessions on a hosted app are temporary. Save the project file to keep "
               "your library and conditions.")
    if not MOL.rdkit_available():
        st.warning("RDKit is not installed: compound chemistry is limited.")

FIG_W, FIG_H = FIG_SIZES[size_name]
PL.configure(FIG_W, FIG_H, FONT)
SLUG = re.sub(r"[^A-Za-z0-9]+", "_", ss.project).strip("_") or "bioscaff"


# ==========================================================================
# Helpers
# ==========================================================================
def mol_name(mid):
    return ss.lib[mid]["name"] if mid in ss.lib else "(removed)"


def unique_name(name):
    names = {e["name"] for e in ss.lib.values()}
    base, k = name, 2
    while name in names:
        name, k = f"{base} ({k})", k + 1
    return name


def add_entry(name, text, fmt, source, smiles_template=None):
    entry = MOL.build_entry(unique_name(name.strip() or "Molecule"), text, fmt, source,
                            smiles_template)
    mid = uuid.uuid4().hex[:8]
    ss.lib[mid] = entry
    ss.inspect = mid
    return mid


def figure_block(name, builder, version, title="", caption=""):
    key = (version, FIG_W, FIG_H, DPI, FONT)
    cached = ss.figs.get(name)
    if not cached or cached["key"] != key:
        PL.configure(FIG_W, FIG_H, FONT)
        fig = builder()
        if fig is None:
            ss.figs.pop(name, None)
            return
        files = {}
        for fmt in ("png", "svg", "pdf"):
            buf = io.BytesIO()
            fig.savefig(buf, format=fmt, dpi=DPI)
            files[fmt] = buf.getvalue()
        prev = io.BytesIO()
        fig.savefig(prev, format="png", dpi=200)
        cached = {"key": key, "files": files, "preview": prev.getvalue(),
                  "width": int(fig.get_figwidth() * 110), "title": title, "caption": caption}
        plt.close(fig)
        ss.figs[name] = cached
    cached["title"], cached["caption"] = title, caption
    if title:
        st.markdown(f"**{title}**")
    st.image(cached["preview"], width=min(cached["width"], 1100))
    if caption:
        st.caption(caption)
    mime = {"png": "image/png", "svg": "image/svg+xml", "pdf": "application/pdf"}
    cols = st.columns([1, 1, 1, 3])
    for col, fmt in zip(cols, ("png", "svg", "pdf")):
        col.download_button(
            f"⬇ {fmt.upper()}" + (f" · {DPI} dpi" if fmt == "png" else " · vector"),
            cached["files"][fmt], f"{SLUG}_{name}.{fmt}", mime[fmt], key=f"dl_{name}_{fmt}")


def check_icon(level):
    return {"ok": "✅", "warn": "⚠️", "info": "ℹ️"}.get(level, "•")


def new_condition(name=None, **kw):
    cid = uuid.uuid4().hex[:8]
    n = len(ss.conds)
    letter = chr(65 + n) if n < 26 else str(n + 1)
    c = {"id": cid, "name": name or f"Condition {letter}", "role": "Test",
         "description": "", "components": [], "challenge": "none", "drug_id": None,
         "bacterium": {"species": "Staphylococcus aureus (coccus)", "width_um": 0.8,
                       "length_um": 0.8},
         "pores": None}
    c.update(kw)
    ss.cdata[cid] = c
    ss.conds.append(cid)
    return cid


def find_mol(*keys):
    for mid, e in ss.lib.items():
        n = e["name"].lower()
        if any((len(k) <= 3 and n == k) or (len(k) > 3 and k in n) for k in keys):
            return mid
    return None


def example_design():
    col = find_mol("collagen", "col")
    ha = find_mol("hyaluron", "ha")
    chi = find_mol("chitosan", "cs")
    pcl = find_mol("caprolactone", "pcl")
    drug = next((m for m, e in ss.lib.items() if e["role"] == "Drug / payload"), None)
    natural = [{"mol_id": m, "conc": 1.0} for m in (col, ha, chi) if m]
    hybrid = natural + ([{"mol_id": pcl, "conc": 1.0}] if pcl else [])
    ch = {"challenge": "drug", "drug_id": drug} if drug else {"challenge": "bacterium"}
    ss.conds, ss.cdata = [], {}
    new_condition("A · Positive control", role="Positive control",
                  description="Challenge in water with no scaffold.", **ch)
    new_condition("B · Natural scaffold", components=copy.deepcopy(natural), **ch)
    new_condition("C · Hybrid scaffold (PCL)", components=copy.deepcopy(hybrid), **ch)
    new_condition("D · Natural baseline", role="Baseline", components=copy.deepcopy(natural),
                  description="Scaffold in water, no challenge.")
    new_condition("E · Hybrid baseline", role="Baseline", components=copy.deepcopy(hybrid),
                  description="Scaffold in water, no challenge.")
    missing = [n for n, m in (("collagen", col), ("hyaluronic acid", ha),
                              ("chitosan", chi), ("PCL", pcl)) if not m]
    return missing, drug is None


def inputs_hash():
    blob = json.dumps({"lib": {k: {x: v[x] for x in ("name", "role", "density", "chain_radius_nm",
                                                      "hydrodynamic_radius_nm", "groups", "mw")}
                               for k, v in ss.lib.items()},
                       "conds": [ss.cdata[c] for c in ss.conds], "settings": ss.settings},
                      sort_keys=True, default=str)
    return hashlib.md5(blob.encode()).hexdigest()


# ==========================================================================
# Header
# ==========================================================================
def is_stale():
    return bool(ss.run) and ss.run["hash"] != inputs_hash()


header = st.container()


def render_header():
    n_mol = len(ss.lib)
    n_ver = sum(bool(e.get("verified")) for e in ss.lib.values())
    n_cond = len(ss.conds)
    run = ss.run
    stale = is_stale()
    header.markdown(f"""
<div class="hero"><h1>{ss.project or 'BioScaff Studio'}</h1>
<p>Build a molecule library, define any number of conditions, and compare hydration,
charge, molecular sieving, drug release and bacterial passage, with uncertainty
intervals, publication figures and a written report.</p></div>
<div class="steps">
 <div class="step {'done' if n_mol and n_ver == n_mol else ('warn' if n_mol else '')}">
   <b>① Library</b><span>{n_mol} molecule(s) · {n_ver} verified</span></div>
 <div class="step {'done' if n_cond else ''}"><b>② Conditions</b><span>{n_cond} defined</span></div>
 <div class="step {'warn' if stale else ('done' if run else '')}"><b>③ Run</b>
   <span>{'inputs changed: re-run' if stale else ('done ' + run['at'] if run else 'not run yet')}</span></div>
 <div class="step {'done' if run else ''}"><b>④ Results</b><span>{'ready' if run else '—'}</span></div>
 <div class="step {'done' if run else ''}"><b>⑤ Report</b><span>{'ready' if run else '—'}</span></div>
</div>""", unsafe_allow_html=True)


tabs = st.tabs(["① Molecule library", "② Conditions", "③ Run", "④ Results", "⑤ Report"])

# ==========================================================================
# ① Molecule library
# ==========================================================================
with tabs[0]:
    st.subheader("Add molecules")
    st.caption("Add every entity once: scaffold polymers, proteins, drugs. Water is built in "
               "as the solvent, so you don't need to upload it.")
    method = st.radio("Source", ["Upload files", "PubChem", "PDB ID (RCSB)", "SMILES"],
                      horizontal=True, label_visibility="collapsed")
    if method == "Upload files":
        files = st.file_uploader("Structure files: PDB (proteins or compounds), SDF/MOL, MOL2",
                                 type=["pdb", "ent", "sdf", "mol", "mol2"],
                                 accept_multiple_files=True, key=f"upl_{ss.upl_key}")
        st.caption("PDB files with amino-acid residues are treated as proteins; everything "
                   "else as a compound or oligomer. For compounds prefer SDF/MOL2, which "
                   "carry bond orders.")
        if files and st.button(f"Add {len(files)} file(s) to library", type="primary"):
            bar = st.progress(0.0)
            for i, f in enumerate(files):
                bar.progress(i / len(files), text=f"Reading {f.name}…")
                try:
                    add_entry(f.name.rsplit(".", 1)[0], f.getvalue().decode("utf-8", "replace"),
                              MOL.detect_format(f.name), f.name)
                except Exception as e:  # noqa: BLE001
                    st.error(f"{f.name}: {e}")
            bar.progress(1.0, text="Done")
            ss.upl_key += 1
            st.rerun()
    elif method == "PubChem":
        c1, c2 = st.columns([2, 2])
        q = c1.text_input("Compound name or CID", placeholder="e.g. ciprofloxacin")
        nm = c2.text_input("Name in library (optional)")
        if st.button("Fetch from PubChem", type="primary", disabled=not q):
            with st.spinner("Downloading and computing properties…"):
                try:
                    sdf, title = MOL.fetch_pubchem(q)
                    add_entry(nm or title, sdf, "sdf", f"PubChem: {q}")
                    ok = True
                except Exception as e:  # noqa: BLE001
                    ok = False
                    st.error(str(e))
            if ok:
                st.rerun()
    elif method == "PDB ID (RCSB)":
        c1, c2 = st.columns([2, 2])
        q = c1.text_input("PDB ID", placeholder="four characters, e.g. a collagen-like peptide")
        nm = c2.text_input("Name in library (optional)")
        if st.button("Fetch from RCSB", type="primary", disabled=not q):
            with st.spinner("Downloading and computing properties…"):
                try:
                    add_entry(nm or q.upper(), MOL.fetch_rcsb(q), "pdb", f"RCSB PDB: {q.upper()}")
                    ok = True
                except Exception as e:  # noqa: BLE001
                    ok = False
                    st.error(str(e))
            if ok:
                st.rerun()
    else:
        c1, c2 = st.columns([1, 3])
        nm = c1.text_input("Name", placeholder="e.g. PCL 5-mer")
        smi = c2.text_input("SMILES", placeholder="paste from PubChem, ChemDraw, etc.")
        with st.expander("Examples (no stereocentres, so the SMILES is exact)"):
            st.code("PCL 5-mer:      OCCCCCC(=O)OCCCCCC(=O)OCCCCCC(=O)OCCCCCC(=O)OCCCCCC(=O)O\n"
                    "Ciprofloxacin:  C1CC1N2C=C(C(=O)C3=CC(=C(C=C32)N4CCNCC4)F)C(=O)O",
                    language=None)
        if st.button("Build 3D structure", type="primary", disabled=not (nm and smi)):
            with st.spinner("Generating 3D coordinates and computing properties…"):
                try:
                    add_entry(nm, smi, "smiles", "SMILES")
                    ok = True
                except Exception as e:  # noqa: BLE001
                    ok = False
                    st.error(str(e))
            if ok:
                st.rerun()

    st.divider()
    st.subheader("Library")
    if not ss.lib:
        st.info("No molecules yet. Add them above, then inspect and verify each one.")
    else:
        rows = []
        for mid, e in ss.lib.items():
            nw = sum(1 for c in e["checks"] if c[0] == "warn")
            rows.append({"Name": e["name"], "Role": e["role"],
                         "Type": "Protein" if e["kind"] == "protein" else "Compound",
                         "Formula": e["formula"], "MW (Da)": round(e["mw"], 1),
                         "Atoms": e["n_atoms"],
                         "Checks": "✅ OK" if nw == 0 else f"⚠️ {nw} warning(s)",
                         "Verified": "✔" if e.get("verified") else "—",
                         "Source": e["source"]})
        st.dataframe(pd.DataFrame(rows), hide_index=True)

        ids = list(ss.lib)
        if ss.get("inspect") not in ids:
            ss.inspect = ids[0]
        mid = st.selectbox("Inspect and verify", ids, key="inspect", format_func=mol_name)
        e = ss.lib[mid]
        left, right = st.columns([1.15, 1])
        with left:
            components.html(viewer_html(e["display"], e["display_fmt"], e["name"],
                                        e["kind"] == "protein"), height=520)
            if e["kind"] == "compound" and MOL.rdkit_available():
                try:
                    svg = MOL.depiction(e)
                    b64 = __import__("base64").b64encode(svg.encode()).decode()
                    st.markdown(f"<img src='data:image/svg+xml;base64,{b64}' "
                                f"style='max-width:100%;background:#fff;border-radius:10px'>",
                                unsafe_allow_html=True)
                except Exception as ex:  # noqa: BLE001
                    st.caption(f"2D drawing unavailable: {ex}")
        with right:
            c1, c2 = st.columns(2)
            e["name"] = c1.text_input("Name", e["name"], key=f"mn_{mid}").strip() or e["name"]
            e["role"] = c2.selectbox("Role", MOL.ROLES, index=MOL.ROLES.index(e["role"]),
                                     key=f"mr_{mid}")
            st.markdown(" ".join(f"<span class='pill'>{x}</span>" for x in
                                 [("Protein" if e["kind"] == "protein" else "Compound"),
                                  e["formula"], f"{e['mw']:,.1f} Da", f"{e['n_atoms']:,} atoms"]),
                        unsafe_allow_html=True)
            st.markdown("**Verification checks**")
            for lvl, msg in e["checks"]:
                st.markdown(f"<div class='check-{lvl}'>{check_icon(lvl)} {msg}</div>",
                            unsafe_allow_html=True)
            if e["fmt"] == "pdb" and e["kind"] == "compound":
                tmpl = st.text_input("SMILES template for bond orders", e.get("smiles_template", ""),
                                     key=f"mt_{mid}")
                if st.button("Re-read with template", disabled=not tmpl):
                    try:
                        new = MOL.build_entry(e["name"], e["raw"], "pdb", e["source"], tmpl)
                        new.update(name=e["name"], role=e["role"])
                        ss.lib[mid] = new
                        ok = True
                    except Exception as ex:  # noqa: BLE001
                        ok = False
                        st.error(str(ex))
                    if ok:
                        st.rerun()

            p = e["props"]
            prop_rows = [("Solvent-accessible surface", f"{p['sasa_A2'] / 100:,.1f} nm²"),
                         ("Polar surface fraction", f"{p['polar_sasa_fraction']:.3f}"),
                         ("Polar atoms (N + O)", f"{p['N_plus_O']:,} · "
                                                 f"{p['N_plus_O'] / (e['mw'] / 1000):.0f} per kDa"),
                         ("Van der Waals volume", f"{p['vdw_volume_A3'] / 1000:,.2f} nm³"),
                         ("Radius of gyration", f"{p['radius_of_gyration_A'] / 10:.2f} nm"),
                         ("Length (longest axis)", f"{p['length_A'] / 10:.2f} nm")]
            for k, lab in [("TPSA_A2", "Topological polar surface (Å²)"), ("HBD", "H-bond donors"),
                           ("HBA", "H-bond acceptors"), ("logP_Crippen", "logP (Crippen)"),
                           ("rotatable_bonds", "Rotatable bonds")]:
                if k in p:
                    prop_rows.append((lab, f"{p[k]:.2f}" if isinstance(p[k], float) else str(p[k])))
            q74, qp, qn = MOL.charge_at_pH(e["groups"], ss.settings["pH"], e["permanent"])
            prop_rows.append((f"Net charge at pH {ss.settings['pH']:g}", f"{float(q74):+.2f} e"))
            with st.expander("Properties", expanded=True):
                st.dataframe(pd.DataFrame(prop_rows, columns=["Property", "Value"]),
                             hide_index=True)

            with st.expander("Model inputs (editable)", expanded=False):
                st.caption("Defaults come from the structure. Replace with literature values "
                           "when you have them.")
                c1, c2, c3 = st.columns(3)
                e["density"] = c1.number_input("Density (g/cm³)", 0.5, 3.0, float(e["density"]),
                                               0.01, key=f"md_{mid}")
                e["chain_radius_nm"] = c2.number_input(
                    "Chain radius (nm)", 0.05, 10.0, float(e["chain_radius_nm"]), 0.01,
                    key=f"mf_{mid}", help="Fibre radius used in the network model.")
                e["hydrodynamic_radius_nm"] = c3.number_input(
                    "Hydrodynamic radius (nm)", 0.05, 50.0, float(e["hydrodynamic_radius_nm"]),
                    0.01, key=f"ms_{mid}", help="Used when this molecule diffuses (drugs).")
                if e["groups"]:
                    st.markdown("Ionizable groups · edit pKa if you have literature values")
                    gdf = pd.DataFrame(e["groups"])[["group", "kind", "count", "pKa"]]
                    edited = st.data_editor(
                        gdf, key=f"mg_{mid}", hide_index=True, num_rows="fixed",
                        disabled=["group", "kind"],
                        column_config={"count": st.column_config.NumberColumn(min_value=0, step=1),
                                       "pKa": st.column_config.NumberColumn(format="%.2f")})
                    e["groups"] = edited.to_dict("records")
                else:
                    st.caption("No ionizable groups detected.")

            e["verified"] = st.toggle("I have checked this structure", e.get("verified", False),
                                      key=f"mv_{mid}")
            ext = "pdb" if e["display_fmt"] == "pdb" else "sdf"
            d1, d2, d3, d4 = st.columns(4)
            d1.download_button(f"⬇ .{ext}", e["display"].encode(),
                               f"{re.sub(r'[^A-Za-z0-9]+', '_', e['name'])}.{ext}", key=f"dls_{mid}")
            d2.download_button("⬇ Properties", pd.DataFrame(prop_rows, columns=["Property", "Value"])
                               .to_csv(index=False).encode(),
                               f"{re.sub(r'[^A-Za-z0-9]+', '_', e['name'])}_properties.csv",
                               key=f"dlp_{mid}")
            if e["kind"] == "compound" and MOL.rdkit_available():
                try:
                    d3.download_button("⬇ 2D PNG", MOL.depiction(e, 900, 700, png=True),
                                       f"{re.sub(r'[^A-Za-z0-9]+', '_', e['name'])}_2D.png",
                                       key=f"dl2_{mid}")
                except Exception:  # noqa: BLE001
                    pass
            if d4.button("🗑 Remove", key=f"rm_{mid}"):
                ss.lib.pop(mid)
                for c in ss.cdata.values():
                    c["components"] = [x for x in c["components"] if x["mol_id"] != mid]
                    if c.get("drug_id") == mid:
                        c["drug_id"] = None
                st.rerun()

        scaff = [x for x in ss.lib.values() if x["role"] != "Solvent"]
        if len(scaff) >= 1:
            st.divider()
            st.subheader("Library figures")
            figure_block("molecule_properties", lambda: PL.molecule_properties(scaff),
                         json.dumps([(x["name"], x["mw"], x["hydrodynamic_radius_nm"],
                                      x["props"]["polar_sasa_fraction"]) for x in scaff]),
                         "Molecular descriptors",
                         "Molecular weight, polar surface fraction, polar atoms per kDa and "
                         "equivalent-sphere radius of each molecule.")
            comp = [x for x in scaff if x["kind"] == "compound"]
            if comp and MOL.rdkit_available():
                try:
                    grid = MOL.depiction_grid(comp)
                    st.download_button("⬇ 2D structure panel (SVG)", grid.encode(),
                                       f"{SLUG}_structures.svg", "image/svg+xml")
                except Exception:  # noqa: BLE001
                    pass

# ==========================================================================
# ② Conditions
# ==========================================================================
with tabs[1]:
    st.subheader("Experiment settings")
    c1, c2, c3 = st.columns(3)
    ss.settings["pH"] = c1.number_input("pH", 1.0, 13.0, step=0.1, key="set_pH",
                                        help="7.4 = physiological; wound beds are often 7–8.5.")
    ss.settings["T_C"] = c2.number_input("Temperature (°C)", 4.0, 80.0, step=0.5, key="set_T_C")
    ss.settings["thickness_mm"] = c3.number_input("Scaffold thickness (mm)", 0.01, 20.0,
                                                  step=0.1, key="set_thickness_mm",
                                                  help="Used for drug-release times.")

    st.subheader("Conditions")
    b1, b2, b3, _ = st.columns([1, 1.4, 1, 2])
    if b1.button("➕ Add condition", type="primary"):
        new_condition()
        st.rerun()
    if b2.button("Load example design (A–E)", help="Positive control, natural and hybrid "
                 "scaffolds with a challenge, and both scaffolds without one."):
        missing, nodrug = example_design()
        if missing:
            st.toast("Not in library: " + ", ".join(missing) + ". Pick components manually.")
        st.rerun()
    if b3.button("Clear all", disabled=not ss.conds):
        ss.conds, ss.cdata = [], {}
        st.rerun()

    pickable = [m for m, e in ss.lib.items() if e["role"] != "Solvent"]
    drugs = [m for m, e in ss.lib.items() if e["role"] == "Drug / payload"] or \
            [m for m, e in ss.lib.items() if e["role"] != "Solvent" and e["kind"] == "compound"]
    if not ss.conds:
        st.info("Add a condition, or load the example design and edit it. Any number of "
                "conditions and any molecules from the library can be combined.")

    action = None
    for i, cid in enumerate(ss.conds):
        c = ss.cdata[cid]
        with st.container(border=True):
            h1, h2, h3, h4 = st.columns([3, 2, 0.6, 0.6])
            c["name"] = h1.text_input("Name", c["name"], key=f"cname_{cid}")
            c["role"] = h2.selectbox("Type", ROLES_COND, index=ROLES_COND.index(c["role"]),
                                     key=f"crole_{cid}")
            h3.write("")
            if h3.button("⧉", key=f"cdup_{cid}", help="Duplicate"):
                action = ("dup", cid)
            h4.write("")
            if h4.button("🗑", key=f"cdel_{cid}", help="Delete"):
                action = ("del", cid)
            c["description"] = st.text_input("Description (optional)", c.get("description", ""),
                                             key=f"cdesc_{cid}")
            current = [x["mol_id"] for x in c["components"] if x["mol_id"] in pickable]
            if f"ccomp_{cid}" in ss:            # drop molecules removed or re-roled
                ss[f"ccomp_{cid}"] = [x for x in ss[f"ccomp_{cid}"] if x in pickable]
            if ss.get(f"cdrug_{cid}") is not None and ss[f"cdrug_{cid}"] not in drugs:
                del ss[f"cdrug_{cid}"]
            chosen = st.multiselect("Scaffold components (leave empty for water only)",
                                    pickable, current, format_func=mol_name,
                                    key=f"ccomp_{cid}")
            conc_old = {x["mol_id"]: x["conc"] for x in c["components"]}
            comps = []
            if chosen:
                cols = st.columns(min(len(chosen), 4))
                for j, m in enumerate(chosen):
                    v = cols[j % len(cols)].number_input(
                        f"{mol_name(m)} (% w/v)", 0.0, 100.0, float(conc_old.get(m, 1.0)), 0.1,
                        key=f"cconc_{cid}_{m}")
                    comps.append({"mol_id": m, "conc": v})
            c["components"] = comps
            keys = list(CHALLENGES)
            c["challenge"] = st.radio("Challenge", keys, index=keys.index(c["challenge"]),
                                      format_func=CHALLENGES.get, horizontal=True,
                                      key=f"cchal_{cid}")
            if c["challenge"] == "drug":
                if not drugs:
                    st.warning("No molecule with role 'Drug / payload' in the library.")
                else:
                    idx = drugs.index(c["drug_id"]) if c.get("drug_id") in drugs else 0
                    c["drug_id"] = st.selectbox("Drug", drugs, idx, format_func=mol_name,
                                                key=f"cdrug_{cid}")
            elif c["challenge"] == "bacterium":
                b = c.get("bacterium") or {}
                sp = list(MOD.BACTERIA)
                k1, k2, k3 = st.columns([2, 1, 1])
                species = k1.selectbox("Bacterium", sp,
                                       sp.index(b.get("species")) if b.get("species") in sp else 0,
                                       key=f"cbsp_{cid}")
                wd, ln = MOD.BACTERIA[species]
                if species == b.get("species"):
                    wd, ln = b.get("width_um", wd), b.get("length_um", ln)
                width = k2.number_input("Width (µm)", 0.05, 20.0, float(wd), 0.05,
                                        key=f"cbw_{cid}_{species}")
                length = k3.number_input("Length (µm)", 0.05, 50.0, float(ln), 0.05,
                                         key=f"cbl_{cid}_{species}")
                c["bacterium"] = {"species": species, "width_um": width, "length_um": length}
                if c["components"]:
                    has = st.toggle("Estimate passage from a pore-size distribution",
                                    bool(c.get("pores")), key=f"cbp_{cid}",
                                    help="Without SEM data, enter pore sizes reported in the "
                                         "literature for the same fabrication method and cite "
                                         "the source.")
                    if has:
                        pr = c.get("pores") or {"mean_um": 0.0, "cv": 0.5, "source": ""}
                        p1, p2, p3 = st.columns([1, 1, 2])
                        mean = p1.number_input("Mean pore diameter (µm)", 0.0, 2000.0,
                                               float(pr["mean_um"]), 1.0, key=f"cpm_{cid}")
                        cv = p2.number_input("Spread (CV)", 0.05, 3.0, float(pr["cv"]), 0.05,
                                             key=f"cpc_{cid}",
                                             help="SD ÷ mean of pore diameter; 0.5 if unknown.")
                        src = p3.text_input("Source (citation or 'SEM, this work')",
                                            pr.get("source", ""), key=f"cps_{cid}")
                        c["pores"] = {"mean_um": mean, "cv": cv, "source": src}
                        if mean <= 0:
                            st.caption("Enter a mean pore diameter to enable the estimate.")
                    else:
                        c["pores"] = None
                        st.caption("Without pore sizes, the cationic charge density is used "
                                   "as the antibacterial proxy.")
    if action:
        kind, cid = action
        if kind == "del":
            ss.conds.remove(cid)
            ss.cdata.pop(cid, None)
        else:
            src = copy.deepcopy(ss.cdata[cid])
            new = new_condition(src["name"] + " (copy)")
            src.update(id=new, name=src["name"] + " (copy)")
            ss.cdata[new] = src
        st.rerun()

    if ss.conds:
        with st.expander("Design check", expanded=True):
            issues = []
            names = [ss.cdata[c]["name"] for c in ss.conds]
            if len(set(names)) < len(names):
                issues.append(("warn", "Two conditions share a name."))
            for c in (ss.cdata[x] for x in ss.conds):
                if c["challenge"] == "drug" and not c.get("drug_id"):
                    issues.append(("warn", f"{c['name']}: drug challenge without a drug."))
                if not c["components"] and c["challenge"] == "none":
                    issues.append(("info", f"{c['name']}: water only with no challenge has "
                                           "nothing to measure; it serves as a blank."))
            conc = {}
            for c in (ss.cdata[x] for x in ss.conds):
                for x in c["components"]:
                    conc.setdefault(x["mol_id"], set()).add(round(x["conc"], 3))
            for m, vals in conc.items():
                if len(vals) > 1:
                    issues.append(("info", f"{mol_name(m)} is used at different concentrations "
                                           f"({', '.join(map(str, sorted(vals)))} % w/v). "
                                           "Differences may then reflect amount, not composition."))
            used = {x["mol_id"] for c in ss.cdata.values() for x in c["components"]} | \
                   {c.get("drug_id") for c in ss.cdata.values() if c["challenge"] == "drug"}
            unver = [mol_name(m) for m in used if m in ss.lib and not ss.lib[m].get("verified")]
            if unver:
                issues.append(("warn", "Not yet verified in the library: " + ", ".join(unver)))
            if not issues:
                st.markdown("✅ No issues found.")
            for lvl, msg in issues:
                st.markdown(f"<div class='check-{lvl}'>{check_icon(lvl)} {msg}</div>",
                            unsafe_allow_html=True)
            summary = []
            for c in (ss.cdata[x] for x in ss.conds):
                ch = CHALLENGES[c["challenge"]]
                if c["challenge"] == "drug" and c.get("drug_id"):
                    ch += f": {mol_name(c['drug_id'])}"
                if c["challenge"] == "bacterium":
                    ch += f": {c['bacterium']['species'].split(' (')[0]}"
                summary.append({"Condition": c["name"], "Type": c["role"],
                                "Composition": " + ".join(f"{mol_name(x['mol_id'])} {x['conc']:g}%"
                                                          for x in c["components"]) or "water only",
                                "Challenge": ch})
            st.dataframe(pd.DataFrame(summary), hide_index=True)

# ==========================================================================
# ③ Run
# ==========================================================================
with tabs[2]:
    st.subheader("Run the pipeline")
    st.markdown("Each condition is evaluated with physical models from the structures and "
                "composition you defined. Uncertain inputs are sampled many times, so every "
                "result has a 95 % interval. A typical run takes a few seconds.")
    s = ss.settings
    with st.expander("Uncertainty settings (defaults are sensible)"):
        c1, c2, c3, c4 = st.columns(4)
        s["n_samples"] = c1.select_slider("Monte Carlo samples",
                                          [200, 500, 1000, 2000, 5000, 10000, 20000],
                                          key="set_n_samples")
        s["seed"] = int(c2.number_input("Random seed", 0, 10 ** 6, step=1, key="set_seed"))
        s["sd_pKa"] = c3.number_input("pKa uncertainty (SD)", 0.0, 2.0, step=0.05, key="set_sd_pKa")
        s["rel_conc"] = c4.number_input("Concentration (rel. SD)", 0.0, 1.0, step=0.01,
                                        key="set_rel_conc")
        c1, c2, c3, c4 = st.columns(4)
        s["rel_density"] = c1.number_input("Density (rel. SD)", 0.0, 1.0, step=0.01,
                                           key="set_rel_density")
        s["rel_chain"] = c2.number_input("Chain radius (rel. SD)", 0.0, 1.0, step=0.01,
                                         key="set_rel_chain")
        s["rel_drug"] = c3.number_input("Drug radius (rel. SD)", 0.0, 1.0, step=0.01,
                                        key="set_rel_drug")
        s["rel_thickness"] = c4.number_input("Thickness (rel. SD)", 0.0, 1.0, step=0.01,
                                             key="set_rel_thickness")
        s["rel_pore"] = st.number_input("Mean pore size (rel. SD)", 0.0, 1.0, step=0.01,
                                        key="set_rel_pore")
        if st.button("Reset to defaults"):
            keep = {k: s[k] for k in ("pH", "T_C", "thickness_mm")}
            ss.settings = {**MOD.DEFAULT_SETTINGS, **keep}
            ss._pending = {}
            st.rerun()

    if ss.conds:
        ids = ss.conds
        tests = [c for c in ids if ss.cdata[c]["role"] == "Test" and ss.cdata[c]["components"]] \
            or [c for c in ids if ss.cdata[c]["components"]] or ids
        tests = sorted(tests, key=lambda c: len(ss.cdata[c]["components"]))   # stable
        for k in ("focus_t", "focus_r"):
            if ss.get(k) not in ids:
                ss.pop(k, None)
        c1, c2 = st.columns(2)
        default_t = tests[-1]
        default_r = tests[0] if tests[0] != default_t else ids[0]
        ft = c1.selectbox("Primary comparison: test condition", ids, ids.index(default_t),
                          format_func=lambda x: ss.cdata[x]["name"], key="focus_t")
        fr = c2.selectbox("…against reference condition", ids, ids.index(default_r),
                          format_func=lambda x: ss.cdata[x]["name"], key="focus_r")

    ready = bool(ss.conds) and bool(ss.lib)
    if not ready:
        st.info("Add molecules and at least one condition first.")
    if is_stale():
        st.warning("Inputs changed since the last run. Run again to update the results.")
    if st.button("▶ Run all conditions", type="primary", disabled=not ready):
        t0 = dt.datetime.now()
        bar = st.progress(0.0, text="Preparing…")
        conds = [copy.deepcopy(ss.cdata[c]) for c in ss.conds]
        mols = copy.deepcopy(ss.lib)
        try:
            res = MOD.run_conditions(mols, conds, s, progress=lambda f, n: bar.progress(
                f * 0.9, text=f"Evaluating {n}…"))
            bar.progress(0.95, text="Writing summary…")
            summary = REP.write_summary(res, conds, mols, s, (ss.focus_t, ss.focus_r))
            ss.run = {"results": res, "conds": conds, "mols": mols, "settings": dict(s),
                      "summary": summary, "focus": (ss.focus_t, ss.focus_r),
                      "at": dt.datetime.now().strftime("%H:%M"), "hash": inputs_hash(),
                      "id": uuid.uuid4().hex[:6]}
            ss.figs = {k: v for k, v in ss.figs.items() if k == "molecule_properties"}
            bar.progress(1.0, text=f"Done in {(dt.datetime.now() - t0).total_seconds():.1f} s")
            done = True
        except Exception as e:  # noqa: BLE001
            done = False
            bar.empty()
            st.exception(e)
        if done:
            st.rerun()
    if ss.run and not is_stale():
        st.success(f"Last run at {ss.run['at']}: {len(ss.run['results'])} conditions, "
                   f"{ss.run['settings']['n_samples']:,} samples each.")

# ==========================================================================
# ④ Results
# ==========================================================================
SECTION_TEXT = {}
with tabs[3]:
    if not ss.run:
        st.info("Run the pipeline to see results.")
    else:
        R, conds, rs = ss.run["results"], ss.run["conds"], ss.run["settings"]
        order = [c["id"] for c in conds]
        idx = {cid: i for i, cid in enumerate(order)}
        items = [(R[c]["name"], idx[c], R[c]["summary"]) for c in order]
        scaff_items = [it for it, c in zip(items, order) if R[c]["has_scaffold"]]
        ver = ss.run["id"]
        S = ss.run["summary"]
        if is_stale():
            st.warning("Inputs changed since this run. Results below are from the last run.")

        st.subheader("Key results")
        table = REP.results_table(R, order)
        st.dataframe(table, hide_index=True)
        st.download_button("⬇ Results table (CSV)", table.to_csv(index=False).encode(),
                           f"{SLUG}_results.csv", "text/csv")
        if "Primary comparison" in S:
            with st.container(border=True):
                st.markdown("#### Primary comparison")
                st.markdown(S["Primary comparison"])

        st.divider()
        figure_block("composition", lambda: PL.composition(
            [(c["name"], {ss.run["mols"][x["mol_id"]]["name"]: x["conc"]
                          for x in c["components"]}) for c in conds]), ver,
            "Composition of each condition", "Scaffold components in % w/v; water is the solvent.")
        figure_block("overview_panel", lambda: PL.panel(items, ["polar_sasa_fraction",
                                                                "cationic_per_kDa", "D_rel",
                                                                "pore_pass_fraction"]), ver,
                     "Overview", "Key metric per condition: median and 95 % interval.")

        st.divider()
        st.subheader("💧 Hydration capacity")
        if "Hydration capacity" in S:
            st.markdown(S["Hydration capacity"])
        c1, c2 = st.columns(2)
        with c1:
            figure_block("polar_surface", lambda: PL.metric_bars(scaff_items, "polar_sasa_fraction"),
                         ver, "Polar surface fraction",
                         "Mass-weighted share of solvent-accessible surface on N and O atoms.")
        with c2:
            figure_block("polar_atoms", lambda: PL.metric_bars(scaff_items, "polar_per_kDa"),
                         ver, "Polar atoms per kDa", "H-bonding N + O atoms per mass of scaffold.")

        st.subheader("⚡ Charge and antibacterial proxy")
        if "Charge and antibacterial proxy" in S:
            st.markdown(S["Charge and antibacterial proxy"])
        c1, c2 = st.columns(2)
        with c1:
            figure_block("charge_vs_pH", lambda: PL.charge_curves(
                [(R[c]["name"], idx[c], R[c]["curves"]["charge_pH"]) for c in order
                 if "charge_pH" in R[c]["curves"]], rs["pH"]), ver, "Net charge versus pH",
                "Henderson–Hasselbalch titration of all ionizable groups; band = 95 % interval.")
        with c2:
            figure_block("cationic_charge", lambda: PL.metric_bars(scaff_items, "cationic_per_kDa"),
                         ver, f"Cationic charge at pH {rs['pH']:g}",
                         "Positive charge per kDa, a proxy for contact antibacterial activity.")

        st.subheader("🕸️ Network density and molecular sieving")
        if "Network density" in S:
            st.markdown(S["Network density"])
        c1, c2 = st.columns(2)
        with c1:
            figure_block("volume_fraction", lambda: PL.metric_bars(scaff_items, "phi"), ver,
                         "Polymer volume fraction", "Concentration ÷ density, summed over components.")
        with c2:
            markers = sorted({(R[c]["drug"], R[c]["drug_radius_nm"]) for c in order if "drug" in R[c]})
            figure_block("sieving", lambda: PL.sieving(
                [(R[c]["name"], idx[c], R[c]["curves"]["sieving"]) for c in order
                 if R[c]["has_scaffold"]], markers), ver, "Molecular sieving",
                "Ogston model: how strongly each network slows solutes of a given size.")

        if any("D_rel" in R[c]["summary"] for c in order):
            st.subheader("💊 Drug transport and release")
            st.markdown(S.get("Drug transport and release", ""))
            c1, c2 = st.columns(2)
            with c1:
                figure_block("release", lambda: PL.release(
                    [(R[c]["name"], idx[c], R[c]["curves"]["release"]) for c in order
                     if "release" in R[c]["curves"]]), ver, "Predicted release profile",
                    f"Fickian release from a {rs['thickness_mm']:g} mm slab; band = 95 % interval.")
            with c2:
                figure_block("t50", lambda: PL.metric_bars(items, "t50_h", log=True), ver,
                             "Time to 50 % release", "Log scale.")

        if any(R[c]["challenge"] == "bacterium" for c in order):
            st.subheader("🦠 Bacterial challenge")
            st.markdown(S.get("Bacterial challenge", ""))
            c1, c2 = st.columns(2)
            with c1:
                figure_block("bacterial_passage", lambda: PL.metric_bars(
                    items, "pore_pass_fraction", ylabel="Pores wider than bacterium (fraction)"),
                    ver, "Bacterial size exclusion",
                    "Share of pores wider than the bacterial cell width.")
            with c2:
                widths = sorted({(R[c]["bacterium"]["species"].split(" (")[0],
                                  R[c]["bacterium"]["width_um"]) for c in order
                                 if "pores" in R[c]["curves"]})
                figure_block("pore_distributions", lambda: PL.pores(
                    [(R[c]["name"], idx[c], R[c]["curves"]["pores"]) for c in order
                     if "pores" in R[c]["curves"]], widths), ver, "Pore-size distributions",
                    "Log-normal distributions from the entered pore sizes; dashed = cell width.")

        notes = [(R[c]["name"], n) for c in order for n in R[c]["notes"]]
        if notes:
            with st.expander("Model notes"):
                for nm, n in notes:
                    st.markdown(f"- **{nm}**: {n}")

# ==========================================================================
# ⑤ Report
# ==========================================================================
with tabs[4]:
    if not ss.run:
        st.info("Run the pipeline to generate the report.")
    else:
        S = ss.run["summary"]
        st.subheader("Written summary")
        st.caption("Generated from the numbers. It reports what was computed and how robust "
                   "each difference is; drawing the conclusion about your hypothesis is yours.")
        for sec in ["Overview", "Primary comparison", "Hydration capacity",
                    "Charge and antibacterial proxy", "Network density",
                    "Drug transport and release", "Bacterial challenge"]:
            if sec in S:
                st.markdown(f"##### {sec}")
                st.markdown(S[sec])
        with st.expander("Methods"):
            st.markdown(S["Methods"])
        with st.expander("Limitations", expanded=True):
            st.markdown(S["Limitations"])

        R, conds = ss.run["results"], ss.run["conds"]
        order = [c["id"] for c in conds]
        lib_df = pd.DataFrame([{"Name": e["name"], "Role": e["role"],
                                "Type": "Protein" if e["kind"] == "protein" else "Compound",
                                "Formula": e["formula"], "MW (Da)": f"{e['mw']:,.1f}",
                                "Source": e["source"]} for e in ss.run["mols"].values()])
        cond_df = pd.DataFrame([{"Condition": c["name"], "Type": c["role"],
                                 "Composition": " + ".join(
                                     f"{ss.run['mols'][x['mol_id']]['name']} {x['conc']:g}% w/v"
                                     for x in c["components"]) or "water only",
                                 "Challenge": CHALLENGES[c["challenge"]] + (
                                     f": {ss.run['mols'][c['drug_id']]['name']}"
                                     if c["challenge"] == "drug" and c.get("drug_id") in ss.run["mols"]
                                     else (f": {c['bacterium']['species']}"
                                           if c["challenge"] == "bacterium" else ""))}
                                for c in conds])
        res_df = REP.results_table(R, order)
        figs = [(v["title"] or k, v["caption"], v["files"]["png"]) for k, v in ss.figs.items()]
        html_doc = REP.html_report(ss.project, ss.author, S,
                                   {"Conditions": cond_df, "Results": res_df}, figs, lib_df)
        st.divider()
        c1, c2 = st.columns(2)
        c1.download_button("⬇ Report (HTML, print to PDF from the browser)", html_doc.encode(),
                           f"{SLUG}_report.html", "text/html", type="primary")
        zbuf = io.BytesIO()
        with zipfile.ZipFile(zbuf, "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr("report.html", html_doc)
            z.writestr("data/results_summary.csv", res_df.to_csv(index=False))
            z.writestr("data/conditions.csv", cond_df.to_csv(index=False))
            z.writestr("data/molecules.csv", lib_df.to_csv(index=False))
            for cid in order:
                r = R[cid]
                nm = re.sub(r"[^A-Za-z0-9]+", "_", r["name"]).strip("_")
                if r["samples"]:
                    z.writestr(f"data/samples/{nm}.csv",
                               pd.DataFrame({k: np.asarray(v) for k, v in r["samples"].items()})
                               .to_csv(index=False))
                for cname, curve in r["curves"].items():
                    cols = ["x", "median", "lo95", "hi95"] if len(curve) == 4 else ["x", "density"]
                    z.writestr(f"data/curves/{nm}_{cname}.csv",
                               pd.DataFrame(dict(zip(cols, curve))).to_csv(index=False))
            for k, v in ss.figs.items():
                for fmt, data in v["files"].items():
                    z.writestr(f"figures/{SLUG}_{k}.{fmt}", data)
            for mid, e in ss.run["mols"].items():
                ext = "pdb" if e["display_fmt"] == "pdb" else "sdf"
                z.writestr(f"structures/{re.sub(r'[^A-Za-z0-9]+', '_', e['name'])}.{ext}",
                           e["display"])
            z.writestr("project.bioscaff.json", project_json())
            z.writestr("methods.txt", S["Methods"] + "\n\nLimitations\n" + S["Limitations"])
        c2.download_button("⬇ Everything (.zip): report, figures, data, structures",
                           zbuf.getvalue(), f"{SLUG}_bundle.zip", "application/zip")
        st.caption(f"{len(figs)} figures included, each also as SVG and PDF in the zip.")

render_header()
