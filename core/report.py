"""Auto-written summary and the HTML report."""
from __future__ import annotations

import base64
import datetime as dt
import html

import numpy as np
import pandas as pd

from core.models import METRICS, prob_greater

REFERENCES = [
    "Ogston AG, Preston BN, Wells JD (1973). On the transport of compact particles "
    "through solutions of chain-polymers. Proc R Soc Lond A 333:297–316.",
    "Amsden B (1998). Solute diffusion within hydrogels. Mechanisms and models. "
    "Macromolecules 31:8382–8395.",
    "Crank J (1975). The Mathematics of Diffusion, 2nd ed. Oxford University Press.",
    "Shrake A, Rupley JA (1973). Environment and exposure to solvent of protein atoms. "
    "J Mol Biol 79:351–371.",
    "Tsai J, Taylor R, Chothia C, Gerstein M (1999). The packing density in proteins: "
    "standard radii and volumes. J Mol Biol 290:253–266.",
    "Rego N, Koes D (2015). 3Dmol.js: molecular visualization with WebGL. "
    "Bioinformatics 31:1322–1324.",
    "RDKit: Open-source cheminformatics. https://www.rdkit.org",
]

LIMITATIONS = [
    "These are model-based screening estimates computed from single structures and the "
    "stated composition. They are not molecular dynamics simulations or measurements.",
    "Ionization uses generic group pKa values unless replaced with literature values; "
    "neighbouring charges and the polymer environment shift real pKa values.",
    "The Ogston model treats the scaffold as a random network of rigid fibres in water. "
    "It ignores drug binding, swelling, degradation and hydrophobic PCL domains, so "
    "predicted release is faster than typically observed.",
    "Molecular-scale results (Å–nm) do not directly predict bacterial penetration, which "
    "depends on micrometre-scale pores. Bacterial passage is only estimated when a "
    "pore-size distribution is supplied.",
    "Intervals reflect the stated input uncertainties, not experimental replicates.",
]


PERCENT_KEYS = {"pore_pass_fraction"}


def fmt_num(x, key=None):
    if x is None or not np.isfinite(x):
        return "—"
    if key in PERCENT_KEYS:
        return f"{x * 100:.1f} %" if 0.001 < x < 0.999 else f"{x * 100:.0f} %"
    if key == "D_cm2s" or (abs(x) > 0 and (abs(x) < 1e-3 or abs(x) >= 1e5)):
        t = f"{x:.2e}"
    elif abs(x) >= 100:
        t = f"{x:.0f}"
    elif abs(x) >= 10:
        t = f"{x:.1f}"
    else:
        t = f"{x:.3g}"
    return t.replace("-", "−") if not t.startswith("e") else t


def fmt_summary(s, key=None):
    if not s or not np.isfinite(s["median"]):
        return "—"
    m = fmt_num(s["median"], key)
    lo, hi = fmt_num(s["lo"], key), fmt_num(s["hi"], key)
    if lo == hi:
        return m
    sep = " to " if (s["lo"] < 0 or s["hi"] < 0) else "–"
    return f"{m} ({lo}{sep}{hi})"


def fmt_time_h(s):
    """Summary of a time in hours, shown in the most readable unit."""
    med = s["median"]
    for unit, f in (("s", 3600), ("min", 60), ("h", 1), ("days", 1 / 24)):
        if med * f < (60 if unit in ("s", "min") else (48 if unit == "h" else np.inf)):
            sc = {k: s[k] * f for k in ("median", "lo", "hi")}
            return f"{fmt_summary(sc)} {unit}"
    return f"{fmt_summary(s)} h"


def results_table(results: dict, order: list) -> pd.DataFrame:
    keys = [k for k in METRICS if any(k in results[c]["summary"] for c in order)]
    rows = []
    for cid in order:
        r = results[cid]
        row = {"Condition": r["name"], "Type": r["role"]}
        for k in keys:
            label, unit, _ = METRICS[k]
            row[f"{label} ({unit})" if unit else label] = fmt_summary(r["summary"].get(k), k)
        rows.append(row)
    return pd.DataFrame(rows)


def _robust(p):
    if not np.isfinite(p):
        return ""
    q = max(p, 1 - p)
    if q >= 0.95:
        return f"a robust difference (holds in {q:.0%} of samples)"
    if q >= 0.80:
        return f"a likely difference (holds in {q:.0%} of samples)"
    return f"not distinguishable within the input uncertainty ({q:.0%} of samples)"


def _groups(results, order, conditions_by_id):
    """Collapse conditions with identical scaffold composition (e.g. B and D)."""
    seen, out = {}, []
    for cid in order:
        comp = tuple(sorted((x["mol_id"], round(x["conc"], 4))
                            for x in conditions_by_id[cid]["components"]))
        if comp in seen:
            seen[comp][1].append(results[cid]["name"])
        else:
            seen[comp] = (cid, [results[cid]["name"]])
            out.append(seen[comp])
    return [(cid, " / ".join(f"**{n}**" for n in names)) for cid, names in out]


def _rank_sentence(results, groups, key, higher_word="highest", lower_word="lowest"):
    vals = [(cid, label, results[cid]["summary"][key]["median"]) for cid, label in groups
            if key in results[cid]["summary"] and np.isfinite(results[cid]["summary"][key]["median"])]
    if len(vals) < 2:
        return ""
    vals.sort(key=lambda x: x[2], reverse=True)
    (top, tl, _), (bottom, bl, _) = vals[0], vals[-1]
    label, unit, _ = METRICS[key]
    if key.startswith("t") and key.endswith("_h"):
        ft, fb = fmt_time_h(results[top]["summary"][key]), fmt_time_h(results[bottom]["summary"][key])
    else:
        u = f" {unit}" if unit else ""
        ft = fmt_summary(results[top]["summary"][key], key) + u
        fb = fmt_summary(results[bottom]["summary"][key], key) + u
    p = prob_greater(results[top]["samples"][key], results[bottom]["samples"][key])
    return (f"{label} was {higher_word} for {tl} ({ft}) and {lower_word} for {bl} ({fb}), "
            f"{_robust(p)}.")


def write_summary(results, conditions, molecules, settings, focus=None) -> dict:
    order = [c["id"] for c in conditions if c["id"] in results]
    by_id = {c["id"]: c for c in conditions}
    scaffold = _groups(results, [c for c in order if results[c]["has_scaffold"]], by_id)
    drug = [(c, f"**{results[c]['name']}**") for c in order if "D_rel" in results[c]["summary"]]
    bact = [c for c in order if results[c]["challenge"] == "bacterium"]
    S = {}

    names = ", ".join(f"**{results[c]['name']}** ({results[c]['role'].lower()})" for c in order)
    S["Overview"] = (
        f"{len(order)} conditions were evaluated at pH {settings['pH']:g} and "
        f"{settings['T_C']:g} °C: {names}. Each value is the median of "
        f"{settings['n_samples']:,} Monte Carlo samples with a 95 % interval in brackets, "
        "propagating uncertainty in pKa, concentrations, densities, chain and drug sizes, "
        "scaffold thickness and pore size.")

    if focus and focus[0] in results and focus[1] in results and focus[0] != focus[1]:
        t, r = results[focus[0]], results[focus[1]]
        lines = []
        for k in METRICS:
            if k in t["summary"] and k in r["summary"] and k != "conc_total":
                mt, mr = t["summary"][k]["median"], r["summary"][k]["median"]
                if not (np.isfinite(mt) and np.isfinite(mr)) or np.isclose(mt, mr):
                    continue
                label = METRICS[k][0]
                rel = (mt - mr) / abs(mr) * 100 if mr != 0 else np.nan
                p = prob_greater(t["samples"][k], r["samples"][k])
                if mt < 0 and mr < 0:
                    word = "less negative" if mt > mr else "more negative"
                else:
                    word = "higher" if mt > mr else "lower"
                pct = f"{abs(rel):.0f} % " if np.isfinite(rel) else ""
                if k.endswith("_h"):
                    vals = f"{fmt_time_h(t['summary'][k])} vs {fmt_time_h(r['summary'][k])}"
                else:
                    vals = f"{fmt_summary(t['summary'][k], k)} vs {fmt_summary(r['summary'][k], k)}"
                lines.append(f"- {label}: {pct}{word} ({vals}); {_robust(p)}.")
        if lines:
            S["Primary comparison"] = (f"**{t['name']}** compared with **{r['name']}**:\n\n"
                                       + "\n".join(lines))

    if scaffold:
        parts = [_rank_sentence(results, scaffold, "polar_sasa_fraction"),
                 _rank_sentence(results, scaffold, "polar_per_kDa")]
        S["Hydration capacity"] = (
            " ".join(p for p in parts if p) +
            " A larger polar surface and more N/O atoms per mass indicate more sites for "
            "hydrogen bonding with water, a proxy for water uptake.")
        parts = [_rank_sentence(results, scaffold, "cationic_per_kDa"),
                 _rank_sentence(results, scaffold, "net_charge_per_kDa", "most positive",
                                "most negative")]
        S["Charge and antibacterial proxy"] = (
            " ".join(p for p in parts if p) +
            " Protonated amines (for example in chitosan) carry positive charge that can "
            "interact with negatively charged bacterial membranes; higher cationic charge "
            "density is a proxy for contact antibacterial activity, not a measurement of it.")
        parts = [_rank_sentence(results, scaffold, "phi")]
        S["Network density"] = (" ".join(p for p in parts if p) +
                                " Denser networks obstruct diffusing molecules more strongly.")

    if drug:
        parts = [_rank_sentence(results, drug, "D_rel", "highest (least hindered)",
                                "lowest (most hindered)"),
                 _rank_sentence(results, drug, "t50_h", "longest", "shortest")]
        S["Drug transport and release"] = (
            " ".join(p for p in parts if p) +
            f" Release assumes Fickian diffusion from a {settings['thickness_mm']:g} mm "
            "slab with no binding, so absolute times are lower bounds; the ranking between "
            "conditions is the useful output.")

    if bact:
        est = [c for c in bact if "pore_pass_fraction" in results[c]["summary"]]
        txt = []
        for c in est:
            r = results[c]
            b = r.get("bacterium", {})
            src = (r.get("pores") or {}).get("source")
            txt.append(f"**{r['name']}**: {fmt_summary(r['summary']['pore_pass_fraction'], 'pore_pass_fraction')} "
                       f"of pores are wider than the bacterium ({b.get('species', 'custom')}, "
                       f"width {b.get('width_um', 0):g} µm)"
                       + (f"; pore sizes from {src}." if src else "."))
        missing = [results[c]["name"] for c in bact if c not in est]
        if missing:
            txt.append("No pore-size input for " + ", ".join(f"**{m}**" for m in missing) +
                       ", so bacterial passage was not estimated; see the cationic charge "
                       "proxy above. Pore sizes can be entered from literature for a "
                       "similar fabrication method, or from SEM images later.")
        S["Bacterial challenge"] = " ".join(txt)

    notes = sorted({n for c in order for n in results[c]["notes"]})
    S["Limitations"] = "\n".join(f"- {x}" for x in LIMITATIONS + notes)
    S["Methods"] = methods_text(molecules, settings)
    return S


def methods_text(molecules, settings) -> str:
    return (
        "Structures were read with RDKit (compounds) or residue-based parsing (proteins). "
        "Solvent-accessible surface area was computed with the Shrake–Rupley algorithm "
        "(probe 1.4 Å, 100 points per atom) on heavy atoms with united-atom radii; molecular "
        "volume by grid counting of the union of atomic spheres. Ionizable groups were "
        "identified by SMARTS patterns (compounds) or residue type (proteins) and charges "
        "computed with the Henderson–Hasselbalch equation. Polymer volume fractions were "
        "computed from concentration and density. Drug diffusivity in water was estimated "
        "with the Stokes–Einstein equation (water viscosity from the Vogel equation) and "
        "hindrance with the Ogston fibre-network model; release from a slab releasing from "
        "both faces followed Crank's solution of Fick's second law. Bacterial passage used "
        "the fraction of a log-normal pore-size distribution wider than the bacterial cell. "
        f"Uncertainty was propagated by Monte Carlo sampling (n = {settings['n_samples']:,}; "
        f"pKa SD {settings['sd_pKa']:g}; relative SDs: concentration {settings['rel_conc']:.0%}, "
        f"density {settings['rel_density']:.0%}, chain radius {settings['rel_chain']:.0%}, "
        f"drug radius {settings['rel_drug']:.0%}, thickness {settings['rel_thickness']:.0%}, "
        f"pore size {settings['rel_pore']:.0%}; seed {settings['seed']}).")


def md_to_html(text: str) -> str:
    """Tiny markdown subset: **bold**, '- ' bullets, paragraphs."""
    out, items = [], []
    for para in text.split("\n"):
        line = html.escape(para)
        while "**" in line:
            line = line.replace("**", "<strong>", 1).replace("**", "</strong>", 1)
        if para.startswith("- "):
            items.append(f"<li>{line[2:]}</li>")
            continue
        if items:
            out.append("<ul>" + "".join(items) + "</ul>")
            items = []
        if line.strip():
            out.append(f"<p>{line}</p>")
    if items:
        out.append("<ul>" + "".join(items) + "</ul>")
    return "\n".join(out)


CSS = """
body{font-family:Inter,Segoe UI,Helvetica,Arial,sans-serif;color:#102a43;max-width:980px;
margin:40px auto;padding:0 24px;line-height:1.55}
h1{font-size:28px;margin:0}h2{font-size:20px;border-bottom:2px solid #0072B2;padding-bottom:4px;
margin-top:36px}.meta{color:#627d98;margin:6px 0 24px}
.hero{background:linear-gradient(135deg,#0b3c5d,#0072B2);color:#fff;padding:22px 26px;
border-radius:14px}.hero .meta{color:#d9e8f5}
table{border-collapse:collapse;width:100%;font-size:13px;margin:12px 0}
th,td{border:1px solid #d9e2ec;padding:6px 8px;text-align:left;vertical-align:top}
th{background:#f0f4f8}figure{margin:18px 0;text-align:center}
figure img{max-width:100%;border:1px solid #e5eaf0;border-radius:8px}
figcaption{font-size:13px;color:#486581;margin-top:6px}
.note{background:#fff8e6;border-left:4px solid #E69F00;padding:10px 14px;border-radius:6px}
ol li{font-size:13px;margin-bottom:4px}
@media print{body{margin:0}.hero{-webkit-print-color-adjust:exact;print-color-adjust:exact}}
"""


def html_report(project, author, summary: dict, tables: dict, figures: list,
                library: pd.DataFrame) -> str:
    """figures: list of (title, caption, png bytes)."""
    when = dt.datetime.now().strftime("%d %B %Y, %H:%M")
    parts = [f"<!doctype html><html><head><meta charset='utf-8'><title>{html.escape(project)}"
             f"</title><style>{CSS}</style></head><body>",
             f"<div class='hero'><h1>{html.escape(project)}</h1><div class='meta'>"
             f"{html.escape(author) + ' · ' if author else ''}Generated {when} with "
             f"BioScaff Studio</div></div>",
             "<p class='note'>Model-based screening estimates, not simulation or experimental "
             "results. See Limitations.</p>"]
    for sec in ["Overview", "Primary comparison"]:
        if sec in summary:
            parts.append(f"<h2>{sec}</h2>{md_to_html(summary[sec])}")
    parts.append("<h2>Molecules</h2>" + library.to_html(index=False, escape=True))
    for name, df in tables.items():
        parts.append(f"<h2>{html.escape(name)}</h2>" + df.to_html(index=False, escape=True))
    for sec in ["Hydration capacity", "Charge and antibacterial proxy", "Network density",
                "Drug transport and release", "Bacterial challenge"]:
        if sec in summary:
            parts.append(f"<h2>{sec}</h2>{md_to_html(summary[sec])}")
    if figures:
        parts.append("<h2>Figures</h2>")
        for i, (title, cap, png) in enumerate(figures, 1):
            b64 = base64.b64encode(png).decode()
            parts.append(f"<figure><img src='data:image/png;base64,{b64}' alt='{html.escape(title)}'>"
                         f"<figcaption><strong>Figure {i}. {html.escape(title)}.</strong> "
                         f"{html.escape(cap)}</figcaption></figure>")
    for sec in ["Methods", "Limitations"]:
        if sec in summary:
            parts.append(f"<h2>{sec}</h2>{md_to_html(summary[sec])}")
    parts.append("<h2>References</h2><ol>" +
                 "".join(f"<li>{html.escape(r)}</li>" for r in REFERENCES) + "</ol>")
    parts.append("</body></html>")
    return "\n".join(parts)
