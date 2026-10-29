# BioScaff Studio

A Streamlit app for screening biomaterial scaffold designs on an ordinary CPU.
Build a library of molecules, define any number of conditions from them, run
in seconds, and get comparison figures, uncertainty intervals and a written
report. It works for any scaffold project: nothing is hard-coded to a
particular polymer.

```
① Molecule library  →  ② Conditions  →  ③ Run  →  ④ Results  →  ⑤ Report
```

A second page, **MD trajectory analysis**, analyses systems simulated
elsewhere (the later GPU phase).

## What you put in

| Step | Input |
|---|---|
| Library | One entry per molecule: PDB, SDF/MOL or MOL2 files (several at once), a PubChem name/CID, an RCSB PDB ID, or a SMILES string. PDB files with amino-acid residues are treated as proteins; everything else as compounds/oligomers. Water is built in, so don't upload it. |
| Verify | Rotatable 3D view (sticks, spheres, cartoon, surface; save PNG), 2D drawing, automatic checks (hydrogens, 3D coordinates, fragments, missing residues, bond orders), properties, ionizable groups with editable pKa. Tick "I have checked this structure". |
| Conditions | As many as you like. Each has a name, a type (test, positive control, negative control, baseline), scaffold components picked from the library with % w/v, and a challenge: none, a drug from the library, or a bacterium (preset size or custom). |
| Settings | pH, temperature, scaffold thickness. Uncertainty settings have defaults. |

## What it computes

All values come with a 95 % interval from Monte Carlo sampling of the
uncertain inputs.

| Result | Model |
|---|---|
| Hydration capacity | Polar fraction of the solvent-accessible surface (Shrake–Rupley), polar N + O atoms per kDa |
| Charge and antibacterial proxy | Henderson–Hasselbalch over all ionizable groups: net, cationic and anionic charge per kDa; charge-versus-pH curves |
| Network density and sieving | Polymer volume fraction; Ogston fibre-network model for how strongly the scaffold slows a solute of any size |
| Drug transport and release | Stokes–Einstein diffusivity × Ogston hindrance; Fickian release from a slab (Crank) with t50 and t90 |
| Bacterial challenge | Without pore data: cationic charge as the antibacterial proxy. With a pore-size estimate (literature now, SEM later): share of pores wider than the bacterium |

These are model-based screening estimates, not molecular dynamics and not
measurements. The app states this in the report, with the limitations.

## Outputs

* Figures as PNG (300–1200 dpi), SVG and PDF, in single-column, double-column
  or poster sizes.
* Results table with intervals, raw Monte Carlo samples and curves as CSV.
* An auto-written summary: every comparison with its robustness ("holds in
  97 % of samples"), plus methods, limitations and references.
* An HTML report (open it in a browser and print to PDF) and one zip with
  everything, including the structures and the project file.
* **Save project** stores the library and conditions in a `.json` file to
  reload later. Hosted sessions are temporary, so save often.

## Run locally

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python tests/selftest.py        # every line should say PASS
streamlit run app.py
```

## Host free on Streamlit Community Cloud

1. Put this folder in a GitHub repository (keep `.streamlit/`, `core/`,
   `pages/`).
2. On share.streamlit.io choose **Create app**, pick the repository and
   `app.py`, and under *Advanced settings* choose Python 3.11 or 3.12.
3. Deploy. The first build takes a few minutes.

## Project layout

```
app.py                         pipeline interface
pages/2_MD_trajectory_analysis.py   analysis of simulated systems (GPU phase)
core/molecules.py              reading, checks, properties, ionizable groups
core/models.py                 condition models and Monte Carlo
core/plots.py                  pipeline figures
core/report.py                 written summary and HTML report
core/viewer.py                 3D viewer (3Dmol.js)
core/md.py, core/md_figures.py trajectory analysis and figures
tests/selftest.py              known-answer checks
```

## Getting good inputs

* **Proteins:** download from RCSB by PDB ID, e.g. a collagen-like peptide.
* **Polysaccharides and synthetic polymers:** use a short oligomer (3–10
  repeat units) as SDF from PubChem, or build it from SMILES. Stereochemistry
  matters for 3D shape; use isomeric SMILES when you can.
* **Drugs:** PubChem by name.
* **pKa values:** defaults are generic group values. Replace them with
  literature values in *Model inputs* for the molecules that matter (for
  example the drug).
* **Pore sizes:** without SEM, use values reported for the same fabrication
  method (freeze-drying, electrospinning) and enter the citation.
