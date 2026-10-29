"""
Molecule library: reading structures, verification checks and properties.

Inputs: PDB (proteins or compounds), SDF/MOL, MOL2, SMILES, PubChem name/CID,
RCSB PDB ID. Proteins are handled residue by residue; compounds go through
RDKit. Geometry (surface area, volume, size) uses heavy atoms with
united-atom radii so that structures with and without hydrogens are treated
the same way.
"""
from __future__ import annotations

import re
import urllib.parse
import urllib.request

import numpy as np
from scipy.spatial import cKDTree

from core.md import guess_elements, sasa_per_atom

AMINO_ACIDS = {"ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
               "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
               "HYP", "HID", "HIE", "HIP", "HSD", "HSE", "HSP", "CYX", "ASH", "GLH",
               "LYN", "MSE", "SEP", "TPO", "PTR", "NLE"}
WATER_RES = {"HOH", "WAT", "SOL", "TIP3", "TIP3P", "H2O", "DOD", "TIP4", "SPC", "T3P"}
ALIAS = {"HID": "HIS", "HIE": "HIS", "HIP": "HIS", "HSD": "HIS", "HSE": "HIS",
         "HSP": "HIS", "ASH": "ASP", "GLH": "GLU", "LYN": "LYS"}

# Average residue masses in a chain (Da); a chain adds one water (18.015).
RESIDUE_MASS = {
    "ALA": 71.079, "ARG": 156.188, "ASN": 114.104, "ASP": 115.089, "CYS": 103.139,
    "CYX": 103.139, "GLN": 128.131, "GLU": 129.116, "GLY": 57.052, "HIS": 137.141,
    "ILE": 113.160, "LEU": 113.160, "LYS": 128.174, "MET": 131.193, "PHE": 147.177,
    "PRO": 97.117, "SER": 87.078, "THR": 101.105, "TRP": 186.213, "TYR": 163.176,
    "VAL": 99.133, "HYP": 113.116, "MSE": 178.09, "NLE": 113.160,
}
ATOMIC_MASS = {"H": 1.008, "C": 12.011, "N": 14.007, "O": 15.999, "S": 32.06,
               "P": 30.974, "F": 18.998, "CL": 35.45, "BR": 79.904, "I": 126.90,
               "NA": 22.990, "K": 39.098, "CA": 40.078, "MG": 24.305, "ZN": 65.38,
               "FE": 55.845, "SE": 78.971, "SI": 28.085, "B": 10.81}

# United-atom radii (Å) for heavy atoms, simplified from Tsai et al. (1999).
UNITED_RADII = {"C": 1.88, "N": 1.64, "O": 1.46, "S": 1.77, "P": 1.80, "SE": 1.90,
                "F": 1.47, "CL": 1.75, "BR": 1.85, "I": 1.98, "B": 1.80, "SI": 2.10}
DEFAULT_RADIUS = 1.80
KNOWN_ELEMENTS = set(UNITED_RADII) | {"H", "NA", "K", "CA", "MG", "ZN", "FE"}

ROLES = ["Scaffold component", "Drug / payload", "Solvent", "Other"]
PACKING_COEFFICIENT = 0.68   # typical for organic solids and amorphous polymers

# ---------------------------------------------------------------- ionizable groups
# Generic group pKa values (approximate, aqueous, 25 °C). Users can override
# them per molecule with literature values.
PROTEIN_GROUPS = [
    ("Asp side chain", "ASP", "acid", 3.9),
    ("Glu side chain", "GLU", "acid", 4.1),
    ("His side chain", "HIS", "base", 6.0),
    ("Cys side chain", "CYS", "acid", 8.3),
    ("Tyr side chain", "TYR", "acid", 10.1),
    ("Lys side chain", "LYS", "base", 10.5),
    ("Arg side chain", "ARG", "base", 12.5),
]
TERMINI = [("N-terminus", "base", 8.0), ("C-terminus", "acid", 3.1)]

# (name, SMARTS whose first atom is the ionizable centre, kind, pKa).
# Order matters: a centre matched by an earlier pattern is not counted again.
COMPOUND_GROUPS = [
    ("Sulfate / sulfonic acid", "[SX4](=O)(=O)[OX2H1]", "acid", -2.0),
    ("Phosphate (first OH)", "[PX4](=O)[OX2H1]", "acid", 2.0),
    ("Uronic acid carboxyl", "[CX3](=O)([OX2H1])[CX4;R]@[OX2;R]", "acid", 3.0),
    ("Carboxylic acid", "[CX3](=O)[OX2H1]", "acid", 4.0),
    ("Thiol", "[SX2H1]", "acid", 8.3),
    ("Phenol", "[OX2H1]c", "acid", 10.0),
    ("Amino-sugar amine (glucosamine-type)",
     "[NX3;H2;!$(N[C,S,P]=[O,S,N]);!$(Na)][CX4;R]@[CX4;R]@[OX2;R]", "base", 6.5),
    ("Guanidine", "[CX3](=[NX2])([NX3])[NX3]", "base", 12.5),
    ("Amidine", "[CX3;!$(C(N)(N)=N)](=[NX2;!$(N~[O,N])])([#6])[NX3;H2,H1]", "base", 11.0),
    ("Imidazole", "[nX2;r5]:[c;r5]:[nX3;H1;r5]", "base", 6.0),
    ("Primary aliphatic amine",
     "[NX3;H2;!$(N[C,S,P]=[O,S,N]);!$(Na);!$(N[#7,#8])][CX4]", "base", 10.6),
    ("Secondary aliphatic amine",
     "[NX3;H1;!$(N[C,S,P]=[O,S,N]);!$(Na);!$(N[#7,#8])]([CX4])[CX4]", "base", 10.5),
    ("Tertiary aliphatic amine",
     "[NX3;H0;!$(N[C,S,P]=[O,S,N]);!$(Na);!$(N[#7,#8])]([CX4])([CX4])[CX4]", "base", 9.8),
    ("Aromatic amine (aniline-type)", "[NX3;H2,H1;!$(N[C,S]=[O,S,N])]c", "base", 4.6),
    ("Pyridine-type N", "[nX2;r6]", "base", 5.2),
]


def rdkit_available() -> bool:
    try:
        import rdkit  # noqa: F401
        return True
    except ImportError:
        return False


# ================================================================ reading

def detect_format(filename: str) -> str:
    ext = filename.rsplit(".", 1)[-1].lower()
    return {"ent": "pdb", "mol": "sdf", "sd": "sdf"}.get(ext, ext)


def parse_pdb_atoms(text: str) -> dict:
    """First model, altloc A, water removed. Returns arrays and cleaned text."""
    keep, names, resn, chains, resseq, xyz, elem_col = [], [], [], [], [], [], []
    altlocs = False
    seen_model = False
    for line in text.splitlines():
        rec = line[:6].strip()
        if rec == "MODEL":
            if seen_model:
                break
            seen_model = True
            continue
        if rec == "ENDMDL":
            break
        if rec not in ("ATOM", "HETATM"):
            continue
        alt = line[16:17].strip()
        if alt:
            altlocs = True
            if alt not in ("A", "1"):
                continue
        rn = line[17:21].strip()
        if rn in WATER_RES:
            continue
        try:
            x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
        except ValueError:
            continue
        try:
            rs = int(line[22:26])
        except ValueError:
            rs = 0
        keep.append(line)
        names.append(line[12:16].strip())
        resn.append(rn)
        chains.append(line[21:22].strip() or "_")
        resseq.append(rs)
        xyz.append((x, y, z))
        elem_col.append(line[76:78].strip() if len(line) >= 78 else "")
    if not keep:
        raise ValueError("No atoms found in the PDB file.")
    el = guess_elements(names, resn, elem_col if all(elem_col) else None)
    return {"names": np.array(names), "resnames": np.array(resn), "chains": np.array(chains),
            "resseq": np.array(resseq), "coords": np.array(xyz, float),
            "elements": el, "altlocs": altlocs,
            "text": "\n".join(keep) + "\nEND\n"}


def is_protein(atoms: dict) -> bool:
    aa = np.isin(atoms["resnames"], list(AMINO_ACIDS))
    n_res = len(set(zip(atoms["chains"][aa], atoms["resseq"][aa])))
    return aa.mean() >= 0.5 and n_res >= 2


def _parse_molblock_light(block: str) -> dict:
    """Minimal V2000 reader used only when RDKit is not installed."""
    lines = block.splitlines()
    counts = lines[3]
    n_atoms, n_bonds = int(counts[0:3]), int(counts[3:6])
    xyz, el = [], []
    for ln in lines[4:4 + n_atoms]:
        xyz.append((float(ln[0:10]), float(ln[10:20]), float(ln[20:30])))
        el.append(ln[31:34].strip().upper())
    return {"coords": np.array(xyz), "elements": np.array(el), "n_bonds": n_bonds}


def _largest_fragment(mol):
    from rdkit import Chem
    frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
    if len(frags) <= 1:
        return mol, 1
    best = max(frags, key=lambda m: m.GetNumHeavyAtoms())
    return best, len(frags)


def _embed_3d(mol, seed=42):
    from rdkit import Chem
    from rdkit.Chem import AllChem
    m = Chem.AddHs(Chem.RemoveHs(mol))
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    if AllChem.EmbedMolecule(m, params) != 0:
        params.useRandomCoords = True
        if AllChem.EmbedMolecule(m, params) != 0:
            raise ValueError("Could not generate 3D coordinates for this molecule.")
    try:
        AllChem.MMFFOptimizeMolecule(m, maxIters=2000)
    except Exception:  # noqa: BLE001
        pass
    return m


def mol_from_smiles(smiles: str, seed=42):
    from rdkit import Chem
    m = Chem.MolFromSmiles(smiles.strip())
    if m is None:
        raise ValueError("RDKit could not parse this SMILES string.")
    return _embed_3d(m, seed)


def read_compound(text: str, fmt: str, smiles_template: str | None = None):
    """Return (rdkit Mol with explicit H and 3D coordinates, list of checks)."""
    from rdkit import Chem
    from rdkit.Chem import AllChem
    checks = []
    if fmt == "sdf":
        n_rec = max(1, len([b for b in text.split("$$$$") if b.strip()]))
        mol = Chem.MolFromMolBlock(text.split("$$$$")[0], removeHs=False)
        if n_rec > 1:
            checks.append(("warn", f"File holds {n_rec} molecules; only the first is used."))
    elif fmt == "mol2":
        mol = Chem.MolFromMol2Block(text, removeHs=False)
    elif fmt == "pdb":
        mol = Chem.MolFromPDBBlock(text, removeHs=False, proximityBonding=True)
        if smiles_template:
            tmpl = Chem.MolFromSmiles(smiles_template)
            if tmpl is None:
                raise ValueError("The SMILES template could not be parsed.")
            mol = AllChem.AssignBondOrdersFromTemplate(tmpl, Chem.RemoveHs(mol))
            mol = Chem.AddHs(mol, addCoords=True)
            checks.append(("ok", "Bond orders assigned from the SMILES template."))
        else:
            checks.append(("warn", "PDB files carry no bond orders, so double bonds "
                                   "(C=O, carboxylates, amides) may be missed. Add the "
                                   "molecule's SMILES below, or upload SDF/MOL2."))
    else:
        raise ValueError(f"Unsupported format: {fmt}")
    if mol is None:
        raise ValueError("RDKit could not read this structure (format or valence problem).")
    mol, n_frag = _largest_fragment(mol)
    if n_frag > 1:
        checks.append(("warn", f"{n_frag} separate fragments found (e.g. counter-ions "
                               "or salts); the largest was kept."))
    pos = mol.GetConformer().GetPositions() if mol.GetNumConformers() else None
    if pos is None or np.allclose(pos[:, 2], 0):
        mol = _embed_3d(mol)
        checks.append(("warn", "No 3D coordinates in the file; 3D structure generated "
                               "with ETKDG + MMFF."))
    elif not any(a.GetSymbol() == "H" for a in mol.GetAtoms()):
        mol = Chem.AddHs(mol, addCoords=True)
        checks.append(("warn", "No hydrogens in the file; hydrogens added."))
    else:
        checks.append(("ok", "3D coordinates with hydrogens."))
    return mol, checks


# ================================================================ geometry

def united_radii(elements) -> np.ndarray:
    return np.array([UNITED_RADII.get(e, DEFAULT_RADIUS) for e in elements], float)


def union_volume(coords, radii, max_points=2_000_000, chunk=250_000) -> float:
    """Volume (Å³) of the union of atomic spheres, by grid counting."""
    coords = np.asarray(coords, float)
    rmax = radii.max()
    lo, hi = coords.min(0) - rmax, coords.max(0) + rmax
    ext = hi - lo
    spacing = max(0.2, (np.prod(ext) / max_points) ** (1 / 3))
    n = np.maximum(np.ceil(ext / spacing).astype(int), 1)
    axes = [lo[i] + (np.arange(n[i]) + 0.5) * spacing for i in range(3)]
    tree = cKDTree(coords)
    k = min(8, len(coords))
    inside = 0
    gx, gy, gz = np.meshgrid(*axes, indexing="ij")
    grid = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])
    for s in range(0, len(grid), chunk):
        d, j = tree.query(grid[s:s + chunk], k=k, distance_upper_bound=rmax)
        if k == 1:
            d, j = d[:, None], j[:, None]
        valid = np.isfinite(d)
        jj = np.where(valid, j, 0)
        inside += int((valid & (d < radii[jj])).any(1).sum())
    return inside * spacing ** 3


def geometry_props(coords, elements, n_points=100) -> dict:
    """Heavy-atom geometric properties shared by proteins and compounds."""
    elements = np.asarray(elements)
    heavy = elements != "H"
    xyz = np.asarray(coords, float)[heavy]
    el = elements[heavy]
    r = united_radii(el)
    sasa = sasa_per_atom(xyz, r, probe=1.4, n_points=n_points)
    polar = np.isin(el, ["N", "O"])
    vol = union_volume(xyz, r)
    c = xyz - xyz.mean(0)
    rg = float(np.sqrt((c ** 2).sum(1).mean()))
    evals, evecs = np.linalg.eigh(np.cov(c.T)) if len(c) > 2 else (np.ones(3), np.eye(3))
    proj = c @ evecs[:, -1]
    length = float(proj.max() - proj.min() + 2 * r.mean())
    return {
        "heavy_atoms": int(heavy.sum()),
        "sasa_A2": float(sasa.sum()),
        "polar_sasa_fraction": float(sasa[polar].sum() / max(sasa.sum(), 1e-9)),
        "vdw_volume_A3": float(vol),
        "radius_of_gyration_A": rg,
        "length_A": length,
        "N_plus_O": int(polar.sum()),
        "chain_radius_nm": float(np.sqrt(vol / (np.pi * length)) / 10),
        "hydrodynamic_radius_nm": float((3 * vol / (4 * np.pi)) ** (1 / 3) / 10),
    }


# ================================================================ chemistry

def protein_groups(atoms: dict) -> list:
    res = {}
    for ch, rs, rn in zip(atoms["chains"], atoms["resseq"], atoms["resnames"]):
        res.setdefault((ch, rs), rn)
    names = [ALIAS.get(v, v) for v in res.values()]
    groups = []
    for label, rn, kind, pka in PROTEIN_GROUPS:
        n = names.count(rn)
        if n:
            groups.append({"group": label, "kind": kind, "count": n, "pKa": pka})
    aa_chains = {ch for (ch, _), rn in res.items() if rn in AMINO_ACIDS}
    for label, kind, pka in TERMINI:
        groups.append({"group": label, "kind": kind, "count": len(aa_chains), "pKa": pka})
    return groups


def compound_groups(mol) -> tuple[list, float]:
    """Ionizable groups found by SMARTS on the neutralised molecule.

    Returns (groups, permanent charge), where the permanent charge counts
    charges the neutraliser cannot remove (e.g. quaternary ammonium).
    """
    from rdkit import Chem
    from rdkit.Chem.MolStandardize import rdMolStandardize
    m = rdMolStandardize.Uncharger().uncharge(Chem.RemoveHs(mol))
    permanent = float(sum(a.GetFormalCharge() for a in m.GetAtoms()))
    used, groups = set(), []
    for label, smarts, kind, pka in COMPOUND_GROUPS:
        patt = Chem.MolFromSmarts(smarts)
        n = 0
        for match in m.GetSubstructMatches(patt):
            if match[0] in used:
                continue
            used.add(match[0])
            n += 1
        if n:
            groups.append({"group": label, "kind": kind, "count": n, "pKa": pka})
    return groups, permanent


def charge_at_pH(groups, pH, permanent=0.0, pka_shift=None):
    """Henderson–Hasselbalch net, cationic and anionic charge (elementary charges).

    pH may be an array. pka_shift: optional array (len(groups), ...) added to pKa.
    """
    pH = np.asarray(pH, float)
    pos = np.full(np.broadcast(pH).shape, max(permanent, 0.0))
    neg = np.full_like(pos, min(permanent, 0.0))
    for i, g in enumerate(groups):
        pka = g["pKa"] + (0 if pka_shift is None else pka_shift[i])
        if g["kind"] == "base":
            pos = pos + g["count"] / (1 + 10 ** (pH - pka))
        else:
            neg = neg - g["count"] / (1 + 10 ** (pka - pH))
    return pos + neg, pos, neg


def default_density(kind: str, mw: float, vdw_volume_A3: float) -> float:
    """g/cm³. Proteins 1.35; compounds estimated from vdW volume / packing."""
    if kind == "protein":
        return 1.35
    v_cm3_per_mol = vdw_volume_A3 * 1e-24 * 6.02214076e23 / PACKING_COEFFICIENT
    return float(np.clip(mw / v_cm3_per_mol, 0.8, 2.5))


def guess_role(kind: str, formula: str, mw: float) -> str:
    if formula in ("H2O", "OH2"):
        return "Solvent"
    if kind == "protein" or mw >= 1000:
        return "Scaffold component"
    return "Drug / payload"


# ================================================================ entry point

def build_entry(name: str, text: str, fmt: str, source: str,
                smiles_template: str | None = None) -> dict:
    """Read a structure and compute everything the library shows."""
    checks = []
    if fmt == "smiles":
        mol = mol_from_smiles(text)
        fmt_read = "smiles"
        checks.append(("ok", "3D structure generated from SMILES (ETKDG + MMFF)."))
        atoms = None
    elif fmt == "pdb":
        atoms = parse_pdb_atoms(text)
        mol = None
        if atoms["altlocs"]:
            checks.append(("warn", "Alternate locations found; only altloc A was kept."))
        if not is_protein(atoms):
            atoms = None
            if not rdkit_available():
                raise RuntimeError("RDKit is needed for non-protein PDB files.")
            mol, c2 = read_compound(text, "pdb", smiles_template)
            checks += c2
        fmt_read = "pdb"
    elif fmt in ("sdf", "mol2"):
        atoms = None
        if rdkit_available():
            mol, c2 = read_compound(text, fmt)
            checks += c2
        else:
            mol = None
        fmt_read = fmt
    else:
        raise ValueError(f"Unsupported file type '{fmt}'. Use PDB, SDF, MOL or MOL2.")

    entry = {"name": name, "source": source, "fmt": fmt_read, "raw": text,
             "smiles_template": smiles_template or ""}

    if atoms is not None:                                           # ---- protein
        el = atoms["elements"]
        res = {}
        for ch, rs, rn in zip(atoms["chains"], atoms["resseq"], atoms["resnames"]):
            res.setdefault((ch, rs), rn)
        chains = sorted({k[0] for k in res})
        unknown = sorted({rn for rn in res.values()
                          if rn not in AMINO_ACIDS and rn not in WATER_RES})
        mw = sum(RESIDUE_MASS.get(ALIAS.get(rn, rn), 110.0) for rn in res.values()
                 if rn in AMINO_ACIDS) + 18.015 * len(chains)
        mw += sum(ATOMIC_MASS.get(e, 12.0) for e, rn in zip(el, atoms["resnames"])
                  if rn not in AMINO_ACIDS and e != "H")
        gaps = 0
        for ch in chains:
            nums = sorted(k[1] for k in res if k[0] == ch)
            gaps += int(np.sum(np.diff(nums) > 1)) if len(nums) > 1 else 0
        has_h = bool((el == "H").any())
        checks.insert(0, ("ok", f"Protein: {len(res)} residues in {len(chains)} chain(s)."))
        if not has_h:
            checks.append(("info", "No hydrogens (normal for crystal structures). "
                                   "Properties use heavy atoms with united-atom radii."))
        if gaps:
            checks.append(("warn", f"{gaps} gap(s) in residue numbering: possible "
                                   "missing residues."))
        if unknown:
            checks.append(("warn", "Non-standard residues: " + ", ".join(unknown[:8])))
        counts = {}
        for e in el:
            counts[e] = counts.get(e, 0) + 1
        formula = "".join(f"{e.title()}{counts[e] if counts[e] > 1 else ''}"
                          for e in sorted(counts, key=lambda x: (x != "C", x != "H", x)))
        entry.update(kind="protein", display=atoms["text"], display_fmt="pdb",
                     formula=formula + ("" if has_h else " (heavy atoms)"), mw=float(mw),
                     n_atoms=int(len(el)), groups=protein_groups(atoms), permanent=0.0,
                     sequence_chains=len(chains), n_residues=len(res))
        coords, elements = atoms["coords"], el
        extra = {}
    elif mol is not None:                                           # ---- compound
        from rdkit import Chem
        from rdkit.Chem import Crippen, Descriptors, rdMolDescriptors
        elements = np.array([a.GetSymbol().upper() for a in mol.GetAtoms()])
        coords = mol.GetConformer().GetPositions()
        groups, permanent = compound_groups(mol)
        formula = rdMolDescriptors.CalcMolFormula(mol)
        molblock = Chem.MolToMolBlock(mol)
        big = mol.GetNumAtoms() > 999                  # V3000 territory: show as PDB
        entry.update(kind="compound", molblock=molblock,
                     display=Chem.MolToPDBBlock(mol) if big else molblock,
                     display_fmt="pdb" if big else "sdf",
                     formula=formula, mw=float(Descriptors.MolWt(mol)),
                     n_atoms=mol.GetNumAtoms(), groups=groups, permanent=permanent,
                     smiles=Chem.MolToSmiles(Chem.RemoveHs(mol)))
        extra = {"TPSA_A2": rdMolDescriptors.CalcTPSA(mol),
                 "HBD": rdMolDescriptors.CalcNumHBD(mol),
                 "HBA": rdMolDescriptors.CalcNumHBA(mol),
                 "logP_Crippen": Crippen.MolLogP(mol),
                 "rotatable_bonds": rdMolDescriptors.CalcNumRotatableBonds(mol)}
        if not groups and permanent == 0:
            checks.append(("info", "No ionizable groups detected: neutral at all pH."))
    else:                                                           # ---- no RDKit
        light = _parse_molblock_light(text.split("$$$$")[0]) if fmt == "sdf" else None
        if light is None:
            raise RuntimeError("RDKit is required to read MOL2 files.")
        elements, coords = light["elements"], light["coords"]
        counts = {}
        for e in elements:
            counts[e] = counts.get(e, 0) + 1
        formula = "".join(f"{e.title()}{counts[e] if counts[e] > 1 else ''}"
                          for e in sorted(counts, key=lambda x: (x != "C", x != "H", x)))
        entry.update(kind="compound", display=text.split("$$$$")[0] + "$$$$\n",
                     molblock=text.split("$$$$")[0],
                     display_fmt="sdf", formula=formula,
                     mw=float(sum(ATOMIC_MASS.get(e, 12.0) for e in elements)),
                     n_atoms=int(len(elements)), groups=[], permanent=0.0)
        extra = {}
        checks.append(("warn", "RDKit not installed: ionizable groups and descriptors "
                               "were skipped."))

    odd = sorted(set(elements) - KNOWN_ELEMENTS)
    if odd:
        checks.append(("warn", "Unusual elements (default radius used): " + ", ".join(odd)))
    if entry["n_atoms"] > 30000:
        checks.append(("warn", "Large structure: property calculation may take a while."))
    geo = geometry_props(coords, elements)
    entry["props"] = {**geo, **extra}
    entry["density"] = default_density(entry["kind"], entry["mw"], geo["vdw_volume_A3"])
    entry["chain_radius_nm"] = round(geo["chain_radius_nm"], 3)
    entry["hydrodynamic_radius_nm"] = round(geo["hydrodynamic_radius_nm"], 3)
    entry["role"] = guess_role(entry["kind"], entry["formula"], entry["mw"])
    if entry["role"] == "Solvent":
        checks.append(("info", "Water is modelled implicitly as the solvent; this entry "
                               "is kept for reference and not added to conditions."))
    entry["checks"] = checks
    entry["verified"] = False
    return entry


# ================================================================ fetching

def _get(url: str, timeout=30) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "BioScaff-Studio/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", errors="replace")


def fetch_pubchem(query: str) -> tuple[str, str]:
    """3D SDF from PubChem by name or CID. Returns (sdf text, title)."""
    q = query.strip()
    kind = "cid" if q.isdigit() else "name"
    base = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound"
    path = f"{base}/{kind}/{urllib.parse.quote(q)}/SDF"
    try:
        return _get(path + "?record_type=3d"), q
    except Exception:  # noqa: BLE001
        try:
            return _get(path), q          # 2D record; 3D is generated on reading
        except Exception as e:  # noqa: BLE001
            raise ValueError(f"PubChem has no structure for '{q}' ({e}).") from e


def fetch_rcsb(pdb_id: str) -> str:
    pid = pdb_id.strip().upper()
    if not re.fullmatch(r"[0-9][A-Z0-9]{3}", pid):
        raise ValueError("A PDB ID has four characters, e.g. 1BKV.")
    try:
        return _get(f"https://files.rcsb.org/download/{pid}.pdb")
    except Exception as e:  # noqa: BLE001
        raise ValueError(f"Could not download {pid} from RCSB ({e}).") from e


# ================================================================ drawing

def depiction(entry: dict, width=420, height=320, png=False):
    """2D structure drawing of a compound (SVG text, or PNG bytes)."""
    from rdkit import Chem
    from rdkit.Chem import rdDepictor
    from rdkit.Chem.Draw import rdMolDraw2D
    m = Chem.MolFromMolBlock(entry.get("molblock") or entry["display"])
    m = Chem.RemoveHs(m)
    rdDepictor.Compute2DCoords(m)
    d = rdMolDraw2D.MolDraw2DCairo(width, height) if png else rdMolDraw2D.MolDraw2DSVG(width, height)
    d.drawOptions().clearBackground = True
    d.DrawMolecule(m, legend=entry["name"])
    d.FinishDrawing()
    return d.GetDrawingText()


def depiction_grid(entries: list, per_row=4, size=(300, 260)) -> str:
    from rdkit import Chem
    from rdkit.Chem import Draw, rdDepictor
    mols, legends = [], []
    for e in entries:
        m = Chem.RemoveHs(Chem.MolFromMolBlock(e.get("molblock") or e["display"]))
        rdDepictor.Compute2DCoords(m)
        mols.append(m)
        legends.append(e["name"])
    return Draw.MolsToGridImage(mols, molsPerRow=per_row, subImgSize=size,
                                legends=legends, useSVG=True)
