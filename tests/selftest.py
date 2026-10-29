"""
Quick self-test. Run from the project folder after installing requirements:

    python tests/selftest.py

It checks the chemistry (RDKit), geometry and models on molecules whose
answers are known, then runs a small pipeline. Every line should say PASS.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from core import models as MOD  # noqa: E402
from core import molecules as MOL  # noqa: E402
from core import report as REP  # noqa: E402

failures = 0


def check(label, ok, detail=""):
    global failures
    failures += not ok
    print(f"{'PASS' if ok else 'FAIL'}  {label}" + (f"   [{detail}]" if detail else ""))


def groups_of(entry):
    return {g["group"]: g["count"] for g in entry["groups"]}


# ---------------------------------------------------------------- physics
check("Water viscosity at 25 °C ≈ 0.89 mPa·s",
      abs(MOD.water_viscosity(298.15) * 1e3 - 0.890) < 0.01)
check("Slab release: τ50 = π/64", abs(MOD.TAU50 - np.pi / 64) < 1e-5, f"{MOD.TAU50:.5f}")
check("Ogston: no polymer gives D/D0 = 1", MOD.ogston(0.5, 0.8, 0.0) == 1.0)
check("Single-atom SASA is exact",
      abs(MOL.sasa_per_atom(np.zeros((1, 3)), np.array([1.7]))[0] - 4 * np.pi * 3.1 ** 2) < 1e-6)

if not MOL.rdkit_available():
    print("FAIL  RDKit is not installed: pip install rdkit")
    sys.exit(1)

# ---------------------------------------------------------------- chemistry
cip = MOL.build_entry("Ciprofloxacin", "C1CC1N2C=C(C(=O)C3=CC(=C(C=C32)N4CCNCC4)F)C(=O)O",
                      "smiles", "test")
check("Ciprofloxacin formula C17H18FN3O3", cip["formula"] == "C17H18FN3O3", cip["formula"])
check("Ciprofloxacin MW 331.3", abs(cip["mw"] - 331.35) < 0.1, f"{cip['mw']:.2f}")
g = groups_of(cip)
check("Ciprofloxacin: 1 carboxylic acid, 1 secondary amine, aryl-N not basic",
      g.get("Carboxylic acid") == 1 and g.get("Secondary aliphatic amine") == 1
      and "Tertiary aliphatic amine" not in g, str(g))

pcl = MOL.build_entry("PCL 5-mer", "OCCCCCC(=O)OCCCCCC(=O)OCCCCCC(=O)OCCCCCC(=O)OCCCCCC(=O)O",
                      "smiles", "test")
check("PCL 5-mer formula C30H52O11", pcl["formula"] == "C30H52O11", pcl["formula"])
check("PCL: only the end carboxyl is ionizable", groups_of(pcl) == {"Carboxylic acid": 1},
      str(groups_of(pcl)))

chito = MOL.build_entry(
    "GlcN trimer (constitution only)",
    "NC1C(O)C(O)C(CO)OC1OC2C(CO)OC(OC3C(CO)OC(O)C(N)C3O)C(N)C2O", "smiles", "test")
check("GlcN trimer formula C18H35N3O13", chito["formula"] == "C18H35N3O13", chito["formula"])
check("GlcN trimer: 3 amino-sugar amines (pKa 6.5)",
      groups_of(chito) == {"Amino-sugar amine (glucosamine-type)": 3}, str(groups_of(chito)))
q, qp, _ = MOL.charge_at_pH(chito["groups"], 6.5)
check("Amino-sugar amines half protonated at pH = pKa", abs(float(qp) - 1.5) < 1e-6)

# SDF round trip, 2D-only input, and salts
sdf = cip["molblock"] + "$$$$\n"
cip2 = MOL.build_entry("Cipro from SDF", sdf, "sdf", "test")
check("SDF input reads back the same formula", cip2["formula"] == cip["formula"])
try:
    svg = MOL.depiction(cip)
    check("2D drawing (SVG)", svg.strip().startswith("<?xml") or "<svg" in svg)
except Exception as e:  # noqa: BLE001
    check("2D drawing (SVG)", False, str(e))

# ---------------------------------------------------------------- pipeline
for e in (cip, pcl, chito):
    e["role"] = "Drug / payload" if e is cip else "Scaffold component"
mols = {"cip": cip, "pcl": pcl, "chi": chito}
conds = [
    {"id": "A", "name": "Water + drug", "role": "Positive control", "components": [],
     "challenge": "drug", "drug_id": "cip"},
    {"id": "B", "name": "Chitosan", "role": "Test",
     "components": [{"mol_id": "chi", "conc": 2.0}], "challenge": "drug", "drug_id": "cip"},
    {"id": "C", "name": "Chitosan + PCL", "role": "Test",
     "components": [{"mol_id": "chi", "conc": 2.0}, {"mol_id": "pcl", "conc": 2.0}],
     "challenge": "drug", "drug_id": "cip"},
]
R = MOD.run_conditions(mols, conds, {"n_samples": 500})
check("Pipeline: denser network hinders the drug more",
      R["C"]["summary"]["D_rel"]["median"] < R["B"]["summary"]["D_rel"]["median"] < 1.0)
S = REP.write_summary(R, conds, mols, {**MOD.DEFAULT_SETTINGS, "n_samples": 500}, ("C", "B"))
check("Summary written", "Primary comparison" in S and "Methods" in S)

print("\nAll checks passed." if not failures else f"\n{failures} check(s) failed.")
sys.exit(1 if failures else 0)
