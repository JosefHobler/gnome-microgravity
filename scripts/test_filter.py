"""
Test the microgravity filter with known materials that SHOULD score high/low.
Run this to validate the scoring logic before running on full GNoME dataset.
"""

import pandas as pd
import sys
sys.path.append(".")
from gnome_microgravity_filter import (
    parse_composition, density_contrast_score, crystal_quality_score,
    containerless_score, commercial_value_score, novelty_score
)

# Known materials and expected behavior
test_cases = [
    {
        "name": "W-Cu alloy (tungsten-copper)",
        "formula": "W3Cu1",
        "expect": "HIGH density contrast — W is 19.25, Cu is 8.96 g/cm³. "
                  "Classic case where gravity causes sedimentation.",
        "crystal_system": "cubic",
        "bandgap": None,
    },
    {
        "name": "GaAs (gallium arsenide)", 
        "formula": "Ga1As1",
        "expect": "HIGH crystal quality score — premier semiconductor, "
                  "defect-free crystals worth enormous premium.",
        "crystal_system": "cubic",
        "bandgap": 1.42,
    },
    {
        "name": "InSb (indium antimonide)",
        "formula": "In1Sb1",
        "expect": "HIGH overall — semiconductor + density contrast + "
                  "crystal quality all matter.",
        "crystal_system": "cubic",
        "bandgap": 0.17,
    },
    {
        "name": "NaCl (table salt)",
        "formula": "Na1Cl1",
        "expect": "LOW — no commercial value in space manufacturing, "
                  "similar densities, cheap on Earth.",
        "crystal_system": "cubic",
        "bandgap": 8.5,
    },
    {
        "name": "Ni-W-Re superalloy",
        "formula": "Ni3W1Re1",
        "expect": "HIGH — extreme density contrast (Ni=8.9, W=19.25, Re=21.02), "
                  "superalloy application, high melting point.",
        "crystal_system": "cubic",
        "bandgap": None,
    },
    {
        "name": "Li-Pb battery material",
        "formula": "Li3Pb1",
        "expect": "HIGH density contrast — Li is 0.534, Pb is 11.34 g/cm³. "
                  "21x density ratio! Impossible to keep mixed on Earth.",
        "crystal_system": "cubic",
        "bandgap": None,
    },
    {
        "name": "Hf-Ta-W refractory",
        "formula": "Hf1Ta1W1",
        "expect": "HIGH containerless score — all elements >2500K melting point. "
                  "Container contamination is a real problem.",
        "crystal_system": "cubic",
        "bandgap": None,
    },
    {
        "name": "Si-Ge semiconductor",
        "formula": "Si1Ge1",
        "expect": "MODERATE-HIGH — important semiconductor, crystal quality matters, "
                  "but density contrast is modest (2.33 vs 5.32).",
        "crystal_system": "cubic",
        "bandgap": 0.95,
    },
]

print("=" * 70)
print("  MICROGRAVITY FILTER — VALIDATION TEST")
print("=" * 70)

for tc in test_cases:
    comp = parse_composition(tc["formula"])
    row = pd.Series({
        "Reduced Formula": tc["formula"],
        "Crystal System": tc["crystal_system"],
        "Bandgap": tc["bandgap"],
        "Decomposition Energy Per Atom": 0.01,
    })
    
    dc = density_contrast_score(comp)
    cq = crystal_quality_score(row)
    cl = containerless_score(comp)
    cv, sectors = commercial_value_score(comp)
    nv = novelty_score(row)
    
    total = dc * 0.30 + cq * 0.25 + cl * 0.15 + cv * 0.20 + nv * 0.10
    
    print(f"\n{'─' * 70}")
    print(f"  {tc['name']}  ({tc['formula']})")
    print(f"  Expected: {tc['expect']}")
    print(f"  Density contrast:  {dc:6.2f}  │  Crystal quality: {cq:5.2f}")
    print(f"  Containerless:     {cl:6.2f}  │  Commercial:      {cv:5.2f}")
    print(f"  Novelty:           {nv:6.2f}  │  TOTAL (raw):     {total:5.2f}")
    print(f"  Sectors: {', '.join(sectors) if sectors else 'none'}")
    
    # Element density breakdown
    densities = {e: f"{d:.2f}" for e, d in 
                 [(e, __import__('gnome_microgravity_filter').ELEMENT_DENSITY.get(e, 0)) 
                  for e in comp] if d > 0.01}
    print(f"  Densities (g/cm³): {densities}")

print(f"\n{'=' * 70}")
print("  If scores match expectations, run on full GNoME dataset:")
print("  python gnome_microgravity_filter.py --data_path <path_to_csv>")
print("=" * 70)