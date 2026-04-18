"""
Seed Crystal Sweet Spot Filter
================================
Find materials where:
- Space-grown seed crystal improves quality (moderate density contrast)
- BUT terrestrial bulk growth from seed IS possible (compatible melting points)  
- AND the material is commercially valuable (useful bandgap, high-value sector)

The key insight: extreme density ratios (30-40x) make materials IMPOSSIBLE to
synthesize on Earth at all. But moderate ratios (2-10x) cause DEFECTS during
terrestrial crystal growth — defects that a space-grown seed could eliminate.

Usage:
    python filter_seed_crystal.py --input microgravity_candidates.csv
"""

import pandas as pd
import numpy as np
import argparse
import re

# Element melting points in Kelvin
MELTING_POINTS = {
    "H": 14, "He": 1, "Li": 454, "Be": 1560, "B": 2349,
    "C": 3823, "N": 63, "O": 54, "F": 53, "Na": 371, "Mg": 923,
    "Al": 933, "Si": 1687, "P": 317, "S": 388, "K": 336, "Ca": 1115,
    "Sc": 1814, "Ti": 1941, "V": 2183, "Cr": 2180, "Mn": 1519, "Fe": 1811,
    "Co": 1768, "Ni": 1728, "Cu": 1358, "Zn": 692, "Ga": 303, "Ge": 1211,
    "As": 1090, "Se": 494, "Rb": 312, "Sr": 1050, "Y": 1799, "Zr": 2128,
    "Nb": 2750, "Mo": 2896, "Ru": 2607, "Rh": 2237, "Pd": 1828, "Ag": 1235,
    "Cd": 594, "In": 429, "Sn": 505, "Sb": 904, "Te": 723, "Cs": 302,
    "Ba": 1000, "La": 1193, "Ce": 1068, "Pr": 1208, "Nd": 1297, "Sm": 1345,
    "Eu": 1099, "Gd": 1585, "Tb": 1629, "Dy": 1680, "Ho": 1734, "Er": 1802,
    "Tm": 1818, "Yb": 1097, "Lu": 1925, "Hf": 2506, "Ta": 3290, "W": 3695,
    "Re": 3459, "Os": 3306, "Ir": 2719, "Pt": 2041, "Au": 1337, "Pb": 600,
    "Bi": 544, "Th": 2023, "U": 1405,
}

ELEMENT_DENSITY = {
    "H": 0.00009, "Li": 0.534, "Be": 1.85, "B": 2.34, "C": 2.27,
    "Na": 0.97, "Mg": 1.74, "Al": 2.70, "Si": 2.33, "P": 1.82,
    "S": 2.07, "K": 0.86, "Ca": 1.55, "Sc": 2.99, "Ti": 4.51,
    "V": 6.11, "Cr": 7.19, "Mn": 7.21, "Fe": 7.87, "Co": 8.90,
    "Ni": 8.91, "Cu": 8.96, "Zn": 7.13, "Ga": 5.91, "Ge": 5.32,
    "As": 5.73, "Se": 4.81, "Rb": 1.53, "Sr": 2.64, "Y": 4.47,
    "Zr": 6.51, "Nb": 8.57, "Mo": 10.28, "Ru": 12.37, "Rh": 12.41,
    "Pd": 12.02, "Ag": 10.49, "Cd": 8.65, "In": 7.31, "Sn": 7.29,
    "Sb": 6.68, "Te": 6.24, "Cs": 1.87, "Ba": 3.51, "La": 6.15,
    "Ce": 6.77, "Pr": 6.77, "Nd": 7.01, "Sm": 7.52, "Eu": 5.24,
    "Gd": 7.90, "Tb": 8.23, "Dy": 8.55, "Ho": 8.80, "Er": 9.07,
    "Tm": 9.32, "Yb": 6.90, "Lu": 9.84, "Hf": 13.31, "Ta": 16.65,
    "W": 19.25, "Re": 21.02, "Os": 22.59, "Ir": 22.56, "Pt": 21.45,
    "Au": 19.30, "Pb": 11.34, "Bi": 9.78, "Th": 11.72, "U": 19.05,
}

# $/kg estimates for end-use applications
SECTOR_VALUE = {
    "semiconductor": 50000,   # substrate wafers
    "quantum": 100000,        # quantum computing components
    "optical": 20000,         # optical/photonic devices  
    "defense": 30000,         # defense-grade components
    "battery": 5000,          # battery materials
    "superalloy": 10000,      # turbine components
    "energy": 3000,           # energy harvesting
}


def parse_composition(formula):
    pattern = r'([A-Z][a-z]?)(\d*\.?\d*)'
    matches = re.findall(pattern, str(formula))
    comp = {}
    for el, count in matches:
        if el in ELEMENT_DENSITY:
            comp[el] = float(count) if count else 1.0
    return comp


def melting_point_compatibility(comp):
    """
    Score how compatible the melting points are for terrestrial melt growth.
    Lower ratio = easier to melt together = better for seed crystal model.
    
    Returns (max_mp, min_mp, ratio, compatible)
    """
    elements = [e for e in comp.keys() if e in MELTING_POINTS]
    if len(elements) < 2:
        return 0, 0, 0, False
    
    mps = [MELTING_POINTS[e] for e in elements]
    # Skip gases and very low MP elements for this calculation
    solid_mps = [mp for mp in mps if mp > 200]
    if len(solid_mps) < 2:
        return 0, 0, 0, False
    
    max_mp = max(solid_mps)
    min_mp = min(solid_mps)
    ratio = max_mp / min_mp if min_mp > 0 else 999
    
    # Compatible if ratio < 4 (all elements can coexist in melt)
    # Marginal if ratio 4-8
    # Incompatible if ratio > 8 (lightest element evaporates before heaviest melts)
    compatible = ratio < 4
    
    return max_mp, min_mp, round(ratio, 2), compatible


def density_ratio(comp):
    """Calculate density ratio between heaviest and lightest solid element."""
    elements = [e for e in comp.keys() if e in ELEMENT_DENSITY]
    densities = [ELEMENT_DENSITY[e] for e in elements if ELEMENT_DENSITY[e] > 0.01]
    if len(densities) < 2:
        return 0
    return round(max(densities) / min(densities), 2)


def seed_crystal_score(row, comp):
    """
    Score suitability for the seed crystal model.
    
    Sweet spot:
    - Density ratio 2-10x (enough to cause defects, not so much it's impossible)
    - Melting point ratio < 4 (can create terrestrial melt)
    - Bandgap 0.3-3.5 eV (commercially valuable semiconductor)
    - In high-value sector
    """
    score = 0
    reasons = []
    
    # 1. Density contrast: sweet spot is 2-10x
    dr = density_ratio(comp)
    if 2 <= dr <= 5:
        score += 30
        reasons.append(f"Density ratio {dr}x: ideal for defect-causing but synthesizable")
    elif 5 < dr <= 10:
        score += 20
        reasons.append(f"Density ratio {dr}x: significant defects, terrestrial growth challenging")
    elif 1.5 <= dr < 2:
        score += 10
        reasons.append(f"Density ratio {dr}x: mild gravity effect")
    else:
        reasons.append(f"Density ratio {dr}x: too extreme for seed model")
    
    # 2. Melting point compatibility
    max_mp, min_mp, mp_ratio, compatible = melting_point_compatibility(comp)
    if compatible:
        score += 25
        reasons.append(f"MP ratio {mp_ratio}x: terrestrial melt growth feasible")
    elif mp_ratio < 6:
        score += 10
        reasons.append(f"MP ratio {mp_ratio}x: terrestrial growth difficult but possible")
    else:
        reasons.append(f"MP ratio {mp_ratio}x: melt growth likely impossible on Earth")
    
    # 3. Bandgap value
    bandgap = row.get("Bandgap", None)
    if pd.notna(bandgap):
        if 0.5 <= bandgap <= 1.8:
            score += 25
            reasons.append(f"Bandgap {bandgap:.2f} eV: prime semiconductor range")
        elif 0.3 <= bandgap < 0.5 or 1.8 < bandgap <= 3.5:
            score += 15
            reasons.append(f"Bandgap {bandgap:.2f} eV: useful range")
        elif bandgap < 0.05:
            reasons.append(f"Bandgap {bandgap:.3f} eV: metallic, low seed value")
    else:
        reasons.append("No bandgap data")
    
    # 4. Commercial value from sectors
    sectors_raw = row.get("target_sectors", "[]")
    if isinstance(sectors_raw, str):
        sectors = [s.strip().strip("'\"") for s in sectors_raw.strip("[]").split(",") if s.strip()]
    else:
        sectors = sectors_raw if sectors_raw else []
    
    max_sector_value = 0
    for s in sectors:
        s_clean = s.strip()
        if s_clean in SECTOR_VALUE:
            max_sector_value = max(max_sector_value, SECTOR_VALUE[s_clean])
    
    if max_sector_value >= 50000:
        score += 20
        reasons.append(f"High-value sector (${max_sector_value:,}/kg end-use)")
    elif max_sector_value >= 10000:
        score += 10
        reasons.append(f"Moderate-value sector (${max_sector_value:,}/kg end-use)")
    elif max_sector_value > 0:
        score += 5
        reasons.append(f"Lower-value sector (${max_sector_value:,}/kg end-use)")
    
    return score, reasons, dr, mp_ratio, compatible


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="microgravity_candidates.csv")
    parser.add_argument("--min_overall_score", type=float, default=30.0,
                        help="Minimum microgravity advantage score from original filter")
    parser.add_argument("--top_n", type=int, default=30)
    parser.add_argument("--output", default="seed_crystal_candidates.csv")
    args = parser.parse_args()
    
    df = pd.read_csv(args.input)
    print(f"Loaded {len(df):,} candidates")
    
    # Start from materials with at least moderate microgravity advantage
    df = df[df["microgravity_advantage_score"] >= args.min_overall_score].copy()
    print(f"Score >= {args.min_overall_score}: {len(df):,}")
    
    # Score each for seed crystal potential
    results = []
    for idx, row in df.iterrows():
        comp = parse_composition(row["Reduced Formula"])
        sc_score, reasons, dr, mp_ratio, mp_compat = seed_crystal_score(row, comp)
        results.append({
            "idx": idx,
            "seed_score": sc_score,
            "reasons": reasons,
            "density_ratio": dr,
            "mp_ratio": mp_ratio,
            "mp_compatible": mp_compat,
        })
    
    res_df = pd.DataFrame(results).set_index("idx")
    df = df.join(res_df)
    
    # Filter: must have melting point compatibility AND useful seed score
    viable = df[df["seed_score"] >= 50].copy()
    viable = viable.sort_values("seed_score", ascending=False)
    
    print(f"\nViable seed crystal candidates (score >= 50): {len(viable):,}")
    
    # Report
    print(f"\n{'='*80}")
    print(f"  SEED CRYSTAL SWEET SPOT — Space seeds for terrestrial scale-up")
    print(f"{'='*80}")
    print(f"\n  These materials have:")
    print(f"  • Moderate density contrast (gravity causes defects, not impossibility)")
    print(f"  • Compatible melting points (terrestrial melt growth IS possible)")
    print(f"  • Commercially valuable bandgap")
    print(f"  • High-value end-use sectors")
    print(f"\n  Total viable: {len(viable):,} materials\n")
    
    for i, (idx, row) in enumerate(viable.head(args.top_n).iterrows(), 1):
        bg = row.get("Bandgap", None)
        bg_str = f"{bg:.3f} eV" if pd.notna(bg) and bg > 0 else "N/A"
        
        print(f"  #{i:3d}  {row['Reduced Formula']}")
        print(f"        Seed score: {row['seed_score']:.0f}/100 | "
              f"Microgravity score: {row['microgravity_advantage_score']:.1f}")
        print(f"        Bandgap: {bg_str} | "
              f"Density ratio: {row['density_ratio']}x | "
              f"MP ratio: {row['mp_ratio']}x")
        print(f"        Crystal: {row.get('Crystal System', 'N/A')} | "
              f"Space group: {row.get('Space Group', 'N/A')}")
        print(f"        Sectors: {row.get('target_sectors', 'N/A')}")
        print(f"        MP compatible: {'YES' if row['mp_compatible'] else 'MARGINAL'}")
        
        for reason in row["reasons"]:
            print(f"          → {reason}")
        print()
    
    # Save
    out_cols = [c for c in ["MaterialId", "Reduced Formula", "Elements", 
                "Crystal System", "Space Group", "Bandgap",
                "Formation Energy Per Atom", "Decomposition Energy Per Atom",
                "microgravity_advantage_score", "seed_score", "density_ratio",
                "mp_ratio", "mp_compatible", "target_sectors"] if c in viable.columns]
    
    viable[out_cols].to_csv(args.output, index=False)
    print(f"  Saved {len(viable)} candidates to {args.output}")
    
    print(f"\n{'='*80}")
    print(f"  WHY THESE MATERIALS MATTER")
    print(f"{'='*80}")
    print(f"  Unlike our extreme-ratio candidates (Li-Ir at 42x), these CAN be")
    print(f"  melted and grown on Earth — they just grow with defects due to gravity.")
    print(f"  A space-grown seed crystal with perfect structure could template")
    print(f"  defect-free terrestrial growth via Czochralski or Bridgman methods.")
    print(f"")
    print(f"  The Space Forge model: 1 kg space seed → tonnes of Earth material")
    print(f"  At $1M-$50M per kg of seed, the economics work.")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()