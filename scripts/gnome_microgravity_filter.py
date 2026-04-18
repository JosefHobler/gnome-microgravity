"""
GNoME Microgravity Advantage Filter
====================================
Filters DeepMind's GNoME dataset (380k+ novel stable materials) to find
materials whose synthesis would specifically benefit from microgravity
manufacturing in space.

Microgravity advantages:
1. No sedimentation → can mix elements with vastly different densities
2. No convection → defect-free crystal growth
3. No buoyancy → uniform phase distribution
4. Containerless processing → ultra-pure materials

Usage:
    pip install pandas pymatgen numpy
    
    # Download GNoME data first:
    # gsutil -m cp -r gs://gdm_materials_discovery/ data/
    # OR: python scripts/download_data_wget.py (from the GNoME repo)
    
    python gnome_microgravity_filter.py --data_path ./stable_materials_summary.csv
"""

import pandas as pd
import numpy as np
import argparse
import json
from pathlib import Path
from collections import Counter
import re

# =============================================================================
# ELEMENT DATA: densities, melting points, costs, applications
# =============================================================================

# Element densities in g/cm³ — key for sedimentation scoring
ELEMENT_DENSITY = {
    "H": 0.00009, "He": 0.00018, "Li": 0.534, "Be": 1.85, "B": 2.34,
    "C": 2.27, "N": 0.0013, "O": 0.0014, "F": 0.0017, "Ne": 0.0009,
    "Na": 0.97, "Mg": 1.74, "Al": 2.70, "Si": 2.33, "P": 1.82,
    "S": 2.07, "Cl": 0.0032, "Ar": 0.0018, "K": 0.86, "Ca": 1.55,
    "Sc": 2.99, "Ti": 4.51, "V": 6.11, "Cr": 7.19, "Mn": 7.21,
    "Fe": 7.87, "Co": 8.90, "Ni": 8.91, "Cu": 8.96, "Zn": 7.13,
    "Ga": 5.91, "Ge": 5.32, "As": 5.73, "Se": 4.81, "Br": 3.12,
    "Kr": 0.0037, "Rb": 1.53, "Sr": 2.64, "Y": 4.47, "Zr": 6.51,
    "Nb": 8.57, "Mo": 10.28, "Tc": 11.50, "Ru": 12.37, "Rh": 12.41,
    "Pd": 12.02, "Ag": 10.49, "Cd": 8.65, "In": 7.31, "Sn": 7.29,
    "Sb": 6.68, "Te": 6.24, "I": 4.93, "Xe": 0.0059, "Cs": 1.87,
    "Ba": 3.51, "La": 6.15, "Ce": 6.77, "Pr": 6.77, "Nd": 7.01,
    "Pm": 7.26, "Sm": 7.52, "Eu": 5.24, "Gd": 7.90, "Tb": 8.23,
    "Dy": 8.55, "Ho": 8.80, "Er": 9.07, "Tm": 9.32, "Yb": 6.90,
    "Lu": 9.84, "Hf": 13.31, "Ta": 16.65, "W": 19.25, "Re": 21.02,
    "Os": 22.59, "Ir": 22.56, "Pt": 21.45, "Au": 19.30, "Hg": 13.53,
    "Tl": 11.85, "Pb": 11.34, "Bi": 9.78, "Po": 9.20, "At": 7.00,
    "Rn": 0.0097, "Fr": 2.48, "Ra": 5.50, "Ac": 10.07, "Th": 11.72,
    "Pa": 15.37, "U": 19.05, "Np": 20.45,
}

# Melting points in Kelvin — high melting points make containerless processing valuable
ELEMENT_MELTING_POINT = {
    "Li": 453, "Be": 1560, "B": 2349, "C": 3823, "Na": 371, "Mg": 923,
    "Al": 933, "Si": 1687, "P": 317, "S": 388, "K": 336, "Ca": 1115,
    "Sc": 1814, "Ti": 1941, "V": 2183, "Cr": 2180, "Mn": 1519, "Fe": 1811,
    "Co": 1768, "Ni": 1728, "Cu": 1358, "Zn": 692, "Ga": 303, "Ge": 1211,
    "As": 1090, "Se": 494, "Rb": 312, "Sr": 1050, "Y": 1799, "Zr": 2128,
    "Nb": 2750, "Mo": 2896, "Ru": 2607, "Rh": 2237, "Pd": 1828, "Ag": 1235,
    "Cd": 594, "In": 429, "Sn": 505, "Sb": 904, "Te": 723, "Cs": 302,
    "Ba": 1000, "La": 1193, "Ce": 1068, "Pr": 1208, "Nd": 1297, "Sm": 1345,
    "Eu": 1099, "Gd": 1585, "Tb": 1629, "Dy": 1680, "Ho": 1734, "Er": 1802,
    "Tm": 1818, "Yb": 1097, "Lu": 1925, "Hf": 2506, "Ta": 3290, "W": 3695,
    "Re": 3459, "Os": 3306, "Ir": 2719, "Pt": 2041, "Au": 1337, "Hg": 234,
    "Tl": 577, "Pb": 600, "Bi": 544, "Th": 2023, "U": 1405,
}

# High-value application sectors per element group
# Used to score commercial potential
HIGH_VALUE_ELEMENTS = {
    # Semiconductor materials
    "semiconductor": {"Si", "Ge", "Ga", "As", "In", "P", "Sb", "Se", "Te", "Cd", "Zn", "Sn"},
    # Battery / energy storage
    "battery": {"Li", "Na", "K", "Co", "Ni", "Mn", "Fe", "P", "S", "Ti", "V"},
    # Quantum computing
    "quantum": {"Nb", "Ta", "Al", "Ti", "Si", "Ge", "Ga", "As", "In", "Sb"},
    # Superalloys / aerospace
    "superalloy": {"Ni", "Co", "Cr", "Mo", "W", "Re", "Ta", "Al", "Ti", "Hf", "Nb", "Cu"},
    # Optical / photonics
    "optical": {"Si", "Ge", "Ga", "As", "In", "P", "Se", "Te", "Zn", "Cd", "Zr", "Ba", "La"},
    # Defense / advanced
    "defense": {"W", "Re", "Ta", "Mo", "Nb", "Hf", "Ir", "Os", "Rh", "Ru", "Cu", "Pb"},
    "energy":  {"Li", "Pb", "Be", "U", "Th", "Zr", "Nb"}
}


def parse_composition(formula: str) -> dict:
    """Parse a chemical formula into element:count dict."""
    pattern = r'([A-Z][a-z]?)(\d*\.?\d*)'
    matches = re.findall(pattern, str(formula))
    composition = {}
    for element, count in matches:
        if element in ELEMENT_DENSITY:
            composition[element] = float(count) if count else 1.0
    return composition


def density_contrast_score(composition: dict) -> float:
    """
    Score how much sedimentation would affect this material on Earth.
    Higher score = bigger density differences between elements = more 
    benefit from microgravity (no sedimentation).
    
    This is the #1 indicator of microgravity advantage.
    """
    elements = [e for e in composition.keys() if e in ELEMENT_DENSITY]
    if len(elements) < 2:
        return 0.0
    
    densities = [ELEMENT_DENSITY[e] for e in elements]
    max_d, min_d = max(densities), min(densities)
    
    if min_d < 0.01:  # skip gases
        densities = [d for d in densities if d > 0.01]
        if len(densities) < 2:
            return 0.0
        max_d, min_d = max(densities), min(densities)
    
    # Ratio of heaviest to lightest element
    ratio = max_d / min_d if min_d > 0 else 0
    
    # Also consider the spread across all elements
    std_dev = np.std(densities)
    
    # Combined score: high ratio AND high spread = strong microgravity case
    score = (ratio * 0.6) + (std_dev * 0.4)
    return round(score, 3)


def crystal_quality_score(row: pd.Series) -> float:
    """
    Score how much crystal quality would improve in microgravity.
    Materials that need high crystal perfection benefit most.
    
    Factors:
    - Semiconductor materials need defect-free crystals
    - Low bandgap materials are more sensitive to defects
    - Higher symmetry crystals benefit more from uniform growth
    """
    score = 0.0
    
    composition = parse_composition(row.get("Reduced Formula", ""))
    elements = set(composition.keys())
    
    # Semiconductor elements present → crystal quality matters enormously
    semiconductor_overlap = elements & HIGH_VALUE_ELEMENTS["semiconductor"]
    if semiconductor_overlap:
        score += len(semiconductor_overlap) * 2.0
    
    # Bandgap in semiconductor range (0.1 - 3.5 eV) → defect-sensitive
    bandgap = row.get("Bandgap", None)
    if pd.notna(bandgap) and 0.1 < bandgap < 3.5:
        score += 3.0
    
    # High symmetry crystals benefit more from convection-free growth
    crystal_system = str(row.get("Crystal System", ""))
    symmetry_bonus = {
        "cubic": 2.0, "hexagonal": 1.5, "tetragonal": 1.0,
        "trigonal": 1.0, "orthorhombic": 0.5
    }
    score += symmetry_bonus.get(crystal_system.lower(), 0)
    
    return round(score, 3)


def containerless_score(composition: dict) -> float:
    """
    Score benefit from containerless processing in space.
    High-melting-point materials contaminate crucibles on Earth.
    In space, electromagnetic levitation allows container-free processing.
    """
    elements = [e for e in composition.keys() if e in ELEMENT_MELTING_POINT]
    if not elements:
        return 0.0
    
    max_mp = max(ELEMENT_MELTING_POINT.get(e, 0) for e in elements)
    
    # Normalize: materials above 2000K benefit significantly
    if max_mp > 2500:
        return 3.0
    elif max_mp > 2000:
        return 2.0
    elif max_mp > 1500:
        return 1.0
    return 0.0


def commercial_value_score(composition: dict) -> tuple:
    """
    Score commercial potential and identify target sectors.
    Returns (score, list_of_sectors).
    """
    elements = set(composition.keys())
    score = 0.0
    sectors = []
    
    for sector, sector_elements in HIGH_VALUE_ELEMENTS.items():
        overlap = elements & sector_elements
        if len(overlap) >= 2:  # At least 2 elements in the sector
            sector_score = len(overlap) * 1.5
            score += sector_score
            sectors.append(sector)
    
    return round(score, 3), sectors


def novelty_score(row: pd.Series) -> float:
    """
    Score how novel / hard-to-make-on-Earth this material likely is.
    Materials with many different elements and high formation energy
    are harder to synthesize terrestrially.
    """
    n_elements = len(parse_composition(row.get("Reduced Formula", "")))
    
    # More elements = harder to keep homogeneous under gravity
    element_score = min(n_elements * 0.5, 3.0)
    
    # Slightly above hull = metastable, might need special conditions
    decomp = row.get("Decomposition Energy Per Atom", 0)
    if pd.notna(decomp) and 0 < decomp < 0.05:
        element_score += 1.0
    
    return round(element_score, 3)


def compute_microgravity_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Main scoring pipeline: compute all microgravity advantage scores."""
    
    print(f"Processing {len(df):,} materials...")
    
    # Parse compositions
    print("  → Parsing compositions...")
    df["_composition"] = df["Reduced Formula"].apply(parse_composition)
    df["_n_elements"] = df["_composition"].apply(len)
    
    # Filter: only multi-element materials (single elements don't need microgravity)
    df = df[df["_n_elements"] >= 2].copy()
    print(f"  → {len(df):,} multi-element materials")
    
    # Score 1: Density contrast (sedimentation advantage)
    print("  → Computing density contrast scores...")
    df["density_contrast_score"] = df["_composition"].apply(density_contrast_score)
    
    # Score 2: Crystal quality improvement potential
    print("  → Computing crystal quality scores...")
    df["crystal_quality_score"] = df.apply(crystal_quality_score, axis=1)
    
    # Score 3: Containerless processing benefit
    print("  → Computing containerless processing scores...")
    df["containerless_score"] = df["_composition"].apply(containerless_score)
    
    # Score 4: Commercial value
    print("  → Computing commercial value scores...")
    commercial_results = df["_composition"].apply(commercial_value_score)
    df["commercial_score"] = commercial_results.apply(lambda x: x[0])
    df["target_sectors"] = commercial_results.apply(lambda x: x[1])
    
    # Score 5: Novelty / synthesis difficulty
    print("  → Computing novelty scores...")
    df["novelty_score"] = df.apply(novelty_score, axis=1)
    
    # === COMPOSITE SCORE ===
    # Weighted combination — density contrast is the strongest signal
    df["microgravity_advantage_score"] = (
        df["density_contrast_score"] * 0.30 +   # Sedimentation = #1 reason
        df["crystal_quality_score"] * 0.25 +     # Crystal perfection
        df["containerless_score"] * 0.15 +       # High-temp processing
        df["commercial_score"] * 0.20 +          # Is it worth making?
        df["novelty_score"] * 0.10               # How novel?
    )
    
    # Normalize to 0-100
    max_score = df["microgravity_advantage_score"].max()
    if max_score > 0:
        df["microgravity_advantage_score"] = (
            df["microgravity_advantage_score"] / max_score * 100
        ).round(2)
    
    print("  → Done!")
    return df


def get_element_density_details(composition: dict) -> str:
    """Human-readable density breakdown."""
    parts = []
    for el, count in sorted(composition.items(), 
                            key=lambda x: ELEMENT_DENSITY.get(x[0], 0)):
        d = ELEMENT_DENSITY.get(el, 0)
        parts.append(f"{el}: {d:.2f} g/cm³")
    return " | ".join(parts)


def generate_report(df: pd.DataFrame, top_n: int = 50) -> str:
    """Generate a human-readable report of top candidates."""
    
    top = df.nlargest(top_n, "microgravity_advantage_score")
    
    lines = []
    lines.append("=" * 80)
    lines.append("  GNoME × MICROGRAVITY FILTER — TOP CANDIDATES")
    lines.append("  Materials most likely to benefit from space manufacturing")
    lines.append("=" * 80)
    lines.append("")
    lines.append(f"  Dataset: {len(df):,} multi-element materials analyzed")
    lines.append(f"  Showing top {top_n} by composite microgravity advantage score")
    lines.append("")
    
    # Summary stats
    lines.append("─" * 80)
    lines.append("  SCORE DISTRIBUTION")
    lines.append("─" * 80)
    lines.append(f"  Score > 80:  {len(df[df['microgravity_advantage_score'] > 80]):,} materials")
    lines.append(f"  Score > 60:  {len(df[df['microgravity_advantage_score'] > 60]):,} materials")
    lines.append(f"  Score > 40:  {len(df[df['microgravity_advantage_score'] > 40]):,} materials")
    lines.append("")
    
    # Sector breakdown of top candidates
    all_sectors = []
    for sectors in top["target_sectors"]:
        all_sectors.extend(sectors)
    sector_counts = Counter(all_sectors)
    
    lines.append("─" * 80)
    lines.append("  TARGET SECTORS (in top candidates)")
    lines.append("─" * 80)
    for sector, count in sector_counts.most_common():
        lines.append(f"  {sector:15s}: {count} materials")
    lines.append("")
    
    # Individual materials
    lines.append("─" * 80)
    lines.append("  TOP CANDIDATES")
    lines.append("─" * 80)
    
    for rank, (idx, row) in enumerate(top.iterrows(), 1):
        comp = row["_composition"]
        lines.append("")
        lines.append(f"  #{rank:3d}  {row['Reduced Formula']}")
        lines.append(f"        Material ID:     {row.get('MaterialId', 'N/A')}")
        lines.append(f"        OVERALL SCORE:    {row['microgravity_advantage_score']:.1f} / 100")
        lines.append(f"        ├─ Density contrast:  {row['density_contrast_score']:.1f}")
        lines.append(f"        ├─ Crystal quality:   {row['crystal_quality_score']:.1f}")
        lines.append(f"        ├─ Containerless:     {row['containerless_score']:.1f}")
        lines.append(f"        ├─ Commercial value:  {row['commercial_score']:.1f}")
        lines.append(f"        └─ Novelty:           {row['novelty_score']:.1f}")
        lines.append(f"        Crystal system:  {row.get('Crystal System', 'N/A')}")
        lines.append(f"        Space group:     {row.get('Space Group', 'N/A')}")
        bandgap = row.get('Bandgap', None)
        if pd.notna(bandgap):
            lines.append(f"        Bandgap:         {bandgap:.3f} eV")
        lines.append(f"        Target sectors:  {', '.join(row['target_sectors']) if row['target_sectors'] else 'general'}")
        lines.append(f"        Element densities: {get_element_density_details(comp)}")
        
        # Why this material benefits from microgravity
        reasons = []
        if row["density_contrast_score"] > 5:
            densities = [ELEMENT_DENSITY[e] for e in comp if e in ELEMENT_DENSITY and ELEMENT_DENSITY[e] > 0.01]
            if len(densities) >= 2:
                ratio = max(densities) / min(densities)
                reasons.append(f"Density ratio {ratio:.1f}x → severe sedimentation on Earth")
        if row["crystal_quality_score"] > 3:
            reasons.append("Semiconductor-grade crystal quality required")
        if row["containerless_score"] > 2:
            reasons.append("High melting point → benefits from containerless processing")
        if reasons:
            lines.append(f"        WHY MICROGRAVITY: {'; '.join(reasons)}")
    
    lines.append("")
    lines.append("=" * 80)
    lines.append("  NEXT STEPS")
    lines.append("=" * 80)
    lines.append("  1. Run molecular dynamics simulations on top candidates")
    lines.append("     to model crystallization behavior without convection/sedimentation")
    lines.append("  2. Cross-reference with existing terrestrial synthesis attempts")
    lines.append("     (check Materials Project for failed synthesis notes)")  
    lines.append("  3. Estimate $/kg value of each material in target application")
    lines.append("  4. Identify candidates where space-made version would be")
    lines.append("     ORDERS OF MAGNITUDE better, not just incrementally")
    lines.append("  5. Contact potential buyers (semiconductor fabs, defense)")
    lines.append("     to validate willingness to pay premium")
    lines.append("=" * 80)
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Filter GNoME materials for microgravity manufacturing advantage"
    )
    parser.add_argument(
        "--data_path", 
        type=str, 
        default="./stable_materials_summary.csv",
        help="Path to stable_materials_summary.csv"
    )
    parser.add_argument(
        "--top_n", type=int, default=50,
        help="Number of top candidates to show in report"
    )
    parser.add_argument(
        "--output_csv", type=str, default="microgravity_candidates.csv",
        help="Output CSV with all scored materials"
    )
    parser.add_argument(
        "--output_report", type=str, default="microgravity_report.txt",
        help="Output text report"
    )
    parser.add_argument(
        "--min_score", type=float, default=50.0,
        help="Minimum score threshold for output CSV"
    )
    args = parser.parse_args()
    
    # Load data
    print(f"\nLoading GNoME data from {args.data_path}...")
    df = pd.read_csv(args.data_path)
    print(f"Loaded {len(df):,} materials")
    
    # Score everything
    df = compute_microgravity_scores(df)
    
    # Generate report
    report = generate_report(df, top_n=args.top_n)
    print(report)
    
    # Save report
    with open(args.output_report, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\nReport saved to {args.output_report}")
    
    # Save filtered CSV
    output_cols = [
        "MaterialId", "Reduced Formula", "Elements", "Crystal System",
        "Space Group", "Bandgap", "Formation Energy Per Atom",
        "Decomposition Energy Per Atom", "NSites", "Density",
        "microgravity_advantage_score", "density_contrast_score",
        "crystal_quality_score", "containerless_score", 
        "commercial_score", "novelty_score", "target_sectors",
        "_n_elements",
    ]
    
    existing_cols = [c for c in output_cols if c in df.columns]
    filtered = df[df["microgravity_advantage_score"] >= args.min_score][existing_cols]
    filtered = filtered.sort_values("microgravity_advantage_score", ascending=False)
    filtered.to_csv(args.output_csv, index=False)
    print(f"Saved {len(filtered):,} candidates (score ≥ {args.min_score}) to {args.output_csv}")
    
    # Quick stats
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Total materials analyzed:  {len(df):,}")
    print(f"Score ≥ 80 (strong):      {len(df[df['microgravity_advantage_score'] >= 80]):,}")
    print(f"Score ≥ 60 (promising):   {len(df[df['microgravity_advantage_score'] >= 60]):,}")
    print(f"Score ≥ 40 (possible):    {len(df[df['microgravity_advantage_score'] >= 40]):,}")


if __name__ == "__main__":
    main()