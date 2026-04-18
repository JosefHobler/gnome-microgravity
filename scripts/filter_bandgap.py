"""
Filter microgravity candidates for commercially valuable semiconductors.
Bandgap ranges and what they're used for:
 
  0.3 - 0.5 eV  → Infrared detectors, thermal imaging (defense $$$)
  0.5 - 1.0 eV  → Thermoelectrics, IR sensors, narrow-gap semiconductors
  1.0 - 1.5 eV  → Solar cells (silicon = 1.1 eV, GaAs = 1.42 eV)
  1.5 - 1.8 eV  → LEDs, wide-gap semiconductors, tandem solar cells
  
Usage:
  python filter_bandgap.py --input microgravity_candidates.csv
"""
 
import pandas as pd
import argparse
 
BANDGAP_CATEGORIES = [
    (0.3, 0.5, "IR_detector", "Infrared detectors / thermal imaging (defense)"),
    (0.5, 1.0, "thermoelectric", "Thermoelectrics / IR sensors / narrow-gap semiconductor"),
    (1.0, 1.5, "solar_cell", "Solar cells / photovoltaics (Si=1.1, GaAs=1.42)"),
    (1.5, 1.8, "wide_gap", "LEDs / wide-gap semiconductor / tandem solar"),
]
 
 
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="microgravity_candidates.csv")
    parser.add_argument("--min_score", type=float, default=80.0)
    parser.add_argument("--output", default="semiconductor_moonshots.csv")
    args = parser.parse_args()
 
    df = pd.read_csv(args.input)
    print(f"Loaded {len(df):,} candidates")
 
    # Filter: score >= threshold AND bandgap in useful range
    strong = df[df["microgravity_advantage_score"] >= args.min_score].copy()
    print(f"Score ≥ {args.min_score}: {len(strong):,}")
 
    has_bg = strong[strong["Bandgap"].notna() & (strong["Bandgap"] > 0.01)].copy()
    print(f"With nonzero bandgap: {len(has_bg):,}")
 
    useful = has_bg[(has_bg["Bandgap"] >= 0.3) & (has_bg["Bandgap"] <= 1.8)].copy()
    print(f"Bandgap 0.3–1.8 eV (useful semiconductors): {len(useful):,}")
 
    # Categorize
    def categorize(bg):
        for lo, hi, code, label in BANDGAP_CATEGORIES:
            if lo <= bg < hi:
                return code
        return "other"
 
    def categorize_label(bg):
        for lo, hi, code, label in BANDGAP_CATEGORIES:
            if lo <= bg < hi:
                return label
        return "other"
 
    useful["bandgap_category"] = useful["Bandgap"].apply(categorize)
    useful["bandgap_application"] = useful["Bandgap"].apply(categorize_label)
    useful = useful.sort_values("microgravity_advantage_score", ascending=False)
 
    # Print report
    print(f"\n{'='*80}")
    print(f"  SEMICONDUCTOR MOONSHOTS — Space-only materials with useful bandgaps")
    print(f"{'='*80}\n")
 
    # Summary by category
    print(f"  By application:\n")
    for lo, hi, code, label in BANDGAP_CATEGORIES:
        count = len(useful[useful["bandgap_category"] == code])
        if count > 0:
            print(f"    {label}")
            print(f"    Bandgap {lo}–{hi} eV: {count} materials\n")
 
    # Top candidates per category
    for lo, hi, code, label in BANDGAP_CATEGORIES:
        cat = useful[useful["bandgap_category"] == code]
        if len(cat) == 0:
            continue
 
        print(f"\n{'─'*80}")
        print(f"  {label.upper()}")
        print(f"  Bandgap range: {lo}–{hi} eV | {len(cat)} candidates")
        print(f"{'─'*80}")
 
        for i, (_, row) in enumerate(cat.head(10).iterrows(), 1):
            print(f"\n  #{i}  {row['Reduced Formula']}")
            print(f"       Score: {row['microgravity_advantage_score']:.1f} | "
                  f"Bandgap: {row['Bandgap']:.3f} eV | "
                  f"Crystal: {row.get('Crystal System', 'N/A')}")
            print(f"       Sectors: {row.get('target_sectors', 'N/A')}")
            print(f"       ID: {row.get('MaterialId', 'N/A')}")
 
    # Save
    useful.to_csv(args.output, index=False)
    print(f"\n{'='*80}")
    print(f"  Saved {len(useful)} semiconductor moonshots to {args.output}")
    print(f"{'='*80}")
 
    # The money question
    print(f"\n  NEXT: Google each formula above.")
    print(f"  If NOBODY has ever synthesized it → potential space-exclusive material.")
    print(f"  If someone TRIED and FAILED on Earth → even better signal.\n")
 
 
if __name__ == "__main__":
    main()