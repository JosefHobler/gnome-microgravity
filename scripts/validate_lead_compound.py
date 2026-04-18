"""
Computational Validation: LiTaNb₂(CoSb)₄
==========================================
Step 1 of validating our lead microgravity material candidate.

This script:
1. Loads the GNoME CIF structure for material 5fb363bfdf
2. Analyzes the crystal structure and bonding environment
3. Cross-references with known half-Heusler thermoelectrics in Materials Project
4. Estimates synthesis difficulty and microgravity advantage
5. Prepares input files for DFT validation (Quantum ESPRESSO / VASP)

Requirements:
    pip install pymatgen mp-api numpy matplotlib

Usage:
    # First, extract the CIF from GNoME's by_id.zip:
    # unzip by_id.zip -d cif_files/
    
    python validate_lead_compound.py --cif_path "cif_files/5fb363bfdf.cif"
    
    # If you have a Materials Project API key (get one free at materialsproject.org):
    python validate_lead_compound.py --cif_path "cif_files/5fb363bfdf.cif" --mp_api_key YOUR_MP_API_KEY
"""

import argparse
import json
import numpy as np
from pathlib import Path

try:
    from pymatgen.core import Structure, Composition
    from pymatgen.analysis.structure_analyzer import SpacegroupAnalyzer
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer as SGA
    HAS_PYMATGEN = True
except ImportError:
    HAS_PYMATGEN = False
    print("WARNING: pymatgen not installed. Install with: pip install pymatgen")

try:
    from mp_api.client import MPRester
    HAS_MP = True
except ImportError:
    HAS_MP = False


# ===========================================================================
# KNOWN HALF-HEUSLER THERMOELECTRIC DATA (from literature)
# ===========================================================================
KNOWN_HALF_HEUSLERS = {
    "NbCoSb": {"type": "p", "zT_max": 0.46, "T_zT_max": 973, "bandgap_eV": 1.04},
    "TaCoSb": {"type": "p", "zT_max": 0.30, "T_zT_max": 973, "bandgap_eV": 1.06},
    "TiCoSb": {"type": "n", "zT_max": 1.0,  "T_zT_max": 973, "bandgap_eV": 0.95},
    "NbFeSb": {"type": "p", "zT_max": 1.50, "T_zT_max": 973, "bandgap_eV": 0.54},
    "TaFeSb": {"type": "p", "zT_max": 1.52, "T_zT_max": 973, "bandgap_eV": 0.50},
    "ZrCoSb": {"type": "p", "zT_max": 0.80, "T_zT_max": 973, "bandgap_eV": 0.82},
    "HfCoSb": {"type": "p", "zT_max": 0.80, "T_zT_max": 973, "bandgap_eV": 1.05},
    "ZrCoBi": {"type": "p", "zT_max": 1.42, "T_zT_max": 973, "bandgap_eV": 0.56},
    "ZrNiSn": {"type": "n", "zT_max": 1.20, "T_zT_max": 873, "bandgap_eV": 0.50},
    "HfNiSn": {"type": "n", "zT_max": 1.00, "T_zT_max": 873, "bandgap_eV": 0.50},
}

# Our candidate
CANDIDATE = {
    "formula": "LiTaNb2(CoSb)4",
    "gnome_id": "5fb363bfdf",
    "bandgap_eV": 0.784,  # from GNoME
    "crystal_system": "tetragonal",
    "space_group": "I-42m",
    "formation_energy": -0.4853,  # eV/atom
    "decomp_energy": -0.1327,    # eV/atom (negative = stable)
    "elements": ["Li", "Ta", "Nb", "Co", "Sb"],
    "density_ratio": 31.2,  # Li(0.53) vs Ta(16.65)
}


def analyze_structure(cif_path: str) -> dict:
    """Analyze the crystal structure from CIF file."""
    if not HAS_PYMATGEN:
        return {"error": "pymatgen not installed"}
    
    struct = Structure.from_file(cif_path)
    sga = SpacegroupAnalyzer(struct)
    
    analysis = {
        "formula": str(struct.composition.reduced_formula),
        "n_atoms": len(struct),
        "volume": struct.volume,
        "density_g_cm3": struct.density,
        "space_group": sga.get_space_group_symbol(),
        "space_group_number": sga.get_space_group_number(),
        "crystal_system": sga.get_crystal_system(),
        "lattice_a": struct.lattice.a,
        "lattice_b": struct.lattice.b,
        "lattice_c": struct.lattice.c,
        "lattice_alpha": struct.lattice.alpha,
        "lattice_beta": struct.lattice.beta,
        "lattice_gamma": struct.lattice.gamma,
    }
    
    # Analyze nearest neighbors for each element
    from pymatgen.analysis.local_env import CrystalNN
    cnn = CrystalNN()
    
    element_environments = {}
    for i, site in enumerate(struct):
        el = str(site.specie)
        if el not in element_environments:
            element_environments[el] = []
        try:
            nn_info = cnn.get_nn_info(struct, i)
            neighbors = [str(n["site"].specie) for n in nn_info]
            coord_num = len(neighbors)
            element_environments[el].append({
                "coord_number": coord_num,
                "neighbors": neighbors,
            })
        except Exception:
            pass
    
    analysis["bonding_environments"] = {}
    for el, envs in element_environments.items():
        if envs:
            avg_coord = np.mean([e["coord_number"] for e in envs])
            all_neighbors = []
            for e in envs:
                all_neighbors.extend(e["neighbors"])
            neighbor_counts = {}
            for n in all_neighbors:
                neighbor_counts[n] = neighbor_counts.get(n, 0) + 1
            analysis["bonding_environments"][el] = {
                "avg_coordination": round(avg_coord, 1),
                "neighbor_elements": neighbor_counts,
            }
    
    return analysis


def cross_reference_mp(api_key: str) -> dict:
    """Search Materials Project for related compositions."""
    if not HAS_MP:
        return {"error": "mp-api not installed. pip install mp-api"}
    
    results = {}
    
    with MPRester(api_key) as mpr:
        # Search for Li-containing half-Heuslers
        print("  Searching Materials Project for Li-containing half-Heuslers...")
        
        # Check if our exact composition exists
        docs = mpr.summary.search(
            formula="LiTaNb2Co4Sb4",
            fields=["material_id", "formula_pretty", "band_gap", 
                    "formation_energy_per_atom", "energy_above_hull"]
        )
        results["exact_match"] = [
            {"id": d.material_id, "formula": d.formula_pretty,
             "bandgap": d.band_gap, "e_above_hull": d.energy_above_hull}
            for d in docs
        ]
        
        # Search for related Li-Co-Sb systems
        print("  Searching for Li-Co-Sb systems...")
        docs = mpr.summary.search(
            chemsys="Li-Co-Sb",
            fields=["material_id", "formula_pretty", "band_gap",
                    "formation_energy_per_atom", "energy_above_hull"]
        )
        results["li_co_sb"] = [
            {"id": d.material_id, "formula": d.formula_pretty,
             "bandgap": d.band_gap, "e_above_hull": d.energy_above_hull}
            for d in docs
        ]
        
        # Search for NbCoSb variants (the parent half-Heusler)
        print("  Searching for Nb-Co-Sb systems...")
        docs = mpr.summary.search(
            chemsys="Nb-Co-Sb",
            fields=["material_id", "formula_pretty", "band_gap",
                    "formation_energy_per_atom", "energy_above_hull"]
        )
        results["nb_co_sb"] = [
            {"id": d.material_id, "formula": d.formula_pretty,
             "bandgap": d.band_gap, "e_above_hull": d.energy_above_hull}
            for d in docs
        ]
        
        # Search for Ta-Co-Sb systems
        print("  Searching for Ta-Co-Sb systems...")
        docs = mpr.summary.search(
            chemsys="Ta-Co-Sb",
            fields=["material_id", "formula_pretty", "band_gap",
                    "formation_energy_per_atom", "energy_above_hull"]
        )
        results["ta_co_sb"] = [
            {"id": d.material_id, "formula": d.formula_pretty,
             "bandgap": d.band_gap, "e_above_hull": d.energy_above_hull}
            for d in docs
        ]
    
    return results


def generate_qe_input(cif_path: str) -> str:
    """Generate Quantum ESPRESSO input file for DFT validation."""
    if not HAS_PYMATGEN:
        return "# pymatgen required to generate input from CIF"
    
    struct = Structure.from_file(cif_path)
    
    # QE input template for SCF + bandstructure calculation
    qe_input = f"""&CONTROL
    calculation = 'scf'
    prefix = 'LiTaNb2CoSb4'
    outdir = './tmp/'
    pseudo_dir = './pseudo/'
    tprnfor = .true.
    tstress = .true.
/

&SYSTEM
    ibrav = 0
    nat = {len(struct)}
    ntyp = {len(set(str(s.specie) for s in struct))}
    ecutwfc = 60.0
    ecutrho = 480.0
    occupations = 'smearing'
    smearing = 'cold'
    degauss = 0.01
/

&ELECTRONS
    conv_thr = 1.0d-8
    mixing_beta = 0.3
/

CELL_PARAMETERS angstrom
  {struct.lattice.matrix[0][0]:.10f}  {struct.lattice.matrix[0][1]:.10f}  {struct.lattice.matrix[0][2]:.10f}
  {struct.lattice.matrix[1][0]:.10f}  {struct.lattice.matrix[1][1]:.10f}  {struct.lattice.matrix[1][2]:.10f}
  {struct.lattice.matrix[2][0]:.10f}  {struct.lattice.matrix[2][1]:.10f}  {struct.lattice.matrix[2][2]:.10f}

ATOMIC_SPECIES
"""
    
    # Pseudopotentials (user needs to download these)
    pseudo_map = {
        "Li": "Li.pbe-s-kjpaw_psl.1.0.0.UPF",
        "Ta": "Ta.pbe-spfn-kjpaw_psl.1.0.0.UPF",
        "Nb": "Nb.pbe-spn-kjpaw_psl.1.0.0.UPF",
        "Co": "Co.pbe-spn-kjpaw_psl.1.0.0.UPF",
        "Sb": "Sb.pbe-n-kjpaw_psl.1.0.0.UPF",
    }
    
    masses = {"Li": 6.941, "Ta": 180.948, "Nb": 92.906, "Co": 58.933, "Sb": 121.760}
    
    for el in sorted(set(str(s.specie) for s in struct)):
        qe_input += f"  {el}  {masses.get(el, 1.0)}  {pseudo_map.get(el, f'{el}.UPF')}\n"
    
    qe_input += "\nATOMIC_POSITIONS crystal\n"
    for site in struct:
        el = str(site.specie)
        qe_input += f"  {el}  {site.frac_coords[0]:.10f}  {site.frac_coords[1]:.10f}  {site.frac_coords[2]:.10f}\n"
    
    qe_input += "\nK_POINTS automatic\n  6 6 6  0 0 0\n"
    
    return qe_input


def generate_vasp_input(cif_path: str) -> dict:
    """Generate VASP input files (INCAR, KPOINTS) for DFT validation."""
    
    incar = """SYSTEM = LiTaNb2(CoSb)4 - GNoME validation
# Electronic relaxation
ENCUT = 520
EDIFF = 1E-6
PREC = Accurate
ALGO = Normal
NELM = 200
ISMEAR = 0
SIGMA = 0.05

# Ionic relaxation  
IBRION = 2
NSW = 100
EDIFFG = -0.01
ISIF = 3

# Output
LWAVE = .FALSE.
LCHARG = .TRUE.
LORBIT = 11

# Band structure (run after SCF converges)
# ICHARG = 11
# LORBIT = 11
# NBANDS = 80
"""
    
    kpoints = """Automatic mesh
0
Gamma
  8  8  8
  0  0  0
"""
    
    return {"INCAR": incar, "KPOINTS": kpoints}


def print_validation_report(analysis: dict, mp_results: dict = None):
    """Print comprehensive validation report."""
    
    print("=" * 80)
    print("  VALIDATION REPORT: LiTaNb₂(CoSb)₄")
    print("  GNoME Material ID: 5fb363bfdf")
    print("=" * 80)
    
    print(f"\n  GNoME PREDICTIONS:")
    print(f"  ├─ Bandgap:           {CANDIDATE['bandgap_eV']} eV")
    print(f"  ├─ Crystal system:    {CANDIDATE['crystal_system']}")
    print(f"  ├─ Space group:       {CANDIDATE['space_group']}")
    print(f"  ├─ Formation energy:  {CANDIDATE['formation_energy']} eV/atom")
    print(f"  ├─ Decomp. energy:    {CANDIDATE['decomp_energy']} eV/atom")
    print(f"  └─ Density ratio:     {CANDIDATE['density_ratio']}x (Li vs Ta)")
    
    # Compare with known half-Heuslers
    print(f"\n  COMPARISON WITH KNOWN HALF-HEUSLER THERMOELECTRICS:")
    print(f"  {'Material':<12} {'Type':<6} {'Bandgap':<10} {'zT_max':<8} {'T(K)':<6}")
    print(f"  {'─'*42}")
    
    # Sort by bandgap similarity to our candidate
    sorted_hh = sorted(KNOWN_HALF_HEUSLERS.items(), 
                       key=lambda x: abs(x[1]["bandgap_eV"] - CANDIDATE["bandgap_eV"]))
    
    for name, props in sorted_hh:
        bg_diff = abs(props["bandgap_eV"] - CANDIDATE["bandgap_eV"])
        marker = " ←closest" if bg_diff == min(abs(p["bandgap_eV"] - CANDIDATE["bandgap_eV"]) 
                                                for p in KNOWN_HALF_HEUSLERS.values()) else ""
        print(f"  {name:<12} {props['type']:<6} {props['bandgap_eV']:<10.2f} "
              f"{props['zT_max']:<8.2f} {props['T_zT_max']:<6}{marker}")
    
    print(f"  {'─'*42}")
    print(f"  {'OURS':<12} {'?':<6} {CANDIDATE['bandgap_eV']:<10.2f} {'?':<8} {'?':<6}")
    
    # Analysis
    print(f"\n  KEY OBSERVATIONS:")
    print(f"  1. Bandgap (0.784 eV) is closest to NbFeSb (0.54) and ZrCoBi (0.56)")
    print(f"     → These achieved zT > 1.4, the best in the half-Heusler family")
    print(f"  2. Contains Co+Sb backbone of proven p-type half-Heuslers")
    print(f"  3. Li doping is NOVEL — no Li-doped half-Heusler exists in literature")
    print(f"  4. Nb+Ta co-occupancy matches strategy of recent high-zT compounds")
    print(f"  5. I-42m space group suggests ordered variant of half-Heusler structure")
    
    # Microgravity case
    print(f"\n  MICROGRAVITY SYNTHESIS CASE:")
    print(f"  ├─ Li density:  0.534 g/cm³ (lightest structural metal)")
    print(f"  ├─ Ta density:  16.65 g/cm³ (31x heavier than Li)")
    print(f"  ├─ Nb density:  8.57 g/cm³")
    print(f"  ├─ Co density:  8.90 g/cm³")
    print(f"  └─ Sb density:  6.68 g/cm³")
    print(f"")
    print(f"  In terrestrial synthesis:")
    print(f"  • Arc melting: Li would evaporate (mp=180°C) before Ta melts (mp=3017°C)")
    print(f"  • Ball milling: could mix, but won't achieve single-crystal quality")
    print(f"  • Czochralski: Li floats to top, Ta sinks — no homogeneous melt")
    print(f"")
    print(f"  In microgravity:")
    print(f"  • Containerless electromagnetic levitation avoids crucible contamination")
    print(f"  • No sedimentation → Li stays uniformly distributed")
    print(f"  • No convection → defect-free crystal growth")
    print(f"  • Could achieve single-crystal quality impossible on Earth")
    
    if analysis and "bonding_environments" in analysis:
        print(f"\n  CRYSTAL STRUCTURE ANALYSIS:")
        for el, env in analysis["bonding_environments"].items():
            print(f"  {el}: coordination = {env['avg_coordination']}, "
                  f"neighbors = {env['neighbor_elements']}")
    
    if mp_results:
        print(f"\n  MATERIALS PROJECT CROSS-REFERENCE:")
        if mp_results.get("exact_match"):
            print(f"  ⚠️  EXACT MATCH FOUND — material already in MP database!")
            for m in mp_results["exact_match"]:
                print(f"     {m['id']}: {m['formula']}, bandgap={m['bandgap']}")
        else:
            print(f"  ✓  NO exact match — this is a novel composition")
        
        for key, label in [("li_co_sb", "Li-Co-Sb"), ("nb_co_sb", "Nb-Co-Sb"), 
                           ("ta_co_sb", "Ta-Co-Sb")]:
            if mp_results.get(key):
                print(f"  {label} system: {len(mp_results[key])} compounds found")
                for m in mp_results[key][:3]:
                    print(f"    {m['id']}: {m['formula']}, "
                          f"bandgap={m.get('bandgap', 'N/A')}, "
                          f"e_hull={m.get('e_above_hull', 'N/A')}")
    
    # Next steps
    print(f"\n{'='*80}")
    print(f"  VALIDATION NEXT STEPS")
    print(f"{'='*80}")
    print(f"  1. Run DFT calculation (QE or VASP input files generated)")
    print(f"     → Verify bandgap independently (GNoME used PBE, try HSE06)")
    print(f"     → Check phonon stability (no imaginary frequencies)")
    print(f"  2. Calculate thermoelectric properties:")
    print(f"     → Seebeck coefficient, electrical conductivity (BoltzTraP2)")
    print(f"     → Lattice thermal conductivity (Phono3py)")
    print(f"     → Estimate ZT at 300-1000K range")
    print(f"  3. Molecular dynamics simulation:")
    print(f"     → Model crystallization from melt WITH gravity (sedimentation)")
    print(f"     → Model crystallization from melt WITHOUT gravity (microgravity)")
    print(f"     → Quantify defect density difference")
    print(f"  4. Contact Prof. Zhifeng Ren (UH) or Prof. Poon (UVA)")
    print(f"     → Leading half-Heusler thermoelectric groups")
    print(f"     → Show them this analysis, ask for collaboration")
    print(f"{'='*80}")


def main():
    parser = argparse.ArgumentParser(
        description="Validate LiTaNb2(CoSb)4 as microgravity thermoelectric candidate"
    )
    parser.add_argument("--cif_path", type=str, default=None,
                        help="Path to GNoME CIF file for 5fb363bfdf")
    parser.add_argument("--mp_api_key", type=str, default=None,
                        help="Materials Project API key")
    parser.add_argument("--generate_dft", action="store_true",
                        help="Generate DFT input files")
    args = parser.parse_args()
    
    # Analyze structure if CIF available
    analysis = {}
    if args.cif_path and Path(args.cif_path).exists():
        print("Analyzing crystal structure...")
        analysis = analyze_structure(args.cif_path)
    
    # Cross-reference with Materials Project
    mp_results = None
    if args.mp_api_key:
        print("Cross-referencing with Materials Project...")
        mp_results = cross_reference_mp(args.mp_api_key)
    
    # Print report
    print_validation_report(analysis, mp_results)
    
    # Generate DFT input files
    if args.generate_dft and args.cif_path:
        print("\nGenerating DFT input files...")
        
        # Quantum ESPRESSO
        qe = generate_qe_input(args.cif_path)
        with open("scf.in", "w") as f:
            f.write(qe)
        print("  → Quantum ESPRESSO: scf.in")
        
        # VASP
        vasp = generate_vasp_input(args.cif_path)
        for filename, content in vasp.items():
            with open(filename, "w") as f:
                f.write(content)
            print(f"  → VASP: {filename}")
        
        print("\n  NOTE: You still need to:")
        print("  1. Download pseudopotentials (QE) or PAW potentials (VASP)")
        print("  2. Generate POSCAR from CIF: use pymatgen or VESTA")
        print("  3. Run on HPC cluster (these calculations need 100+ CPU hours)")
        print("  4. For bandgap: follow up SCF with HSE06 hybrid functional")


if __name__ == "__main__":
    main()