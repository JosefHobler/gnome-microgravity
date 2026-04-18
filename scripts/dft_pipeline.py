"""
DFT Validation Pipeline: MgTaNb(SbRh)₃
========================================
Complete setup for independent verification of GNoME predictions.

This script:
1. Loads the CIF from GNoME dataset
2. Generates Quantum ESPRESSO inputs (free, open-source)
3. Generates VASP inputs (needs license)  
4. Sets up BoltzTraP2 for thermoelectric property calculation
5. Provides step-by-step instructions for each calculation

WHAT WE'RE VERIFYING:
- Bandgap: GNoME predicts 0.796-0.884 eV (PBE level)
- Stability: formation energy -0.4853 eV/atom
- Thermoelectric ZT: unknown — this is what we want to calculate

COMPUTE OPTIONS (cheapest to most expensive):
1. Materials Project MPComplete: Submit structure, get DFT free (weeks wait)
2. XSEDE/ACCESS: Free HPC allocation for researchers (apply, ~2 week approval)
3. Google Cloud + QE: ~$50-200 for a full calculation
4. University HPC: Ask a professor (this is your co-founder conversation)

Requirements:
    pip install pymatgen pymatgen-io-vasp numpy ase

Usage:
    # First, extract the CIF:
    # unzip by_id.zip -d cif_files/
    # Find the material ID from seed_crystal_candidates.csv
    
    python dft_pipeline.py --cif_path cif_files/MATERIAL_ID.cif --output_dir ./dft_calc/
"""

import argparse
import os
import numpy as np
from pathlib import Path

try:
    from pymatgen.core import Structure
    from pymatgen.io.vasp.sets import MPRelaxSet, MPStaticSet
    from pymatgen.io.vasp.inputs import Incar, Kpoints, Poscar
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
    HAS_PYMATGEN = True
except ImportError:
    HAS_PYMATGEN = False
    print("Install pymatgen: pip install pymatgen")


# Pseudopotentials for Quantum ESPRESSO (SSSP efficiency library - free)
QE_PSEUDOS = {
    "Mg": {"file": "Mg.pbe-spnl-kjpaw_psl.1.0.0.UPF", "mass": 24.305, "ecutwfc": 40},
    "Ta": {"file": "Ta.pbe-spfn-kjpaw_psl.1.0.0.UPF", "mass": 180.948, "ecutwfc": 50},
    "Nb": {"file": "Nb.pbe-spn-kjpaw_psl.1.0.0.UPF", "mass": 92.906, "ecutwfc": 45},
    "Sb": {"file": "Sb.pbe-n-kjpaw_psl.1.0.0.UPF", "mass": 121.760, "ecutwfc": 45},
    "Rh": {"file": "Rh.pbe-spn-kjpaw_psl.1.0.0.UPF", "mass": 102.906, "ecutwfc": 45},
}


def load_structure(cif_path):
    """Load and analyze the crystal structure."""
    struct = Structure.from_file(cif_path)
    sga = SpacegroupAnalyzer(struct)
    
    print(f"  Formula: {struct.composition.reduced_formula}")
    print(f"  Atoms: {len(struct)}")
    print(f"  Space group: {sga.get_space_group_symbol()} ({sga.get_space_group_number()})")
    print(f"  Crystal system: {sga.get_crystal_system()}")
    print(f"  Lattice: a={struct.lattice.a:.4f} b={struct.lattice.b:.4f} c={struct.lattice.c:.4f}")
    print(f"  Volume: {struct.volume:.2f} Å³")
    print(f"  Density: {struct.density:.4f} g/cm³")
    
    return struct


def generate_qe_scf(struct, output_dir):
    """Generate Quantum ESPRESSO SCF input."""
    
    ecutwfc = max(QE_PSEUDOS.get(str(s.specie), {}).get("ecutwfc", 50) 
                  for s in struct)
    ecutrho = ecutwfc * 8  # PAW needs 8-10x
    
    elements = sorted(set(str(s.specie) for s in struct))
    
    scf = f"""! ============================================================
! MgTaNb(SbRh)3 — SCF calculation (Step 1)
! Verify total energy and electronic structure
! ============================================================
&CONTROL
    calculation = 'scf'
    prefix = 'MgTaNbSbRh'
    outdir = './tmp/'
    pseudo_dir = './pseudo/'
    tprnfor = .true.
    tstress = .true.
    verbosity = 'high'
/

&SYSTEM
    ibrav = 0
    nat = {len(struct)}
    ntyp = {len(elements)}
    ecutwfc = {ecutwfc}
    ecutrho = {ecutrho}
    occupations = 'smearing'
    smearing = 'cold'
    degauss = 0.005
    ! For bandgap: after SCF converges, re-run with nbnd = {len(struct) * 5}
/

&ELECTRONS
    conv_thr = 1.0d-8
    mixing_beta = 0.3
    electron_maxstep = 200
/

CELL_PARAMETERS angstrom
"""
    for i in range(3):
        scf += f"  {struct.lattice.matrix[i][0]:16.10f} {struct.lattice.matrix[i][1]:16.10f} {struct.lattice.matrix[i][2]:16.10f}\n"
    
    scf += "\nATOMIC_SPECIES\n"
    for el in elements:
        pseudo = QE_PSEUDOS.get(el, {})
        scf += f"  {el:4s} {pseudo.get('mass', 1.0):10.4f}  {pseudo.get('file', f'{el}.UPF')}\n"
    
    scf += "\nATOMIC_POSITIONS crystal\n"
    for site in struct:
        el = str(site.specie)
        scf += f"  {el:4s} {site.frac_coords[0]:14.10f} {site.frac_coords[1]:14.10f} {site.frac_coords[2]:14.10f}\n"
    
    # K-point grid — adjust based on cell size
    nk = max(1, int(30 / max(struct.lattice.a, struct.lattice.b, struct.lattice.c)))
    scf += f"\nK_POINTS automatic\n  {nk} {nk} {nk}  0 0 0\n"
    
    filepath = os.path.join(output_dir, "scf.in")
    with open(filepath, "w") as f:
        f.write(scf)
    print(f"  → {filepath}")
    return nk


def generate_qe_bands(struct, output_dir, nk):
    """Generate Quantum ESPRESSO band structure input."""
    
    ecutwfc = max(QE_PSEUDOS.get(str(s.specie), {}).get("ecutwfc", 50) 
                  for s in struct)
    
    elements = sorted(set(str(s.specie) for s in struct))
    nbnd = len(struct) * 5  # enough bands to see conduction band
    
    bands = f"""! ============================================================
! MgTaNb(SbRh)3 — Band structure (Step 2)
! Run AFTER scf.in converges. Uses charge density from SCF.
! ============================================================
&CONTROL
    calculation = 'bands'
    prefix = 'MgTaNbSbRh'
    outdir = './tmp/'
    pseudo_dir = './pseudo/'
    verbosity = 'high'
/

&SYSTEM
    ibrav = 0
    nat = {len(struct)}
    ntyp = {len(elements)}
    ecutwfc = {ecutwfc}
    ecutrho = {ecutwfc * 8}
    occupations = 'smearing'
    smearing = 'cold'
    degauss = 0.005
    nbnd = {nbnd}
/

&ELECTRONS
    conv_thr = 1.0d-8
/

CELL_PARAMETERS angstrom
"""
    for i in range(3):
        bands += f"  {struct.lattice.matrix[i][0]:16.10f} {struct.lattice.matrix[i][1]:16.10f} {struct.lattice.matrix[i][2]:16.10f}\n"
    
    bands += "\nATOMIC_SPECIES\n"
    for el in elements:
        pseudo = QE_PSEUDOS.get(el, {})
        bands += f"  {el:4s} {pseudo.get('mass', 1.0):10.4f}  {pseudo.get('file', f'{el}.UPF')}\n"
    
    bands += "\nATOMIC_POSITIONS crystal\n"
    for site in struct:
        el = str(site.specie)
        bands += f"  {el:4s} {site.frac_coords[0]:14.10f} {site.frac_coords[1]:14.10f} {site.frac_coords[2]:14.10f}\n"
    
    # High-symmetry k-path for triclinic (general path)
    bands += """
K_POINTS crystal_b
5
  0.0000 0.0000 0.0000  30  ! Gamma
  0.5000 0.0000 0.0000  30  ! X
  0.5000 0.5000 0.0000  30  ! M
  0.0000 0.0000 0.5000  30  ! Z
  0.0000 0.0000 0.0000  1   ! Gamma
"""
    
    filepath = os.path.join(output_dir, "bands.in")
    with open(filepath, "w") as f:
        f.write(bands)
    print(f"  → {filepath}")


def generate_qe_dos(struct, output_dir):
    """Generate Quantum ESPRESSO DOS input."""
    
    ecutwfc = max(QE_PSEUDOS.get(str(s.specie), {}).get("ecutwfc", 50) 
                  for s in struct)
    elements = sorted(set(str(s.specie) for s in struct))
    
    # NSCF with dense k-grid for DOS
    nk_dense = max(1, int(50 / max(struct.lattice.a, struct.lattice.b, struct.lattice.c)))
    
    nscf = f"""! ============================================================
! MgTaNb(SbRh)3 — NSCF for DOS (Step 3a)
! Dense k-grid for accurate density of states
! ============================================================
&CONTROL
    calculation = 'nscf'
    prefix = 'MgTaNbSbRh'
    outdir = './tmp/'
    pseudo_dir = './pseudo/'
/

&SYSTEM
    ibrav = 0
    nat = {len(struct)}
    ntyp = {len(elements)}
    ecutwfc = {ecutwfc}
    ecutrho = {ecutwfc * 8}
    occupations = 'tetrahedra'
    nbnd = {len(struct) * 5}
/

&ELECTRONS
    conv_thr = 1.0d-8
/

CELL_PARAMETERS angstrom
"""
    for i in range(3):
        nscf += f"  {struct.lattice.matrix[i][0]:16.10f} {struct.lattice.matrix[i][1]:16.10f} {struct.lattice.matrix[i][2]:16.10f}\n"
    
    nscf += "\nATOMIC_SPECIES\n"
    for el in elements:
        pseudo = QE_PSEUDOS.get(el, {})
        nscf += f"  {el:4s} {pseudo.get('mass', 1.0):10.4f}  {pseudo.get('file', f'{el}.UPF')}\n"
    
    nscf += "\nATOMIC_POSITIONS crystal\n"
    for site in struct:
        el = str(site.specie)
        nscf += f"  {el:4s} {site.frac_coords[0]:14.10f} {site.frac_coords[1]:14.10f} {site.frac_coords[2]:14.10f}\n"
    
    nscf += f"\nK_POINTS automatic\n  {nk_dense} {nk_dense} {nk_dense}  0 0 0\n"
    
    filepath = os.path.join(output_dir, "nscf.in")
    with open(filepath, "w") as f:
        f.write(nscf)
    print(f"  → {filepath}")
    
    # DOS post-processing input
    dos_in = """! ============================================================
! DOS post-processing (Step 3b)
! Run: dos.x < dos.in
! ============================================================
&DOS
    prefix = 'MgTaNbSbRh'
    outdir = './tmp/'
    fildos = 'MgTaNbSbRh.dos'
    degauss = 0.005
    DeltaE = 0.01
/
"""
    filepath = os.path.join(output_dir, "dos.in")
    with open(filepath, "w") as f:
        f.write(dos_in)
    print(f"  → {filepath}")


def generate_vasp_inputs(struct, output_dir):
    """Generate VASP input files using pymatgen's MPRelaxSet."""
    
    vasp_dir = os.path.join(output_dir, "vasp")
    os.makedirs(vasp_dir, exist_ok=True)
    
    # Step 1: Relaxation
    relax_dir = os.path.join(vasp_dir, "01_relax")
    os.makedirs(relax_dir, exist_ok=True)
    relax_set = MPRelaxSet(struct)
    relax_set.write_input(relax_dir)
    print(f"  → {relax_dir}/ (relaxation)")
    
    # Step 2: Static SCF (after relaxation)
    static_dir = os.path.join(vasp_dir, "02_static")
    os.makedirs(static_dir, exist_ok=True)
    static_set = MPStaticSet(struct)
    static_set.write_input(static_dir)
    print(f"  → {static_dir}/ (static SCF)")
    
    # Step 3: HSE06 for accurate bandgap
    hse_dir = os.path.join(vasp_dir, "03_hse06")
    os.makedirs(hse_dir, exist_ok=True)
    
    hse_incar = """SYSTEM = MgTaNb(SbRh)3 - HSE06 bandgap
! Use CONTCAR from 02_static as POSCAR
! Use CHGCAR from 02_static (ICHARG=11)

! HSE06 hybrid functional for accurate bandgap
LHFCALC = .TRUE.
HFSCREEN = 0.2
ALGO = Damped
TIME = 0.4
PRECFOCK = Fast

! Electronic
ENCUT = 520
EDIFF = 1E-5
PREC = Accurate
ISMEAR = 0
SIGMA = 0.05
ICHARG = 11
LORBIT = 11

! This is expensive! ~10-50x more than PBE
! Consider using fewer k-points
LWAVE = .FALSE.
LCHARG = .FALSE.
"""
    with open(os.path.join(hse_dir, "INCAR"), "w") as f:
        f.write(hse_incar)
    
    # Copy POSCAR and KPOINTS from static
    poscar = Poscar(struct)
    poscar.write_file(os.path.join(hse_dir, "POSCAR"))
    kpoints = Kpoints.gamma_automatic([4, 4, 4])
    kpoints.write_file(os.path.join(hse_dir, "KPOINTS"))
    print(f"  → {hse_dir}/ (HSE06 bandgap)")


def generate_boltztrap_instructions(output_dir):
    """Instructions for BoltzTraP2 thermoelectric calculation."""
    
    instructions = """
# ============================================================
# BoltzTraP2: Thermoelectric Property Calculation
# ============================================================
# After completing the VASP or QE electronic structure:
#
# Install:
#   pip install BoltzTraP2
#
# From VASP output:
#   btp2 -vv interpolate -m 5 path/to/vasp/02_static/
#
# This calculates:
#   - Seebeck coefficient S(T, μ)
#   - Electrical conductivity σ/τ(T, μ) 
#   - Electronic thermal conductivity κ_e/τ(T, μ)
#   - Power factor S²σ/τ(T, μ)
#
# For ZT estimation, you also need:
#   - Lattice thermal conductivity (from Phono3py, see below)
#   - Relaxation time τ (from experiment or AMSET)
#
# Quick ZT estimate:
#   ZT = S² × σ × T / (κ_e + κ_L)
#
# If S > 200 μV/K and κ_L < 3 W/mK at 800K, 
# this material is competitive with state-of-art half-Heuslers.
# ============================================================

# ============================================================
# Phono3py: Lattice Thermal Conductivity (κ_L)
# ============================================================
# This is computationally expensive (~100-1000 CPU hours)
#
# Install:
#   pip install phono3py
#
# Step 1: Generate displaced structures
#   phono3py -d --dim 2 2 2 -c POSCAR
#
# Step 2: Run VASP on each displaced structure
#   (this generates FORCES_FC3)
#
# Step 3: Calculate thermal conductivity
#   phono3py --mesh 11 11 11 --fc3 --fc2 --br
#
# Target: κ_L at 300-1000K
# For reference:
#   NbFeSb: κ_L ≈ 10 W/mK (before optimization)
#   Best half-Heuslers after nanostructuring: κ_L ≈ 2-4 W/mK
#   If our material has κ_L < 5 W/mK → very promising
# ============================================================
"""
    filepath = os.path.join(output_dir, "THERMOELECTRIC_INSTRUCTIONS.txt")
    with open(filepath, "w") as f:
        f.write(instructions)
    print(f"  → {filepath}")


def generate_run_scripts(output_dir):
    """Generate PBS/SLURM job submission scripts."""
    
    slurm = """#!/bin/bash
#SBATCH --job-name=MgTaNbSbRh_scf
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=24
#SBATCH --time=12:00:00
#SBATCH --partition=normal
#SBATCH --output=scf_%j.out

# Load modules (adjust for your HPC)
module load quantum-espresso/7.2
# OR: module load vasp/6.4

# Quantum ESPRESSO
mpirun -np 48 pw.x -npool 4 < scf.in > scf.out

# Check convergence
grep "convergence has been achieved" scf.out
grep "highest occupied" scf.out
grep "total energy" scf.out
"""
    filepath = os.path.join(output_dir, "submit_slurm.sh")
    with open(filepath, "w") as f:
        f.write(slurm)
    
    # Cloud run script (for Google Cloud / AWS)
    cloud = """#!/bin/bash
# ============================================================
# Run DFT on Google Cloud (cheapest option without HPC access)
# ============================================================
# Estimated cost: $50-200 for full SCF+bands+DOS
#
# 1. Create a VM:
#    gcloud compute instances create dft-calc \\
#      --machine-type=c2-standard-30 \\
#      --image-family=ubuntu-2204-lts \\
#      --image-project=ubuntu-os-cloud \\
#      --boot-disk-size=100GB
#
# 2. SSH in and install QE:
#    sudo apt update
#    sudo apt install -y quantum-espresso
#
# 3. Download pseudopotentials:
#    mkdir pseudo && cd pseudo
#    wget https://pseudopotentials.quantum-espresso.org/upf_files/Mg.pbe-spnl-kjpaw_psl.1.0.0.UPF
#    wget https://pseudopotentials.quantum-espresso.org/upf_files/Ta.pbe-spfn-kjpaw_psl.1.0.0.UPF
#    wget https://pseudopotentials.quantum-espresso.org/upf_files/Nb.pbe-spn-kjpaw_psl.1.0.0.UPF
#    wget https://pseudopotentials.quantum-espresso.org/upf_files/Sb.pbe-n-kjpaw_psl.1.0.0.UPF
#    wget https://pseudopotentials.quantum-espresso.org/upf_files/Rh.pbe-spn-kjpaw_psl.1.0.0.UPF
#    cd ..
#
# 4. Run calculations in order:
#    mpirun -np 15 pw.x < scf.in > scf.out           # ~1-4 hours
#    mpirun -np 15 pw.x < bands.in > bands.out        # ~1-2 hours  
#    mpirun -np 15 pw.x < nscf.in > nscf.out          # ~2-4 hours
#    mpirun -np 15 dos.x < dos.in > dos.out            # ~minutes
#    bands.x < bands_pp.in > bands_pp.out              # ~minutes
#
# 5. Extract bandgap from DOS output:
#    grep "highest occupied" scf.out
#    # Look at the DOS file for the gap
#
# 6. DON'T FORGET to delete the VM when done!
#    gcloud compute instances delete dft-calc
# ============================================================
"""
    filepath = os.path.join(output_dir, "run_on_cloud.sh")
    with open(filepath, "w") as f:
        f.write(cloud)
    
    print(f"  → {os.path.join(output_dir, 'submit_slurm.sh')}")
    print(f"  → {os.path.join(output_dir, 'run_on_cloud.sh')}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate DFT inputs for MgTaNb(SbRh)3 validation"
    )
    parser.add_argument("--cif_path", type=str, required=True,
                        help="Path to GNoME CIF file")
    parser.add_argument("--output_dir", type=str, default="./dft_calc",
                        help="Output directory for all calculation files")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 60)
    print("  DFT VALIDATION PIPELINE: MgTaNb(SbRh)₃")
    print("=" * 60)
    
    # Load structure
    print("\n  Loading crystal structure...")
    struct = load_structure(args.cif_path)
    
    # Generate QE inputs
    print("\n  Generating Quantum ESPRESSO inputs...")
    qe_dir = os.path.join(args.output_dir, "qe")
    os.makedirs(qe_dir, exist_ok=True)
    nk = generate_qe_scf(struct, qe_dir)
    generate_qe_bands(struct, qe_dir)
    generate_qe_dos(struct, qe_dir)
    
    # Generate VASP inputs
    if HAS_PYMATGEN:
        print("\n  Generating VASP inputs...")
        generate_vasp_inputs(struct, args.output_dir)
    
    # Thermoelectric instructions
    print("\n  Generating thermoelectric calculation instructions...")
    generate_boltztrap_instructions(args.output_dir)
    
    # Run scripts
    print("\n  Generating run scripts...")
    generate_run_scripts(args.output_dir)
    
    # Summary
    print(f"\n{'='*60}")
    print(f"  CALCULATION PLAN")
    print(f"{'='*60}")
    print(f"""
  Step 1: SCF (self-consistent field)
     → Verifies total energy and basic electronic structure
     → Runtime: 1-4 hours on 15-48 cores
     → OUTPUT: total energy, charge density
     
  Step 2: Band structure  
     → Maps electronic bands along high-symmetry k-path
     → Runtime: 1-2 hours
     → OUTPUT: bandgap value (compare with GNoME's 0.80 eV)
     
  Step 3: Density of states
     → Full electronic DOS for thermoelectric analysis
     → Runtime: 2-4 hours
     → OUTPUT: DOS near Fermi level (sharp peak = good Seebeck)
     
  Step 4 (optional): HSE06 hybrid functional
     → More accurate bandgap (PBE typically underestimates by 30-50%)
     → Runtime: 10-50x longer than PBE
     → OUTPUT: corrected bandgap
     
  Step 5: BoltzTraP2 thermoelectric properties
     → Uses DOS from Step 3
     → Runtime: minutes
     → OUTPUT: Seebeck coefficient, power factor vs temperature
     
  Step 6: Phono3py lattice thermal conductivity
     → Most expensive calculation (~100-1000 CPU hours)
     → OUTPUT: κ_L(T) — needed for ZT estimate
     
  WHAT WE WANT TO SEE:
  ✓ PBE bandgap 0.5-1.0 eV (confirms GNoME)
  ✓ Seebeck coefficient > 200 μV/K at 800K  
  ✓ Lattice thermal conductivity < 5 W/mK
  ✓ No imaginary phonon frequencies (dynamically stable)
  
  If all four check out → ZT > 1 is plausible
  → This material is worth pursuing experimentally
""")
    print(f"  All files generated in: {args.output_dir}/")
    print(f"  Cheapest path: Google Cloud VM (~$50-200)")
    print(f"  Free path: email a professor with HPC access")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()