# Discover Microgravity Materials

This pipeline filters the DeepMind GNoME dataset (a database of hundreds of thousands of predicted stable materials) to find novel materials that would specifically benefit from **in-space manufacturing (microgravity)**.

## The Pipeline Architecture

### 1. The Core Filter (`scripts/gnome_microgravity_filter.py`)
This script takes the massive dataset and scores materials based on how much they need microgravity. It favors materials with:
*   **Density Contrast**: Elements with vastly different weights that would normally suffer from sedimentation on Earth.
*   **Containerless Processing**: Materials with extremely high melting points that would contaminate terrestrial crucibles.
*   **Crystal Perfection**: High-symmetry crystals and semiconductors that benefit most from convection-free growth.
*   **Commercial Value**: Filtering into high-value sectors (e.g. quantum, energy, superalloys).

### 2. Business Model Filters (`scripts/filter_seed_crystal.py` & `scripts/filter_bandgap.py`)
Refines microgravity candidates into specific commercial strategies:
*   **Space Seed Model**: Locates materials with moderate density differences and compatible melting points, where a high-quality seed crystal could be grown in space and brought back to Earth to template defect-free terrestrial growth.
*   **Semiconductor Model**: Isolates materials matching valuable bandgap requirements.

### 3. The Validation Engine (`scripts/dft_pipeline.py` & `scripts/validate_lead_compound.py`)
Once a golden candidate is identified, this pipeline automatically generates the supercomputing code and inputs (Quantum ESPRESSO, VASP, BoltzTraP2) needed for Density Functional Theory (DFT) simulations. This mathematically verifies the material properties (like bandgap and thermoelectric factors) before running expensive physical lab experiments.

## Installation & Setup

1. Clone this repository.
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Put the dataset in place. You will need DeepMind's GNoME dataset `stable_materials_summary.csv` placed in the `data/` directory.

## Usage
Run the pipeline scripts in sequential order:

```bash
# 1. Generate the initial microgravity report and candidates
python scripts/gnome_microgravity_filter.py --data_path data/stable_materials_summary.csv --output_csv outputs/microgravity_candidates.csv --output_report outputs/microgravity_report.txt

# 2. Filter for specific viable business models (e.g., Space Seed)
python scripts/filter_seed_crystal.py --input outputs/microgravity_candidates.csv --output outputs/seed_crystal_candidates.csv
```
