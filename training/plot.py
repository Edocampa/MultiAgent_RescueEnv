"""
plot_from_saved_data.py - Utility to regenerate the comparison plot
from saved .npy files, useful after a crash or partial resume.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
from config import TRAIN, CONVERGENCE

def load_and_plot(seed, save_dir):
    out_dir = os.path.join(save_dir, f"seed_{seed}")
    
    # Check if the main directory exists
    if not os.path.exists(out_dir):
        print(f"Error: Directory {out_dir} not found.")
        return

    print(f"\nScanning for results in: {out_dir}")
    
    # List of experiments we expect to find
    experiments_to_load = [
        {"id": 1, "folder": "exp_1_level10_sparse", "label": "Level 10 - SPARSE (no shaping)"},
        {"id": 2, "folder": "exp_2_level5_sparse",  "label": "Level 5 - SPARSE (no shaping)"},
        {"id": 3, "folder": "exp_3_level3_sparse",  "label": "Level 3 - SPARSE (base case)"},
        {"id": 4, "folder": "exp_4_level5_hier",    "label": "Level 5 - HIERARCHICAL (shaping from V*_3)"},
        {"id": 5, "folder": "exp_5_level10_hier",   "label": "Level 10 - HIERARCHICAL (shaping from V*_5)"},
    ]

    loaded_results = []

    # Loop through the expected folders and load the 'sr.npy' file
    for exp in experiments_to_load:
        file_path = os.path.join(out_dir, exp["folder"], "sr.npy")
        if os.path.exists(file_path):
            sr_data = np.load(file_path)
            loaded_results.append({
                "exp_id": exp["id"],
                "label": exp["label"],
                "sr_log": sr_data.tolist()
            })
            print(f"  [OK] Loaded: {exp['folder']}")
        else:
            print(f"  [WARNING] Missing: {file_path}")

    if not loaded_results:
        print("No results found to plot.")
        return

    # Call the exact same plotting function from your main script
    from train_hierarchical import plot_comparison
    print("\nGenerating new comparison.png...")
    plot_comparison(loaded_results, out_dir, seed)

if __name__ == "__main__":
    PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
    
    # Assicurati che questo sia il nome della cartella dove hai salvato la run!
    SAVE_DIR = os.path.join(PROJECT_ROOT, "results_baseline/resultsdifficult_gamma")
    
    # Carica i risultati per i seed definiti nel config (o forzalo a 42 se serve)
    for s in TRAIN["seeds"]:
        load_and_plot(seed=s, save_dir=SAVE_DIR)