import os
from pathlib import Path
import numpy as np
from utils.file_io import load_pickle
from utils.clustering import cluster_columns
from rules.modular_rule_system import FeatureExtractor, load_trained_rules

# -----------------------
# Integrated Main Pipeline
# -----------------------
def main():
    # Step 1: Load column profiles from multiple datasets
    base_path = Path("results")
    datasets_column_profile = []
    dataset_names = []
    for dataset_folder in os.listdir(base_path):
        dataset_path = base_path / dataset_folder
        if dataset_path.is_dir():
            column_profile_path = dataset_path / "column_profile.dictionary"
            if column_profile_path.exists():
                datasets_column_profile.append(str(column_profile_path))
                dataset_names.append(dataset_folder)

    column_profiles = []
    for path, dataset_name in zip(datasets_column_profile, dataset_names):
        dataset_column_profiles = load_pickle(path)
        for col in dataset_column_profiles:
            col['dataset_name'] = dataset_name
        column_profiles.extend(dataset_column_profiles)

    print(f"Loaded {len(column_profiles)} columns across {len(dataset_names)} datasets.")

    # Step 2: Load trained rule objects
    trained_rules = load_trained_rules()
    extractor = FeatureExtractor()

    # Step 3: Cluster columns
    clusters = cluster_columns(column_profiles, eps=0.5, min_samples=4, plot_eps=False)
    print(f"\nIdentified {len(clusters)} clusters.")

    # Step 4: Evaluate rules within each cluster
    col_lookup = {col['column_name']: col for col in column_profiles}

    for cid, colnames in clusters.items():
        print(f"\n=== Cluster {cid} ===")
        for colname in colnames:
            col_profile = col_lookup[colname]
            features = extractor.extract(col_profile)
            for rule in trained_rules:
                if not rule.applies(features):
                    print(f"Rule '{rule.name}' fails for column '{col_profile['dataset_name']}_{colname}'")

if __name__ == "__main__":
    main()
