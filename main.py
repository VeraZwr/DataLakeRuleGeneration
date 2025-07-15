import os
import pickle

from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

from rules.loader import load_all_rules, load_rules_from_yaml
from rules.evaluation import get_shared_rules_per_cluster, detect_cell_errors_in_clusters
from utils.clustering import cluster_columns
from pathlib import Path
from utils.file_io import load_pickle, csv_to_column_dict
import pandas as pd
import json
import numpy as np
from rules.cluster_matcher import ClusterBasedColumnMatcher
from rules.train_clean_rules import train_clean_rules
from rules.dictionary_rule import SIMPLE_RULE_PROFILES
from utils.rule_utils import serialize_trained_rules

class NumpyEncoder(json.JSONEncoder):
    """Custom encoder for NumPy types."""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# --- Main execution ---
def main():
    # Load column profiles
    #column_profiles = load_pickle("results/hospital/column_profile.dictionary")

    """"
    datasets_column_profile = [
        "results/beers/column_profile.dictionary",
        "results/flights/column_profile.dictionary",
        "results/hospital/column_profile.dictionary",
        "results/movies_1/column_profile.dictionary",
        "results/rayyan/column_profile.dictionary"

    ]
    """
    base_path = Path("results")
    datasets_column_profile = []
    dataset_names = []
    for dataset_folder in os.listdir(base_path):
        dataset_path = base_path / dataset_folder
        if dataset_path.is_dir():  # Only consider directories
            column_profile_path = dataset_path / "column_profile.dictionary"
            if column_profile_path.exists():  # Ensure the column profile file exists
                datasets_column_profile.append(str(column_profile_path))
                dataset_names.append(dataset_folder)  # Use the directory name as the dataset name
    column_profiles = []
    for path, dataset_name in zip(datasets_column_profile, dataset_names):
        dataset_column_profiles = load_pickle(path)
        for col in dataset_column_profiles:
            col['dataset_name'] = dataset_name  # Add dataset name to each column profile
        column_profiles.extend(dataset_column_profiles)

    #for path in datasets_column_profile:
        # print(f"Loading: {path}")
     #   column_profiles.extend(load_pickle(path))
    # Load all rules (dictionary + custom)
    rules = load_all_rules()
    #rules = load_rules_from_yaml("/Users/veraz/PycharmProjects/DataLakeRuleGeneration/rules.yaml")

    #run k-distance plot to pick eps
    #_ = cluster_columns(column_profiles, min_samples=5, plot_eps=True)

    # Step 1: Cluster columns using full profile features
    print("\nClustering all columns based on full profile features...")
    #clusters = cluster_columns(column_profiles, n_clusters=5) #kMeans
    #clusters = cluster_columns(column_profiles, eps=0.3, min_samples=5, plot_eps=False)
    for eps in [0.3, 0.4, 0.5]:
        clusters = cluster_columns(column_profiles, eps=eps, min_samples=5, plot_eps=False)
        print(f"eps={eps} => {len(clusters)} clusters")
        clustered_columns_with_dataset = {}
        clustered_features = {}

        for cid, colnames in clusters.items():
            clustered_columns_with_dataset[cid] = [
                f"{col['dataset_name']}_{col['column_name']}" for col in column_profiles if col['column_name'] in colnames
            ]
        # Build feature dicts for this cluster
        clustered_features[cid] = [
            col for col in column_profiles if col['column_name'] in colnames
        ]
        #for cid, colnames in clusters.items(): # without table name
        for cid, colnames in clustered_columns_with_dataset.items():
            print(f"  Cluster {cid}: {colnames}")

            # Plug-In: Use the cluster matcher to analyze and generate cluster-specific rules
            matcher = ClusterBasedColumnMatcher()
            cluster_analysis = matcher.analyze_clusters(clustered_features)
            trained_rules = train_clean_rules(cluster_analysis, clustered_features, matcher, SIMPLE_RULE_PROFILES,
                                              quintet_base_path="datasets/Quintet")
            # Save trained rules to JSON
            serialize_trained_rules(trained_rules, "results/trained_rules_clean.json")


            print("Trained rules including feature ranges saved to 'results/trained_rules_with_ranges.json'")
            # Optional: Save or print trained rules
            print(f"\nTrained rules for eps={eps}:")
            for cid, rulelist in trained_rules.items():
                print(f"Cluster {cid}: {list(rulelist.keys())}")

        """
                # Show learned cluster-level rules
                for cid, analysis in cluster_analysis.items():
                    print(f"\n=== Cluster {cid} ===")
                    print(f"Profile: {analysis['profile']}")
                    print(f"Rules: {analysis['rules']}")
                    print(f"Error patterns: {analysis['error_patterns']}")
                    print(f"Feature importance: {analysis['feature_importance']}")
                    print(f"Quality metrics: {analysis['quality_metrics']}")

                cluster_analysis_clean = {
                    int(k): v for k, v in cluster_analysis.items()
                }
                with open("results/cluster_analysis.json", "w") as f:
                    json.dump(cluster_analysis_clean, f, indent=2, cls=NumpyEncoder)
        """

        #detect missing columns
        clustered_columns = set()
        for cluster in clusters.values():
            clustered_columns.update(cluster)

        all_columns = set(col['column_name'] for col in column_profiles)

        missing = all_columns - clustered_columns

        print(f"Total columns: {len(all_columns)}")
        print(f"Clustered columns: {len(clustered_columns)}")
        print(f"Missing columns: {len(missing)}")
        print(f"Missing column names: {missing}")


        # Step 2: Determine shared rules for each cluster
        print("\nIdentifying shared rules per cluster...")
        shared = get_shared_rules_per_cluster(rules, column_profiles, clusters, threshold=0.7)

        for cid, rulelist in shared.items():
            print(f"Cluster {cid} shares rules: {rulelist}")

        # Optional Step 3: Detect violations based on shared rules
        print("\nDetecting violations based on shared rules...")
        col_lookup = {col['column_name']: col for col in column_profiles}

        for cid, rule_names in shared.items():
            for rule in rules:
                if rule.name not in rule_names:
                    continue
                for colname in clusters[cid]:
                    col = col_lookup[colname]
                    if not rule.applies(col):
                        print(f"Violation: Rule '{rule.name}' does not hold for column '{colname}' (Cluster {cid})")
        # Load raw data
        raw_dataset = csv_to_column_dict("datasets/Quintet/hospital/dirty.csv")

        # Detect error cells based on shared rules
        cell_errors = detect_cell_errors_in_clusters(clusters, shared, rules, raw_dataset)

        # Save or print
        print(f"\nDetected {len(cell_errors)} error cells.")

   # df = pd.DataFrame(all_cell_errors)
    #df.to_csv("results/hospital/cell_errors.csv", index=False)

    # -----------------------------------
    # Cluster Visualization Code
    # -----------------------------------


if __name__ == "__main__":
    main()
