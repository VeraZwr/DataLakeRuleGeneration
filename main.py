import os
import pickle
from matplotlib import pyplot as plt
from rules.loader import load_all_rules, load_rules_from_yaml
from rules.evaluation import get_shared_rules_per_cluster, detect_cell_errors_in_clusters, get_shared_rules_per_cluster_with_sample_cloumn, detect_error_cells, detect_error_cells_across_tables, detect_combined_errors, detect_dynamic_errors
from utils.clustering import cluster_columns
from pathlib import Path
from utils.file_io import load_pickle, csv_to_column_dict
import pandas as pd
import json
import numpy as np
from utils.metrics import compute_cell_level_scores, compute_actual_errors, evaluate_one_dataset_only
from rules.cluster_matcher import ClusterBasedColumnMatcher
from rules.train_clean_rules import train_clean_rules
from rules.dictionary_rule import SIMPLE_RULE_PROFILES
from utils.rule_utils import serialize_trained_rules


d_name = "hospital"
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

def get_column_profile_by_name(name, profiles):
    for col in profiles:
        if col["column_name"] == name:
            return col
    return None


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
    #single table
    datasets_column_profile.append(str(base_path / d_name/"column_profile.dictionary"))
    dataset_names.append(base_path / d_name)
    # all dataset
    '''
    for dataset_folder in os.listdir(base_path):
        dataset_path = base_path / dataset_folder
        if dataset_path.is_dir():  # Only consider directories
            column_profile_path = dataset_path / "column_profile.dictionary"
            #column_profile_path = base_path / "hospital/column_profile.dictionary"
            if column_profile_path.exists():  # Ensure the column profile file exists
                datasets_column_profile.append(str(column_profile_path))
                dataset_names.append(dataset_folder)  # Use the directory name as the dataset name
                '''
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
    # method 1: cluster by all data proile
    #print("\nClustering all columns based on full profile features...")
    #clusters = cluster_columns(column_profiles, n_clusters=5) #kMeans
    #clusters = cluster_columns(column_profiles, eps=0.3, min_samples=5, plot_eps=False)
    for eps in [0.5]:
        clusters = cluster_columns(column_profiles, eps=eps, min_samples=1, plot_eps=False)
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
        for cid, colnames in clusters.items(): # without table name
        #for cid, colnames in clustered_columns_with_dataset.items():
            print(f"  Cluster {cid}: {colnames}")

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
        # shared = get_shared_rules_per_cluster(rules, column_profiles, clusters, threshold=0.7)
        shared_rules = get_shared_rules_per_cluster_with_sample_cloumn(rules, column_profiles, clusters)

        for cid, rulelist in shared_rules.items():
            print(f"Cluster {cid} shares rules: {rulelist}")
#------------------
#-----------------
        # Step 4: Load raw datasets per table
        raw_dataset = {}
        datasets_path = Path("datasets/Quintet")
        for dataset in dataset_names:
            #raw_csv = datasets_path / dataset / "dirty.csv"
            raw_csv = datasets_path / d_name/"dirty.csv"
            if raw_csv.exists():
                raw_dataset[dataset] = pd.read_csv(raw_csv) # need to read string, both are null -> correct: raha, how to compare
                # print(f"detect errors in dataset {raw_csv}")

        # Step 5: Detect error cells
        #errors = detect_combined_errors(clusters, shared_rules, rules, raw_dataset, column_profiles)
        evaluate_one_dataset_only(rules, shared_rules, clusters, column_profiles,d_name)


   # df = pd.DataFrame(all_cell_errors)
    #df.to_csv("results/hospital/cell_errors.csv", index=False)

    # -----------------------------------
    # Cluster Visualization Code
    # -----------------------------------

    # Load clean data
    clean_dataset_dict = {}
    for dataset in dataset_names:
        clean_csv = datasets_path / dataset / "clean.csv"
        if clean_csv.exists():
            clean_dataset_dict[dataset] = pd.read_csv(clean_csv)
    actual_errors_by_column = compute_actual_errors(clean_dataset_dict, raw_dataset)

    # Print error counts
    print("\n Ground Truth Error Counts Per Table/Column:")
    for (table, column), row_indices in actual_errors_by_column.items():
        print(f"Table: {table} | Column: {column} | Error Count: {len(row_indices)} | Rows: {row_indices}")

    print(f"\n Total Ground Truth Error Cells: {sum(len(v) for v in actual_errors_by_column.values())}")

    # Compute metrics




if __name__ == "__main__":
    main()
