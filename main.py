import argparse
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
from utils.metrics import compute_cell_level_scores, compute_actual_errors, evaluate_one_dataset_only, evaluate_multiple_datasets
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

def get_column_profile_by_name(name, profiles):

    for col in profiles:
        if col["column_name"] == name:
            return col
    return None


# --- Main execution ---
def main(mode = "single",dataset_name=None, dataset_group=None, eps_value=0.5, min_samples=1):
    results_path = Path("results")
    datasets_path = Path("datasets")
    dataset_profiles = []
    dataset_names = []
    config = f"Eps : {eps_value} | Min Samples : {min_samples}"
    # d_name = "hospital"
    #-------------------
    # Single table
    #-------------------
    # single_dataset = "hospital"
    if mode == "single":
        if not dataset_name:
            print("No dataset name provided")
            return

        single_profile = results_path / dataset_group/ dataset_name / "column_profile.dictionary"
        if single_profile.exists():
            dataset_profiles.append(str(single_profile))
            dataset_names.append(dataset_name)
        else:
            print(f"Profile not found for single dataset: {dataset_name}")
            return

    #-----------------
    # multi tables
    #-----------------

    if mode == "multi":
        if not dataset_group:
            print("No dataset group provided")
            return

        group_results_path = results_path / dataset_group
        group_data_path = datasets_path / dataset_group
        multi_dataset_profiles = []
        multi_dataset_names = []
        # datasets_path = Path("results/Quintet")
        for folder in group_results_path.iterdir():
            profile_file = folder / "column_profile.dictionary"
            if folder.is_dir() and profile_file.exists():
                multi_dataset_profiles.append(str(profile_file))
                multi_dataset_names.append(folder.name)

        if not multi_dataset_profiles:
            print(f"No dataset profiles found in the datasets folder: {dataset_group}")
            return

        dataset_profiles = multi_dataset_profiles
        dataset_names = multi_dataset_names

        # Auto-detect tables with both dirty.csv and clean.csv
        multi_table_names = [
            folder.name
            for folder in group_data_path.iterdir()
            if folder.is_dir()
               and (folder / "dirty.csv").exists()
               and (folder / "clean.csv").exists()
        ]
    else:
        multi_table_names = []

    # -------------------------------
    # Load all column profiles with unique IDs
    # -------------------------------
    column_profiles = []
    for path, dname in zip(dataset_profiles, dataset_names):
        dataset_column_profiles = load_pickle(path)
        for col in dataset_column_profiles:
            col['dataset_name'] = dname
            clean_col_name = col['column_name']
            #if clean_col_name.startswith(dname + "::"):  # strip dataset prefix
            #    clean_col_name = clean_col_name[len(dname):]
            col['column_name'] = clean_col_name
            col['unique_id'] = f"{clean_col_name}"
        column_profiles.extend(dataset_column_profiles)

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
    # -------------------------------
    # Load all column profiles
    # -------------------------------
    '''
    column_profiles = []
    for path, dataset_name in zip(dataset_profiles + multi_dataset_profiles, dataset_names + multi_dataset_names):
        dataset_column_profiles = load_pickle(path)
        for col in dataset_column_profiles:
            col['dataset_name'] = dataset_name
            col['unique_id'] = f"{dataset_name}::{col['column_name']}"
        column_profiles.extend(dataset_column_profiles)
    '''
    #---------------
    # Load rules
    #---------------
    rules = load_all_rules()
    #rules = load_rules_from_yaml("/Users/veraz/PycharmProjects/DataLakeRuleGeneration/rules.yaml")

    #run k-distance plot to pick eps
    #_ = cluster_columns(column_profiles, min_samples=5, plot_eps=True)

    # Step 1: Cluster columns using full profile features
    # method 1: cluster by all data proile
    #print("\nClustering all columns based on full profile features...")
    #clusters = cluster_columns(column_profiles, n_clusters=5) #kMeans
    #clusters = cluster_columns(column_profiles, eps=0.3, min_samples=5, plot_eps=False)
    #--------------------
    # Clustering
    #--------------------
    for eps in [0.5]:
        clusters = cluster_columns(column_profiles, eps=eps_value, min_samples=min_samples, plot_eps=False)
        print(f"eps={eps} => {len(clusters)} clusters")

        clustered_columns_with_dataset = {}
        for cid, colnames in clusters.items():
            clustered_columns_with_dataset[cid] = list(set(colnames))

        for cid, colnames in clusters.items():
                print(f"  Cluster {cid}: {colnames}")

        # Missing column detection
        clustered_columns = set(col for cluster in clusters.values() for col in cluster)
        all_columns = set(col['column_name'] for col in column_profiles)
        missing = all_columns - clustered_columns

        print(f"Total columns: {len(all_columns)}")
        print(f"Clustered columns: {len(clustered_columns)}")
        print(f"Missing columns: {len(missing)}")
        print(f"Missing column names: {missing}")

        #--------------------------------------
        # Determine shared rules for each cluster
        #-----------------------------------
        print("\nIdentifying shared rules per cluster...")
        # shared = get_shared_rules_per_cluster(rules, column_profiles, clusters, threshold=0.7)
        shared_rules = get_shared_rules_per_cluster_with_sample_cloumn(rules, column_profiles, clusters)
        for cid, rulelist in shared_rules.items():
            print(f"Cluster {cid} shares rules: {rulelist}")

        # Single table evaluation
        if mode == "single":
            if single_profile.exists():
                evaluate_one_dataset_only(rules, shared_rules, clusters, column_profiles, dataset_group, dataset_name, config)

        # Multi table evaluation
        if multi_table_names:
            evaluate_multiple_datasets(rules, shared_rules, clusters, column_profiles, dataset_group, config)

    # -----------------------------------
    # Load ground truth error
    # -----------------------------------
    # datasets_path = Path("datasets")
    raw_dataset = {}
    clean_dataset_dict = {}

    for dataset in dataset_names:
        dirty_csv = datasets_path / dataset / "dirty.csv"
        clean_csv = datasets_path / dataset / "clean.csv"

        if dirty_csv.exists():
            raw_dataset[dataset] = pd.read_csv(dirty_csv)
        if clean_csv.exists():
            clean_dataset_dict[dataset] = pd.read_csv(clean_csv)

    if clean_dataset_dict and raw_dataset:
        actual_errors_by_column = compute_actual_errors(clean_dataset_dict, raw_dataset)
        print("\nGround Truth Error Counts Per Table/Column:")
        for (table, column), row_indices in actual_errors_by_column.items():
            print(f"Table: {table} | Column: {column} | Error Count: {len(row_indices)} | Rows: {row_indices}")
        print(f"\nTotal Ground Truth Error Cells: {sum(len(v) for v in actual_errors_by_column.values())}")

    # Compute metrics




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run clustering and evaluation.")
    parser.add_argument("--mode", choices=["single", "multi"], default="single", help="Mode: single or multi")
    parser.add_argument("--dataset_name", type=str, help="Dataset name for single mode")
    parser.add_argument("--dataset_group", type=str, help="Dataset group folder for multi mode")
    parser.add_argument("--eps", type=float, default=0.5, help="Clustering epsilon value")
    parser.add_argument("--min_samples", type=int, default=1, help="Clustering minimum sample size")
    args = parser.parse_args()

    main(mode=args.mode, dataset_name=args.dataset_name, dataset_group=args.dataset_group, eps_value=args.eps, min_samples=args.min_samples)

    # python3 main.py --mode single --dataset_name hospital --dataset_group Quintet
    # python3 main.py --mode multi --dataset_group Quintet --min_samples 2