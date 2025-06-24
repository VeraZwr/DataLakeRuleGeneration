import pickle
from rules.loader import load_all_rules
from rules.evaluation import get_shared_rules_per_cluster, detect_cell_errors_in_clusters
from utils.clustering import cluster_columns
from utils.file_io import load_pickle, csv_to_column_dict
import pandas as pd

# --- Main execution ---
def main():
    # Load column profiles
    column_profiles = load_pickle("results/hospital/column_profile.dictionary")

    # Load all rules (dictionary + custom)
    rules = load_all_rules()

    # Step 1: Cluster columns using full profile features
    print("\nClustering all columns based on full profile features...")
    clusters = cluster_columns(column_profiles, n_clusters=5)

    for cid, colnames in clusters.items():
        print(f"  Cluster {cid}: {colnames}")

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


if __name__ == "__main__":
    main()
