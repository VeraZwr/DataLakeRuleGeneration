# main.py

import pickle
from rules.loader import load_all_rules
from profiling.clustering import cluster_columns

# --- Load column profiles ---
def load_column_profiles(path):
    with open(path, "rb") as f:
        return pickle.load(f)

# --- Main execution ---
def main():
    # Load column profiles
    column_profiles = load_column_profiles("results/hospital/column_profile.dictionary")

    # Load all rules (both dictionary + custom rules)
    rules = load_all_rules()

    # Cluster columns based on each rule's relevant features
    for rule in rules:
        print(f"\n🔍 Clustering for rule: {rule.name}")
        print(f"🧬 Using features: {rule.used_features}")

        clusters = cluster_columns(column_profiles, rule.used_features, n_clusters=3)

        if not clusters:
            print("⚠️  Not enough data to cluster.")
            continue

        for cid, colnames in clusters.items():
            print(f"  Cluster {cid}: {colnames}")

if __name__ == "__main__":
    main()
