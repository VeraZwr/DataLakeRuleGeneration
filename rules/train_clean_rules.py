import numpy as np
import pandas as pd
import os
from pathlib import Path


def train_clean_rules(cluster_analysis, clustered_features, matcher, rule_profiles, quintet_base_path=None):
    """
    Train cluster-specific rules using rule profiles, based on clean (outlier-filtered) columns
    to avoid error contamination. If quintet_base_path is provided, aggregate all clean.csv files.
    When clean_data is available, outlier filtering is skipped to avoid conflicts.
    Also returns feature_ranges used for each cluster.
    """
    trained_rules = {}

    # Optional: load multiple clean.csv files if provided
    clean_data = None
    if quintet_base_path:
        all_clean_data = []
        base_path = Path(quintet_base_path)
        for dataset_folder in os.listdir(base_path):
            dataset_path = base_path / dataset_folder / "clean.csv"
            if dataset_path.exists():
                df = pd.read_csv(dataset_path)
                #print(f"Loaded {dataset_path} with shape {df.shape}")
                all_clean_data.append(df)
        if all_clean_data:
            clean_data = pd.concat(all_clean_data, ignore_index=True)
            #print(f"Combined clean data shape: {clean_data.shape}")

    for cid, analysis in cluster_analysis.items():
        features = clustered_features[cid]

        # Step 1: Skip outlier filtering if clean_data is available
        if clean_data is None:
            outliers = matcher.detect_outliers_in_cluster(features, cid)
            outlier_indices = {idx for idx, _, _ in outliers}
            clean_features = [f for i, f in enumerate(features) if i not in outlier_indices]
            if len(clean_features) < 3:
                print(f"Fallback: Using all features for cluster {cid} due to too few after outlier removal")
                clean_features = features
        else:
            clean_features = features  # Trust clean data already filtered

        # Step 2: Compute robust (percentile-based) thresholds
        ranges = {}
        for key in set(k for rule in rule_profiles.values() for k in rule["features"]):
            if clean_data is not None and key in clean_data.columns:
                vals = clean_data[key].dropna().values
            else:
                vals = [f.get(key) for f in clean_features if f.get(key) is not None]
            if not vals or len(vals) == 0:
                continue
            ranges[key] = {
                'low': np.percentile(vals, 5),
                'high': np.percentile(vals, 95),
                'mean': np.mean(vals),
                'std': np.std(vals)
            }

        # Step 3: Create rule set for this cluster based on rule_profiles
        rule_set = {}

        for rule_name, rule in rule_profiles.items():
            conditions = {}
            skip_rule = False

            for feature in rule["features"]:
                if feature not in ranges:
                    skip_rule = True
                    break

                low = ranges[feature]["low"]
                high = ranges[feature]["high"]

                if isinstance(rule["conditions"].get(feature), (int, float)):
                    threshold = ranges[feature]["mean"]
                    conditions[feature] = threshold
                elif callable(rule["conditions"].get(feature, None)):
                    op = rule["conditions"][feature]
                    if op.__name__ == "<lambda>":
                        if op(1e10):
                            conditions[feature] = lambda x, h=high: x <= h
                        else:
                            conditions[feature] = lambda x, l=low: x >= l
                    else:
                        conditions[feature] = op
                else:
                    skip_rule = True

            if skip_rule:
                continue

            rule_set[rule_name] = {
                "conditions": conditions,
                "description": rule["description"]
            }

        trained_rules[cid] = {
            "rules": rule_set,
            "feature_ranges": ranges
        }

    return trained_rules