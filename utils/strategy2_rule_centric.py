
import os
from pathlib import Path
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from utils.file_io import load_pickle
from rules.loader import load_rules_from_yaml

# Function to parse single-expression rule strings into related features
import re


def extract_related_features_from_expression(expression):
    # Extract feature-like tokens from the rule expression
    # e.g., distinct == num_rows AND nulls == 0 -> {'distinct', 'num_rows', 'nulls'}
    tokens = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', expression)
    # Filter out logical operators and comparators
    ignore_tokens = {"AND", "OR", "IN", "matches", "==", ">=", "<=", "<", ">", "not", "is"}
    return set(token for token in tokens if token.lower() not in [t.lower() for t in ignore_tokens])


# Main rule-centric clustering function for string-based rules
def rule_centric_clustering(rules, columns):
    assignments = []

    for rule_name, rule_expression in rules.items():
        related_features = extract_related_features_from_expression(rule_expression)

        if not related_features:
            continue

        feature_keys = sorted(related_features)
        rule_vector = np.ones(len(feature_keys))

        column_vectors = []
        column_names = []
        for col in columns:
            vec = []
            for f in feature_keys:
                val = col.get(f, 0.0)
                if val is None or (isinstance(val, float) and np.isnan(val)):
                    val = 0.0
                vec.append(float(val))
            column_vectors.append(vec)
            column_names.append(col['column_name'])

        column_matrix = np.array(column_vectors)
        similarities = cosine_similarity(column_matrix, rule_vector.reshape(1, -1)).flatten()

        for idx, confidence in enumerate(similarities):
            assignments.append({
                'column_name': column_names[idx],
                'assigned_rule': rule_name,
                'confidence_score': round(float(confidence), 4)
            })

    return assignments


# --- Main execution ---
def main():
    base_path = Path("/Users/veraz/PycharmProjects/DataLakeRuleGeneration/results")
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

    rules = load_rules_from_yaml("/Users/veraz/PycharmProjects/DataLakeRuleGeneration/rules.yaml")

    print("\nClustering columns based on rule-centric strategy using only shared features from rule expressions...")
    assignments = rule_centric_clustering(rules, column_profiles)

    cluster_summary = {}
    for assignment in assignments:
        rule = assignment['assigned_rule']
        if rule not in cluster_summary:
            cluster_summary[rule] = []
        cluster_summary[rule].append({
            'column_name': assignment['column_name'],
            'dataset_name': next(
                col['dataset_name'] for col in column_profiles if col['column_name'] == assignment['column_name']),
            'confidence_score': assignment['confidence_score']
        })

    for rule, columns in cluster_summary.items():
        print(f"\nRule Cluster: {rule}")
        for entry in columns:
            print(f"  {entry['dataset_name']}_{entry['column_name']} (confidence: {entry['confidence_score']})")

    print("\nSummary of Rule-Centric Clusters:")
    for rule, cols in cluster_summary.items():
        print(f"{rule}: {len(cols)} columns")


if __name__ == "__main__":
    main()
