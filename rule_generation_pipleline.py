# generate_rules_from_profile.py

import os
import pickle


def generate_rules_for_column(column_name, profile):
    rules = []

    # Rule 1: NOT NULL
    if profile.get('cells_null_mean', 1.0) < 0.01:
        rules.append("MUST NOT be NULL")

    # Rule 2: Uniqueness
    if profile.get('cells_unique_mean', 0) > 0.95:
        rules.append("LIKELY UNIQUE FIELD (Possible Identifier)")

    # Rule 3: Categorical detection
    if profile.get('words_unique_mean', 0) < 2 and profile.get('cells_length_variance', 999) < 5:
        rules.append("CATEGORICAL FIELD (Fixed value set expected)")

    # Rule 4: Free text field
    if profile.get('characters_alphabet_mean', 0) > 3 and profile.get('words_length_mean', 0) > 3:
        rules.append("FREE TEXT FIELD (Descriptive column)")

    # Rule 5: Identifier format
    if profile.get('characters_punctuation_mean', 0) > 0.1 and profile.get('cells_length_variance', 0) < 3:
        rules.append("STRUCTURED FIELD (e.g., ID, Code)")

    # Rule 6: Numeric values with pattern
    if profile.get('characters_numeric_mean', 0) > 1 and profile.get('characters_punctuation_mean', 0) > 0.05:
        rules.append("NUMERIC FIELD with FORMAT (Use regex or pattern validation)")

    # Rule 7: Length constraint
    if profile.get('cells_length_variance', 100) < 1:
        mean_len = round(profile.get('cells_length_mean', 0))
        rules.append(f"FIXED LENGTH FIELD (~{mean_len} characters)")

    return rules


def generate_column_wise_rules(profile):
    top_keywords = profile.get('dataset_top_keywords', {})
    column_names = [col for col in top_keywords if col.isidentifier()]

    column_rules = {}
    for col in column_names:
        col_profile = profile.copy()
        col_profile['dataset_top_keywords'] = {col: top_keywords.get(col, 0)}
        rules = generate_rules_for_column(col, col_profile)
        if top_keywords.get(col, 0) > 0.6:
            rules.append(f"POSSIBLE DOMAIN: {col}")
        column_rules[col] = rules
    return column_rules


def load_profile(profile_path):
    with open(profile_path, 'rb') as f:
        return pickle.load(f)


def save_rules(rules_dict, output_path):
    with open(output_path, 'wb') as f:
        pickle.dump(rules_dict, f)


if __name__ == "__main__":
    dataset_name = "beers"
    results_folder = "results"
    dataset_folder = os.path.join(results_folder, dataset_name)
    profile_path = os.path.join(dataset_folder, "dataset_profile.dictionary")
    rules_output_path = os.path.join(dataset_folder, "dataset_rules.dictionary")

    if not os.path.exists(profile_path):
        print(f"Profile not found at: {profile_path}")
    else:
        profile = load_profile(profile_path)
        column_rules = generate_column_wise_rules(profile)

        # Print
        print(f"\n=== Rule Suggestions for Dataset: {dataset_name} ===")
        for col, rules in column_rules.items():
            print(f"\nColumn: {col}")
            for rule in rules:
                print(f"  - {rule}")

        # Save
        save_rules(column_rules, rules_output_path)
        print(f"\n✔ Rules saved to {rules_output_path}")
