# generate_rules_from_profile.py

import os
import pickle



def load_profile(profile_path):
    with open(profile_path, "rb") as f:
        return pickle.load(f)
"""    
# Rule templates for generation
FULL_RULE_TEMPLATES = {
    # Cardinality Rules
    'is_primary_key': lambda p: p['dataset_top_keywords'].get('id', 0) == 1.0,
    'is_nullable': lambda p: p.get('characters_unique_mean', 0) < 0.02,
    'is_constant': lambda p: p.get('words_unique_mean', 1) < 0.1,
    'has_low_cardinality': lambda p: p.get('words_unique_mean', 1) < 0.3,

    # Value Distribution Rules
    'value_in_range': lambda p: p.get('characters_numeric_mean', 0) > 1.0,
    'value_histogram_match': lambda p: p.get('characters_numeric_variance', 0) > 1.0,
    'top_n_frequency_check': lambda p: p.get('characters_alphabet_mean', 0) > 3.0,
    'quartile_thresholds': lambda p: p.get('characters_alphabet_variance', 0) > 30,
    'benford_conformity': lambda p: p.get('characters_numeric_mean', 0) > 1.5,

    # Data Type / Format / Pattern Rules
    'matches_regex': lambda p: p.get('characters_alphabet_variance', 0) > 40,
    'length_within': lambda p: p.get('characters_alphabet_mean', 0) > 5,
    'decimal_precision': lambda p: p.get('characters_numeric_variance', 0) < 3,
    'semantic_class_is': lambda p: 'abv' in p['dataset_top_keywords'],
    'domain_is': lambda p: 'state' in p['dataset_top_keywords']
}
"""

def rule_with_params(profile):
    return {
        # Cardinality Rules
        'is_unique': ("is_unique",{"min_uniqueness_ratio": 1.0}) if profile.get('distinct_values', 0) / max(profile.get('num_rows', 1), 1) >= 1.0 else None,
        #'is_primary_key': ("is_primary_key", None) if profile['dataset_column_names_cat'].get('id', 0) == 1.0 and profile.get('dataset_column_index', 0) < 4 else None,
        'is_nullable': ("is_nullable", {"null_threshold": 0.02}) if profile.get('characters_unique_mean', 0) < 0.02 else None,
        'is_constant': ("is_constant", {"unique_word_mean": profile.get('words_unique_mean')}) if profile.get('words_unique_mean', 1) < 0.1 else None,
        'has_low_cardinality': ("has_low_cardinality", {"threshold": 0.3}) if profile.get('words_unique_mean', 1) < 0.3 else None,

        # Value Distribution Rules
        'value_in_range': ("value_in_range", {"min": 0, "max": profile.get('characters_numeric_mean', 0) * 3}) if profile.get('characters_numeric_mean', 0) > 1.0 else None,
        'value_histogram_match': ("value_histogram_match", {"variance_threshold": 1.0}) if profile.get('characters_numeric_variance', 0) > 1.0 else None,
        'top_n_frequency_check': ("top_n_frequency_check", {"alphabet_mean": profile.get('characters_alphabet_mean')}) if profile.get('characters_alphabet_mean', 0) > 3.0 else None,
        'quartile_thresholds': ("quartile_thresholds", {"alphabet_variance": profile.get('characters_alphabet_variance')}) if profile.get('characters_alphabet_variance', 0) > 30 else None,
        'benford_conformity': ("benford_conformity", {"numeric_mean": profile.get('characters_numeric_mean')}) if profile.get('characters_numeric_mean', 0) > 1.5 else None,

        # Data Type / Format / Pattern Rules
        'matches_regex': ("matches_regex", {"alphabet_variance": profile.get('characters_alphabet_variance')}) if profile.get('characters_alphabet_variance', 0) > 40 else None,
        'length_within': ("length_within", {"min_length": 3, "max_length": profile.get('characters_alphabet_mean')}) if profile.get('characters_alphabet_mean', 0) > 5 else None,
        'decimal_precision': ("decimal_precision", {"max_precision": 3}) if profile.get('characters_numeric_variance', 0) < 3 else None,
        'semantic_class_is': ("semantic_class_is", {"class": "percentage"}) if 'abv' in profile['dataset_top_keywords'] else None,
        'domain_is': ("domain_is", {"domain": "US States"}) if 'state' in profile['dataset_top_keywords'] else None
    }

def generate_column_wise_rules(profile):
    column_rules = {}
    for col in profile.get('dataset_top_keywords', {}):
        rules_with_params = rule_with_params(profile)
        column_rules[col] = [r for r in rules_with_params.values() if r]
    return column_rules

def save_rules(rules, output_path):
    with open(output_path, "wb") as f:
        pickle.dump(rules, f)

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
            for rule, params in rules:
                if params:
                    print(f"  - {rule}: {params}")
                else:
                    print(f"  - {rule}")

        # Save
        save_rules(column_rules, rules_output_path)
        print(f"\n✔ Rules saved to {rules_output_path}")

"""
# Rule generator function without parameter
def generate_all_rules_from_profile(profile, rule_templates):
    triggered = []
    for rule_name, condition in rule_templates.items():
        try:
            if condition(profile):
                triggered.append(rule_name)
        except Exception:
            continue
    return triggered

# Simulated function: Load the profile dictionary from a file
def load_profile(profile_path):
    with open(profile_path, "rb") as f:
        return pickle.load(f)

# Apply all rule templates to each column
def generate_column_wise_rules(profile):
    column_rules = {}
    columns = profile.get('dataset_top_keywords', {}).keys()
    for column in columns:
        # For each column, pass entire profile to rule check
        triggered_rules = generate_all_rules_from_profile(profile, FULL_RULE_TEMPLATES)
        column_rules[column] = triggered_rules
    return column_rules

# Save rules dictionary to a file
def save_rules(rules, output_path):
    with open(output_path, "wb") as f:
        pickle.dump(rules, f)

# Main execution block
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
"""
