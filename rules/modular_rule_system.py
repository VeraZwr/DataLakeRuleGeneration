import yaml
import numpy as np

# -----------------------
# Load YAML Rules (Structure Only)
# -----------------------
def load_rule_definitions(yaml_file="rules/rules.yaml"):
    with open(yaml_file, 'r') as f:
        return yaml.safe_load(f)

# -----------------------
# Feature Extractor
# -----------------------
class FeatureExtractor:
    def __init__(self, version="v1"):
        self.version = version

    def extract(self, column_profile):
        # Example: basic statistical features
        features = {
            "distinct": column_profile.get("distinct_num", 0),
            "num_rows": column_profile.get("row_num", 1),
            "nulls": column_profile.get("null_ratio", 0),
            "unique_ratio": column_profile.get("unique_ratio", 0),
        }
        return features

# -----------------------
# Threshold Storage
# -----------------------
TRAINED_THRESHOLDS = {
    "is_primary_key": {
        "unique_ratio": 0.97,
        "nulls": 0.01
    },
    "is_unique": {
        "unique_ratio": 1.0
    }
}

# -----------------------
# Rule Evaluator Class
# -----------------------
class TrainedRule:
    def __init__(self, name, thresholds):
        self.name = name
        self.thresholds = thresholds

    def applies(self, feature_values):
        for feat, threshold in self.thresholds.items():
            if feat not in feature_values:
                return False
            if feature_values[feat] < threshold:
                return False
        return True

# -----------------------
# Rule Loader (from thresholds)
# -----------------------
def load_trained_rules():
    return [TrainedRule(name, thresholds) for name, thresholds in TRAINED_THRESHOLDS.items()]

# -----------------------
# Main Example Usage
# -----------------------
def main():
    # Step 1: Load YAML high-level rule descriptions
    rule_definitions = load_rule_definitions()
    print("Loaded Rule Definitions:", rule_definitions)

    # Step 2: Load trained rules with thresholds
    trained_rules = load_trained_rules()

    # Step 3: Example column profile
    example_col = {"distinct_num": 100, "row_num": 100, "null_ratio": 0.0, "unique_ratio": 1.0}

    # Step 4: Extract features
    extractor = FeatureExtractor()
    features = extractor.extract(example_col)
    print("Extracted Features:", features)

    # Step 5: Rule evaluation
    for rule in trained_rules:
        result = rule.applies(features)
        print(f"Rule: {rule.name} applies? {result}")

if __name__ == "__main__":
    main()
