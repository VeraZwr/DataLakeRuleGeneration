import pickle
import os

def generate_rules_from_profile(profile_path):
    # Load the dataset profile dictionary
    with open(profile_path, "rb") as f:
        profile = pickle.load(f)

    column_names = profile.get("dataset_column_names_cat", [])
    data_types = profile.get("dominant_data_type", [])

    rules = []

    # Generate rules for each column based on its name and inferred data type
    for col, dtype in zip(column_names, data_types):
        rule_set = []
        col_name_lower = col.lower()

        # Basic rules based on detected data type
        if dtype == "int":
            rule_set.append(f"{col} should contain only integers.")
        elif dtype == "float":
            rule_set.append(f"{col} should contain only float or numeric values.")
        elif dtype == "date":
            rule_set.append(f"{col} should follow a standard date format (e.g., YYYY-MM-DD).")
        elif dtype == "string":
            rule_set.append(f"{col} should contain text data and not consist of only special characters.")

        # Name-based pattern suggestions
        if any(x in col_name_lower for x in ["email", "mail"]):
            rule_set.append(f"{col} should match a valid email format (e.g., name@example.com).")
        elif any(x in col_name_lower for x in ["phone", "mobile"]):
            rule_set.append(f"{col} should be a valid phone number (e.g., 10 or 11 digits).")
        elif "id" in col_name_lower:
            rule_set.append(f"{col} should be a unique identifier with no duplicates.")

        # Optional: Rule based on null value ratio (if available)
        null_mean = profile.get("cells_null_mean", None)
        if null_mean and null_mean > 0.1:
            rule_set.append(f"{col} has a high null rate ({null_mean:.2%}), consider filling or validating missing values.")

        # Append if any rules exist
        if rule_set:
            rules.append({
                "column": col,
                "type": dtype,
                "rules": rule_set
            })

    return rules

# Sample usage
if __name__ == "__main__":
    # Replace this path with your actual profile file
    profile_path = "/Users/veraz/PycharmProjects/DataLakeRuleGeneration/results/beers/dataset_profile.dictionary"

    if not os.path.exists(profile_path):
        print("Profile file not found:", profile_path)
    else:
        rule_output = generate_rules_from_profile(profile_path)

        print("Auto-generated rules:\n")
        for entry in rule_output:
            print(f"Column: {entry['column']} (Detected Type: {entry['type']})")
            for r in entry["rules"]:
                print(f"  - {r}")
            print()
