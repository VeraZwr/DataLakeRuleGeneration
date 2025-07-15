import re
import yaml

# --- Input: your clustered output with profiling stats ---

# --- Auto-labeling function ---
def auto_label_cluster(cluster):
    features = cluster["features"]
    if features["avg_distinct_ratio"] > 0.95 and features["avg_null_ratio"] < 0.05:
        return "ID_like"
    elif features["avg_data_type"] == "numeric":
        return "Numeric_like"
    elif features["avg_data_type"] == "string" and features["avg_length"] > 25:
        return "Text_like"
    elif features["avg_data_type"] == "string":
        for name in cluster["members"]:
            if re.search(r"date|time", name, re.IGNORECASE):
                return "Date_like"
        return "Text_like"
    else:
        return "Other"

# --- Merge auto labels with defaults and overrides ---
def build_yaml_config(auto_labeled, overrides=None):
    if overrides is None:
        overrides = {}

    config = {
        "cluster_profiles": {
            "ID_like": {"defaults": [{"rule": "is_unique"}]},
            "Numeric_like": {"defaults": [{"rule": "value_in_range"}]},
            "Text_like": {
                "defaults": [
                    {"rule": "is_nullable"},
                    {"rule": "length_within", "min_length": 2, "max_length": 255}
                ]
            },
            "Date_like": {"defaults": [{"rule": "date_format"}]},
            "Other": {"defaults": []}
        },
        "clusters": [],
        "overrides": overrides
    }

    for cl in auto_labeled:
        config["clusters"].append({
            "cluster_id": cl["cluster_id"],
            "cluster_profile": cl["profile"],
            "members": cl["members"]
        })

    return yaml.dump(config)

# --- Run auto-labeler ---
auto_labeled = []
for cluster in clusters:
    label = auto_label_cluster(cluster)
    auto_labeled.append({
        "cluster_id": cluster["cluster_id"],
        "profile": label,
        "members": cluster["members"]
    })

# --- Optional manual overrides ---
overrides = {
    "hospital_provider_number": [
        {"rule": "is_primary_key"}
    ]
}

# --- Build final YAML ---
yaml_config = build_yaml_config(auto_labeled, overrides)
print("--- Final Cluster Rule YAML ---")
print(yaml_config)
