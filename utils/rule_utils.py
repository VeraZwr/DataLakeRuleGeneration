def get_features_used_by_rules(rules):
    feature_set = set()
    for rule in rules:
        feature_set.update(rule.used_features)
    return list(feature_set)

def validate_rule(rule):
    missing = []
    if not hasattr(rule, "name"):
        missing.append("name")
    if not hasattr(rule, "used_features"):
        missing.append("used_features")
    if missing:
        raise ValueError(f"Rule {rule.__class__.__name__} missing: {', '.join(missing)}")

def serialize_trained_rules(trained_rules, save_path):
    clean_rules = {}
    for cid, cluster_info in trained_rules.items():
        cid_int = int(cid)
        clean_rules[cid_int] = {
            "feature_ranges": {
                k: {stat: float(v) for stat, v in stats.items()}
                for k, stats in cluster_info.get("feature_ranges", {}).items()
            },
            "rules": {
                rule_name: {
                    "description": rule_data["description"],
                    "conditions": {f: "lambda" if callable(c) else c
                                   for f, c in rule_data["conditions"].items()}
                }
                for rule_name, rule_data in cluster_info.get("rules", {}).items()
            }
        }

    import json
    with open(save_path, "w") as f:
        json.dump(clean_rules, f, indent=2)

    print(f"Trained rules saved at {save_path}")

