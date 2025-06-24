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
