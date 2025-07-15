import yaml
import re

# --------------- rule functions ---------------

def rule_not_null(profile, max_null_ratio=0.0):
    return profile["null_ratio"] <= max_null_ratio, f"Null ratio {profile['null_ratio']:.2f} exceeds {max_null_ratio}"

def rule_unique(profile, min_unique_ratio=1.0):
    return profile["unique_ratio"] >= min_unique_ratio, f"Unique ratio {profile['unique_ratio']:.2f} below {min_unique_ratio}"

def rule_range(profile, min=None, max=None):
    if profile["numeric_min_value"] is None or profile["numeric_max_value"] is None:
        return True, "No numeric values to check"
    if min is not None and profile["numeric_min_value"] < min:
        return False, f"Min {profile['numeric_min_value']} < {min}"
    if max is not None and profile["numeric_max_value"] > max:
        return False, f"Max {profile['numeric_max_value']} > {max}"
    return True, "Range OK"

def rule_pattern(profile, pattern):
    dominant_patterns = list(profile["pattern_histogram"].keys())
    for pat in dominant_patterns:
        test = pat.replace("A", "a").replace("0", "1")
        if not re.match(pattern, test):
            return False, f"Pattern {pat} does not match {pattern}"
    return True, "Patterns OK"

def rule_max_length(profile, max_len):
    return profile["max_len"] <= max_len, f"Max len {profile['max_len']} > {max_len}"

def rule_semantic(profile, domain):
    actual = profile.get("semantic_domain_guess_dosolo")
    if not actual:
        return False, "No semantic guess"
    if actual != domain:
        return False, f"Domain {actual} != {domain}"
    return True, f"Domain OK: {actual}"

# --------------- rule runner ---------------

def validate_column(profile, rules):
    results = []
    for r in rules:
        rule_name = r["rule"]
        params = {k: v for k, v in r.items() if k != "rule"}
        func = globals()[f"rule_{rule_name}"]
        passed, msg = func(profile, **params)
        results.append({
            "rule": rule_name,
            "passed": passed,
            "message": msg
        })
    return results

# --------------- load YAML config ---------------

def load_rules(file="rules.yaml"):
    with open(file, "r") as f:
        return yaml.safe_load(f)

# --------------- example profiler output ---------------

column_profile = {
    "null_ratio": 0.0,
    "unique_ratio": 0.98,
    "numeric_min_value": 10000,
    "numeric_max_value": 99999,
    "max_len": 5,
    "pattern_histogram": {"00000": 100},
    "semantic_domain_guess_dosolo": "zipcode"
}

# --------------- usage example ---------------

if __name__ == "__main__":
    rules = load_rules()
    col_rules = rules["zipcode_column"]
    results = validate_column(column_profile, col_rules)

    for r in results:
        print(r)
