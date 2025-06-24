# rules/evaluation.py

def detect_rule_violations(rule, cluster_columns, column_profiles):
    violations = []
    for col in column_profiles:
        if col["column_name"] in cluster_columns and not rule.applies(col):
            violations.append({
                "column_name": col["column_name"],
                "rule": rule.name,
                "reason": rule.description
            })
    return violations


def analyze_rule_in_cluster(rule, cluster_columns, column_profiles):
    passed, violations = [], []

    for col in column_profiles:
        if col["column_name"] not in cluster_columns:
            continue

        if rule.applies(col):
            passed.append(col["column_name"])
        else:
            violations.append({
                "column_name": col["column_name"],
                "rule": rule.name,
                "reason": rule.description
            })

    return {
        "rule": rule.name,
        "cluster_columns": cluster_columns,
        "passed": passed,
        "violated": [v["column_name"] for v in violations],
        "violations": violations,
        "tag": (
            "valid_cluster" if len(passed) == len(cluster_columns) else
            "invalid_cluster" if len(passed) == 0 else
            "mixed_cluster"
        )
    }


def get_shared_rules_per_cluster(rules, column_profiles, clusters, threshold=0.7):
    """
    Determine which rules apply to most columns in each cluster.

    Parameters:
        rules: list of Rule objects (with `applies(col)` method)
        column_profiles: list of column profile dicts
        clusters: dict of {cluster_id: list of column names}
        threshold: float (e.g., 0.7 means rule must apply to 70%+ of cluster)

    Returns:
        shared_rules: dict of {cluster_id: list of rule names}
    """
    shared_rules = {}

    # Build a lookup: column_name -> profile
    col_lookup = {col['column_name']: col for col in column_profiles}

    for cid, colnames in clusters.items():
        applicable_rules = []

        for rule in rules:
            applicable = 0
            for colname in colnames:
                col = col_lookup.get(colname)
                if col and rule.applies(col):
                    applicable += 1
            ratio = applicable / len(colnames)
            if ratio >= threshold:
                applicable_rules.append(rule.name)

        shared_rules[cid] = applicable_rules

    return shared_rules


def detect_cell_errors(rule, column_name, column_data):
    errors = []

    if hasattr(rule, "prepare"):
        rule.prepare(column_data)

    for idx, val in enumerate(column_data):
        if not rule.validate_cell(val):
            errors.append({
                "column": column_name,
                "row_index": idx,
                "value": val,
                "rule": rule.name,
                "reason": rule.description
            })

    return errors


def run_cell_level_checks(rule, cluster_columns, dataset):
    """
    dataset: dict[column_name] -> list of values
    """
    all_errors = []
    for col_name in cluster_columns:
        if col_name in dataset:
            col_data = dataset[col_name]
            errors = detect_cell_errors(rule, col_name, col_data)
            all_errors.extend(errors)
    return all_errors

def detect_cell_errors_in_clusters(clusters, shared_rules, rules, raw_dataset):
    """
    Apply shared rules to each cell in each column of each cluster.

    Returns:
        List of error cell dicts: {column, row_index, value, rule}
    """
    rule_map = {rule.name: rule for rule in rules}
    errors = []

    for cid, colnames in clusters.items():
        for rule_name in shared_rules.get(cid, []):
            rule = rule_map.get(rule_name)
            if not rule or not hasattr(rule, "validate_cell"):
                continue  # skip rules that don't support cell-level validation

            for col in colnames:
                values = raw_dataset.get(col, [])
                if hasattr(rule, "prepare"):
                    rule.prepare(values)

                for i, val in enumerate(values):
                    if not rule.validate_cell(val):
                        errors.append({
                            "column": col,
                            "row_index": i,
                            "value": val,
                            "rule": rule_name,
                            "cluster": cid
                        })

    return errors
