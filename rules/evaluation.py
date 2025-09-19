# rules/evaluation.py
import pandas as pd
import re
import os
import pandas as pd
from spellchecker import SpellChecker
from uszipcode import SearchEngine
import time
# search = SearchEngine()
import spacy
from spellchecker import SpellChecker

import requests

search_cache = {}  # Cache results to avoid repeated API calls
valid_whitelist = set()     # Known valid names (found locally or online)
error_cache = set()
def exists_online_wikipedia(query):
    """Check if the name exists on Wikipedia."""
    query = query.lower().strip()

    # Use cached result if available
    if query in search_cache:
        #print(f"[DEBUG] Cached Wikipedia result for: {query}")
        return search_cache[query]

    try:
        response = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "query",
                "list": "search",
                "srsearch": query,
                "format": "json"
            },
            timeout=10
        )
        if response.status_code == 200:
            data = response.json()
            found = len(data.get("query", {}).get("search", [])) > 0
            search_cache[query] = found
            return found

    except Exception as e:
        print(f"[DEBUG] Wikipedia API error: {e}")

    search_cache[query] = False
    return False

# Abbreviation mapping
ABBREVIATIONS = {
    r"\bst\b": "saint",
    r"\bft\b": "fort",
    r"\bmt\b": "mount"
}

def normalize_abbreviations(text):
    text = text.lower()
    text = re.sub(r"\bst\b", "saint", text)
    text = re.sub(r"\bmt\b", "mount", text)
    return text

nlp = spacy.load("en_core_web_sm")
spell = SpellChecker()

def has_spelling_errors(val):
    """Check spelling locally, then online, with caching."""
    if not isinstance(val, str) or not val.strip():
        return False

    text = normalize_abbreviations(val)

    # Check caches first
    if text in valid_whitelist:
        return False
    if text in error_cache:
        return True

    # Local spell check
    doc = nlp(text)
    misspelled = [
        token.text for token in doc
        if token.ent_type_ not in ("PERSON", "ORG", "GPE", "FAC")
        and token.text.lower() not in spell
    ]
    if not misspelled:
        valid_whitelist.add(text)
        return False

    # Wikipedia check if misspelled
    if exists_online_wikipedia(text):
        valid_whitelist.add(text)
        spell.word_frequency.add(text)
        return False

    # Mark as error
    #print(f"[DEBUG] Spelling error detected for: {val}")
    error_cache.add(text)
    return True


cities_df = pd.read_csv("/Users/veraz/PycharmProjects/DataLakeRuleGeneration/database_US/uscities.csv")  # file from simplemaps



def normalize_city(name):
    name = str(name).lower().strip()

    # Remove dots first
    name = re.sub(r'\.', '', name)
    name = re.sub(r'\s+', ' ', name)  # Collapse spaces

    # Replace abbreviations
    for abbr, full in ABBREVIATIONS.items():
        name = re.sub(abbr, full, name)

    return name
def normalize_state(name):
    """Normalize state input for comparison."""
    name = str(name).strip()
    name = re.sub(r'\.', '', name)  # Remove periods
    return name

def normalize_county(name):
    return str(name).lower().strip()

cities_df["city_norm"] = cities_df["city"].apply(normalize_city)
cities_df["county_norm"] = cities_df["county_name"].apply(normalize_county)

us_cities = set(cities_df["city_norm"])
us_state_ids = set(cities_df["state_id"].str.upper())
us_state_names = set(cities_df["state_name"].str.lower())
us_counties = set(cities_df["county_norm"])

cities_df["zips"] = cities_df["zips"].fillna("").astype(str)
us_zipcodes = set()
for z in cities_df["zips"]:
    if " " in z:  # multiple ZIPs in one cell
        us_zipcodes.update([zip.strip() for zip in z.split(" ")])
    elif z:
        us_zipcodes.add(z.strip())

def is_us_city(city_name):
    return normalize_city(city_name) in us_cities


def is_us_state(state_value):
    """
    Validate if the given state value is a valid US state
    (matches abbreviation or full name).
    """
    state_value_norm = normalize_state(state_value)

    # Check abbreviation (e.g., "FL")
    if state_value_norm.upper() in us_state_ids:
        return True

    # Check full state name (e.g., "Florida")
    if state_value_norm.lower() in us_state_names:
        return True

    return False

def is_us_zip(zip_code):
    """Validate if ZIP exists in US cities data."""
    return str(zip_code).strip() in us_zipcodes

def is_us_county(county_name):
    return normalize_county(county_name) in us_counties


def is_phone_column_by_name(column_name):
    """Check if the column name suggests it's a phone number."""
    return bool(re.search(r'(phone|mobile|contact|cell|fax|tel)', column_name, re.IGNORECASE))
phone_pattern = re.compile(
    r'^\+?1?\s*\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}$'
)
#def is_phone_column_by_content(series):
#    """Check if a column's values look like phone numbers."""
#    non_empty = series.dropna().astype(str)
#    if non_empty.empty:
#        return False

#    match_ratio = non_empty.apply(lambda x: bool(phone_pattern.match(x))).mean()

    # If >70% of rows match phone pattern, consider it a phone column
#    return match_ratio > 0.7
#def is_phone_column(column_name, series):
#    return is_phone_column_by_name(column_name) or is_phone_column_by_content(series)


def count_decimals(value):
    """Count decimal places in a numeric value."""
    try:
        s = str(value)
        if '.' in s:
            return len(s.split('.')[1].rstrip('0'))  # remove trailing zeros
        return 0
    except Exception:
        return 0

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


def get_shared_rules_per_cluster_with_sample_cloumn(rules, column_profiles, clusters):
    shared_rules = {}

    for cid, colnames in clusters.items():
        cluster_colnames = set(colnames)
        applicable_rules = []

        for rule in rules:
            sample_columns = getattr(rule, "sample_column", [])
            # Normalize to list of strings
            if isinstance(sample_columns, str):
                sample_columns = [sample_columns]
            elif isinstance(sample_columns, list):
                # Flatten any nested lists
                flattened = []
                for item in sample_columns:
                    if isinstance(item, list):
                        flattened.extend(item)
                    else:
                        flattened.append(item)
                sample_columns = flattened
            else:
                sample_columns = []

            # Now sample_columns is a list of strings
            #for col in cluster_colnames: print(f"[Debug cluster], {col}")
            #for col in sample_columns: print(f"[Debug sample], {col}")
            if any(col in cluster_colnames for col in sample_columns):
                #print(f"-------------{rule.name}  {sample_columns.col}")
                applicable_rules.append(rule.name)
                print(applicable_rules)

        shared_rules[cid] = applicable_rules

    return shared_rules



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

def detect_error_cells(clusters, shared_rules, rules, column_data_lookup, table_name):
    all_errors = []
    rule_lookup = {rule.description: rule for rule in rules}

    for cid, colnames in clusters.items():

        applicable_rule_descriptions = shared_rules.get(cid, [])

        for rule_desc in applicable_rule_descriptions:
            rule = rule_lookup.get(rule_desc)
            print(rule)
            if not rule:
                continue

            for colname in colnames:
                if colname not in column_data_lookup:
                    continue

                col_data = column_data_lookup[colname]
                rule.prepare(col_data)  # if needed

                error_indices = []
                for idx, val in col_data.items():
                    if not rule.validate_cell(val):
                        error_indices.append(idx)

                if error_indices:
                    all_errors.append({
                        "table": table_name,
                        "cluster": cid,
                        "column": colname,
                        "rule": rule_desc,
                        "error_indices": error_indices
                    })

    return all_errors


def detect_error_cells_across_tables(clusters, shared_rules, rules, raw_dataset):
    errors = []
    rule_lookup = {rule.description: rule for rule in rules}

    for cid, colnames in clusters.items():
        print(f"DEBUG: Processing cluster {cid} with columns: {colnames}")
        for colname in colnames:
            try:
                table, column = colname.split("::", 1)
            except ValueError:
                print(f"⚠️ Skipping malformed column name: {colname}")
                continue
            df = raw_dataset.get(table)
            if df is None:
                print(f"⚠️ No dataset found for table: {table}")
                continue
            if column not in df.columns:
                print(f"⚠️ Column {column} not found in dataset {table}")
                continue

            print(f"✅ Column {column} from table {table} will be checked with rules: {shared_rules.get(cid)}")
        applicable_rule_descriptions = shared_rules.get(cid, [])

        for rule_desc in applicable_rule_descriptions:
            rule = rule_lookup.get(rule_desc)
            if not rule:
                continue

            for table_column in colnames:
                # Split into table + column name
                if "_" not in table_column:
                    continue  # skip malformed names
                table_name, column_name = table_column.split("::", 1)

                df = raw_dataset.get(table_name)
                if df is None or column_name not in df.columns:
                    continue  # skip if data not found

                column_data = df[column_name]

                rule.prepare(column_data)  # optional if applicable

                error_indices = [
                    idx for idx, val in column_data.items()
                    if not rule.validate_cell(val)
                ]

                if error_indices:
                    errors.append({
                        "table": table_name,
                        "cluster": cid,
                        "column": column_name,
                        "rule": rule_desc,
                        "error_indices": error_indices
                    })

    return errors

def statistical_cell_detector(series):
    errors = []
    if series.dtype.kind in "biufc":  # numeric columns
        mean, std = series.mean(), series.std()
        for idx, val in series.items():
            if not pd.isna(val) and abs(val - mean) > 3 * std:
                errors.append(idx)
    elif series.dtype == 'object':
        freq = series.value_counts(normalize=True)
        low_freq_values = set(freq[freq < 0.01].index)
        for idx, val in series.items():
            if val in low_freq_values:
                errors.append(idx)
    return errors

def detect_combined_errors(clusters, shared_rules, rules, raw_dataset, column_profiles=None):
    errors = []
    rule_lookup = {rule.name: rule for rule in rules}
    #print("error detection started")

    for cid, colnames in clusters.items():
        print(f"DEBUG: Processing cluster {cid} with columns: {colnames}")

        for colname in colnames:
            table, column = colname.split("::", 1)
            #print("DEBUG raw_dataset keys:", list(raw_dataset.keys()))

            df = raw_dataset.get(table)
            if df is None or column not in df.columns:
                continue
            series = df[column]
            col_errors = set()

            # Get the profile for this column if available, for the dynamic rules
            current_profile = None
            if column_profiles:
                for p in column_profiles:
                    if p.get("column_name") == column and str(p.get("dataset_name")).endswith(table):
                        current_profile = p
                        break
            for col in column_profiles[:10]:
                print("DEBUG profile:", col['dataset_name'], col['column_name'], "=>", col['unique_id'])

            for rule_name in shared_rules.get(cid, []):
                rule = rule_lookup[rule_name]
                # --- Special case for "is_not_nullable" ---
                if rule.name == "is_not_nullable":
                    null_errors = series[series.isna()].index
                    col_errors.update(null_errors)
                else:
                    if rule.name:
                        print(rule_name)
                        dominant_pattern = None
                        expected_data_type = None
                        max_decimal = None
                        semantic_domain = None

                        # --- Case 1: Flat conditions ---
                        if hasattr(rule, "conditions"):

                            for feature in rule.features:
                                if feature in rule.conditions:
                                    condition_value = rule.conditions[feature]
                                    print(f"[DEBUG] Found condition '{feature}' in flat conditions: {condition_value}")

                                    # Decimal
                                    if rule.name in ("decimal_precision"):  # Numeric or functional condition
                                        # Ensure the column is numeric
                                        max_decimal = condition_value
                                        print(f"[DEBUG] Using decimal from condition '{feature}' | 'max_decimal' is {max_decimal}")

                                    elif "matches_regex" in rule.name  and colname in rule.sample_column:
                                        print("find matches regex")
                                        dominant_pattern = condition_value
                                        print(f"[DEBUG] Using pattern from condition '{feature}'")

                                    # Data type checks
                                    elif feature == "basic_data_type":
                                        expected_data_type = condition_value
                                        print(
                                            f"[DEBUG] Expected data type '{expected_data_type}' for rule '{rule.name}'")

                                    elif feature == "semantic_domain":
                                        semantic_domain = condition_value
                                        print(
                                            f"[DEBUG] Expected data type '{semantic_domain}' for rule '{rule.name}'")



                                    elif "spell_check" in rule.conditions:
                                        col_errors = set()
                                        for idx, val in series.items():
                                            try:
                                                # Use the unified checker (spell check + Wikipedia fallback)
                                                if has_spelling_errors(val):
                                                    print(f"[DEBUG] Spelling error for '{val}' "
                                                          f"| Column: {column} | Index: {idx}")
                                                    col_errors.add(idx)
                                                else:
                                                    continue
                                                    #print(f"[DEBUG] Spelling OK for '{val}'")
                                            except Exception as e:
                                                print(f"[DEBUG] Error during spell check for '{val}': {e}")

                                    # Handle other numeric/callable conditions here
                                    elif callable(condition_value):
                                        col_errors = set()
                                        for idx, val in series.items():
                                            try:
                                                if not condition_value(val):
                                                    #print(f"[DEBUG] Condition '{feature}' failed | "
                                                    #      f"Column: {column} | Index: {idx} | Value: {val}")
                                                    col_errors.add(idx)
                                            except Exception as e:
                                                print(f"[DEBUG] Error applying condition '{feature}': {e}")
                                    else:
                                        col_errors = set()
                                        for idx, val in series.items():
                                            if str(val) != str(condition_value):
                                                #print(f"[DEBUG] Exact match failed for '{feature}' | "
                                                #      f"Column: {column} | Index: {idx} | Value: {val}")
                                                col_errors.add(idx)

                        # --- Case 2: Multiple entries ---
                        #elif hasattr(rule, "sample_column") and isinstance(rule.sample_column, list):
                        #    for entry in shared_rules.get(rule.name, []):
                        #        if colname in entry.get("sample_column", []):
                        #            for cond_key, cond_value in entry["conditions"].items():
                        #                print(
                        #                    f"[DEBUG] Found condition '{cond_key}' in entries for {colname}: {cond_value}")
                        #                if cond_key.lower().endswith("pattern"):
                        #                    dominant_pattern = cond_value
                        #                elif cond_key == "basic_data_type":
                        #                    expected_data_type = cond_value
                        #                elif cond_key == "max_decimal":
                        #                    max_decimal = cond_value
                        #                    print(f"check{max_decimal}")

                        # --- Case 3: Column profile fallback ---
                        #if not dominant_pattern and current_profile and current_profile.get("dominant_pattern"):
                        #    dominant_pattern = current_profile["dominant_pattern"]
                        #    print(f"Found dominant_pattern from column profile for {column}: {dominant_pattern}")

                        # --- Case 4: No rule provided in conditions ---
                        #if not hasattr(rule, "conditions"):
                        #    non_null_values = [v for v in series if pd.notna(v)]
                        #    if non_null_values:
                        #        condition_value = rule.conditions(str(non_null_values[0]))
                        #        print(f"Inferred feature condition for column {column}: {condition_value}")

                        # --- Apply regex checks ---
                        if dominant_pattern:
                            col_errors = set()
                            rule.regex = re.compile(dominant_pattern)
                            for idx, val in series.items():
                                if pd.notna(val):
                                    val_str = str(val).strip()

                                    # --- Phone handling ---
                                    if is_phone_column_by_name(series.name.lower()):  # Column name check
                                        if not phone_pattern.fullmatch(val_str):
                                            col_errors.add(idx)
                                            # print(f"[DEBUG] '{val_str}' is not a valid phone number")
                                    else:
                                        # Regular pattern check
                                        if not rule.regex.fullmatch(val_str):
                                            col_errors.add(idx)
                                            #print(f"[DEBUG] '{val_str}' does not match {dominant_pattern}")

                        # --- Apply decimal_precision checks ---
                        if max_decimal:
                            col_errors = set()
                            numeric_series = pd.to_numeric(series, errors='coerce')
                            for idx, val in numeric_series.items():
                                original_val = series[idx]
                                if pd.isna(val) and pd.notna(original_val):
                                    #print(f"[DEBUG] Non-numeric value detected | Column: {column} | "
                                    #      f"Index: {idx} | Value: {original_val}")
                                    col_errors.add(idx)
                                else:
                                    decimals = count_decimals(val)
                                    if decimals > max_decimal:
                                        #print(f"[DEBUG] Decimal precision failed | Column: {column} | "
                                        #      f"Index: {idx} | Value: {val} | Decimals: {decimals}")
                                     col_errors.add(idx)

                        if semantic_domain:
                            col_errors = set()
                            for idx, val in series.items():
                                #taking too long to use search.by_city
                                if semantic_domain == "city" and not is_us_city(val):
                                    col_errors.add(idx)
                                    #print(f"City should not be: {val}")
                                elif semantic_domain == "state" and not is_us_state(val):
                                    col_errors.add(idx)
                                    #print(f"State should not be: {val}")
                                elif semantic_domain == "region" and not is_us_zip(val):
                                    col_errors.add(idx)
                                    #print(f"Zip should not be: {val}")
                                elif semantic_domain == "county" and not is_us_county(val):
                                    col_errors.add(idx)
                                # print(f"County should not be: {val}")

                if col_errors:
                    total_rows = len(series)  # total rows in this column

                    # Skip if all rows are flagged
                    # if flag every row as an error, then the rule is usually not very useful (too generic or specific)
                    # the column is also potentially in a wrong cluster

                    if len(col_errors) >= total_rows:
                        print(f"[DEBUG] Skipping rule '{rule.name}' for column '{column}' "
                              f"in table '{table}' (flagged all {total_rows} rows).")
                        col_errors.clear()   # Ignore this rule for this column
                    if col_errors:
                        errors.append({
                            "table": table,
                            "cluster": cid,
                            "column": column,
                            "error_indices": sorted(list(col_errors))
                        })
    return errors


def get_column_profile_by_name(name, profiles):
    for col in profiles:
        if col["column_name"] == name:
            return col
    return None


# method 1: use the new column dominant patter - assume the majority data are correct data
# for feature in rule.used_features:
#    print(feature)
#    print(col_profile[feature])

# method 2: static rules - use user set up parameter
def detect_dynamic_errors(clusters, shared_rules, rules, raw_dataset, column_profiles):
    errors = []
    rule_lookup = {r.name: r for r in rules}

    for cid, colnames in clusters.items():
        cluster_has_sample = False  # Track if this cluster has any sample column for its rules

        for rule_name in shared_rules.get(cid, []):
            rule = rule_lookup.get(rule_name)
            if not rule:
                continue

            # Check if this rule needs a sample column
            if rule.name == "matches_regex":
                for sample_col in getattr(rule, 'sample_column', []):
                    if sample_col in colnames:
                        cluster_has_sample = True
                        break  # Found at least one sample column

        # After checking all rules for this cluster
        if not cluster_has_sample and any(
            rule_lookup.get(rn) and rule_lookup[rn].name == "matches_regex"
            for rn in shared_rules.get(cid, [])
        ):
            print(f"Cluster '{cid}' has no sample column available "
                  f"for its rules. Please specify a sample column for this cluster.")
            continue  # Skip processing this cluster

        # Continue with normal error detection
        for rule_name in shared_rules.get(cid, []):
            rule = rule_lookup.get(rule_name)
            if not rule:
                continue

            for colname in colnames:
                col_profile = get_column_profile_by_name(colname, column_profiles)
                if not col_profile:
                    continue

                dataset = col_profile.get('dataset_name')
                column = col_profile.get('column_name')

                if dataset not in raw_dataset or column not in raw_dataset[dataset].columns:
                    continue

                series = raw_dataset[dataset][column]
                sample_profile = None

                if rule.name == "matches_regex":
                    for sample_col in getattr(rule, 'sample_column', []):
                        if sample_col in colnames:
                            sample_profile = get_column_profile_by_name(sample_col, column_profiles)
                            break

                rule.prepare(series, sample_profile)

                mask = series.apply(lambda val: rule.validate_cell(val))
                invalid_indices = series.index[~mask].tolist()

                if invalid_indices:
                    errors.append({
                        "table": dataset,
                        "column": column,
                        "error_indices": invalid_indices,
                        "invalid_values": series.loc[invalid_indices].tolist()
                    })

    return errors




