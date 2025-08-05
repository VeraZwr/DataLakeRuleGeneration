import os
import json
import numpy as np
import pickle
import pandas as pd
from matplotlib.style.core import update_nested_dict
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction import DictVectorizer
from rapidfuzz.distance import Levenshtein


def load_profile(profile_path):
    try:
        with open(profile_path, 'r', encoding='utf-8') as f:
            content = f.read()
            return eval(content)
    except (UnicodeDecodeError, SyntaxError):
        # 若失败，尝试用 pickle 加载
        with open(profile_path, 'rb') as f:
            return pickle.load(f)

def flatten_profile(profile):
    flat = {}
    for k, v in profile.items():
        if isinstance(v, dict):
            for subk, subv in v.items():
                flat[f"{k}_{subk}"] = subv
        else:
            flat[k] = v
    return flat
"""
#update, chose different profile for different rule_type
def filter_profile_by_rule_type(profile, rule_type):
    feature_map = {
        'fd': ['dataset_top_keywords', 'words_unique_mean', 'cells_unique_mean'],
        'pattern': ['characters_alphabet_mean', 'characters_numeric_mean'],
        'typo': ['words_alphabet_variance', 'words_length_variance'],
        'kb': ['dataset_top_keywords', 'dataset_rules_count']
    }

    if rule_type not in feature_map:
        return profile  # fallback to full profile

    selected_keys = feature_map[rule_type]
    return {k: v for k, v in profile.items() if any(sel in k for sel in selected_keys)}
"""
def compute_similarity(profile_a, profile_b):
    vec = DictVectorizer(sparse=False)
    keys_union = list(set(profile_a.keys()).union(set(profile_b.keys())))
    vec.fit([{k: 0 for k in keys_union}])
    v1 = vec.transform([profile_a])[0].reshape(1, -1)
    v2 = vec.transform([profile_b])[0].reshape(1, -1)
    return cosine_similarity(v1, v2)[0][0]

def compute_fd_conflict(df, lhs, rhs):
    grouped = df.groupby(lhs)[rhs].nunique()
    conflict_count = (grouped > 1).sum()
    total_groups = len(grouped)
    support_ratio = total_groups / len(df)
    conflict_ratio = conflict_count / total_groups if total_groups > 0 else 1.0
    return support_ratio, conflict_ratio


def rule_transfer_score(sim, support, conflict, alpha=0.5, beta=0.3, gamma=0.2):
    return alpha * sim + beta * support - gamma * conflict

def is_typo_conflict(values, threshold=2):
    values = list(values)
    for i in range(len(values)):
        for j in range(i+1, len(values)):
            d = Levenshtein.distance(values[i], values[j])
            if d <= threshold:
                return True
    return False

def print_violations(df, lhs, rhs):
    print(f"\nRows violating rule {lhs} -> {rhs}:")
    grouped = df.groupby(lhs)[rhs]
    for key, group in grouped:
        if group.nunique() > 1:
            #print(f"  {lhs} = {key}: {group.unique()}")
            violating_rows = df[df[lhs] == key].drop_duplicates(subset=rhs)
            print(violating_rows[[lhs, rhs]])


def main():
    profile_path_beers = 'results/Quintet/beers/dataset_profile.dictionary'
    profile_path_hospital = 'results/Quintet/hospital/dataset_profile.dictionary'

    profile_beers = flatten_profile(load_profile(profile_path_beers))
    profile_hospital = flatten_profile(load_profile(profile_path_hospital))

    similarity = compute_similarity(profile_beers, profile_hospital)
    print("Dirtiness Profile Similarity (cosine):", round(similarity, 4))

    df_hospital = pd.read_csv('datasets/Quintet/hospital/dirty.csv')  # ensure this file exists
    df_hospital.columns = df_hospital.columns.str.strip().str.lower()
    df_hospital['state'] = df_hospital['state'].str.strip().str.lower()
    df_hospital['city'] = df_hospital['city'].str.strip().str.lower()

    if 'state' in df_hospital.columns and 'city' in df_hospital.columns:
        support, conflict = compute_fd_conflict(df_hospital, 'city', 'state')
        score = rule_transfer_score(similarity, support, conflict)
        print(f"\nRule: city -> state")
        print(f"Support: {support:.2f}, Conflict: {conflict:.2f}")
        print(f"Transferability Score: {score:.2f}")
    else:
        print("Dataset missing required columns 'state' and 'city' for rule evaluation.")

    # Add after rule check
    print_violations(df_hospital, 'state', 'city')

if __name__ == '__main__':
    main()
