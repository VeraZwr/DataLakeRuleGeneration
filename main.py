# main.py

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from datetime import datetime

from rules.loader import load_all_rules
from rules.evaluation import get_shared_rules_per_cluster_with_sample_cloumn
from rules.dictionary_rule import SIMPLE_RULE_PROFILES

from utils.file_io import load_pickle
from utils.metrics import (
    compute_actual_errors,
    evaluate_one_dataset_only,
    evaluate_multiple_datasets,
)

from utils.clustering import (
    cluster_columns,                    # DBSCAN
    k_means_cluster_columns,            # plain K-Means
    get_numeric_feature_keys,           # helper
    cluster_columns_kmeans_by_samples,  # seeded K-Means wrapper
)


# ---------------------------
# Helpers
# ---------------------------

def _to_float_safe(v):
    # Convert v to a finite float; coerce None/NaN/invalid to 0.0
    if v is None:
        return 0.0
    if isinstance(v, (int, float)):
        if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
            return 0.0
        return float(v)
    try:
        f = float(v)
        if np.isnan(f) or np.isinf(f):
            return 0.0
        return f
    except Exception:
        return 0.0

# elbow
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from sklearn.cluster import DBSCAN

# SAFE feature builder (replace your existing _build_feature_matrix with this one)
def _build_feature_matrix(column_profiles, feature_keys):
    X = np.array([
        [_to_float_safe(cp.get(k, 0.0)) for k in feature_keys]
        for cp in column_profiles
    ], dtype=float)
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

def _preprocess_for_metric(X, metric: str):
    """
    Standardize, then (for cosine) L2-normalize so Euclidean behaves ~cosine.
    """
    Xs = StandardScaler().fit_transform(X)
    if metric == "cosine":
        Xs = normalize(Xs)
    return Xs

def _labels_from_clusters(clusters: dict[int, list[str]], column_profiles: list[dict]):
    """
    Build a label vector aligned to column_profiles order. Unassigned -> -1.
    """
    uid_to_idx = {cp["unique_id"]: i for i, cp in enumerate(column_profiles)}
    labels = np.full(len(column_profiles), -1, dtype=int)
    for cid, members in clusters.items():
        for uid in members:
            i = uid_to_idx.get(uid)
            if i is not None:
                labels[i] = int(cid)
    return labels

def run_plain_kmeans(column_profiles, n_clusters: int, metric: str = "euclidean"):
    feature_keys = get_numeric_feature_keys(column_profiles)
    if not feature_keys:
        raise ValueError("No numeric feature keys found.")
    X = _build_feature_matrix(column_profiles, feature_keys)
    Xp = _preprocess_for_metric(X, metric)
    km = KMeans(n_clusters=n_clusters, n_init="auto", random_state=42, max_iter=300)
    labels = km.fit_predict(Xp)
    # Build cluster dict
    clusters = {}
    for cp, lab in zip(column_profiles, labels):
        clusters.setdefault(int(lab), []).append(cp["unique_id"])
    return clusters, Xp

def run_dbscan(column_profiles, eps: float, min_samples: int, metric: str = "euclidean"):
    feature_keys = get_numeric_feature_keys(column_profiles)
    if not feature_keys:
        raise ValueError("No numeric feature keys found.")
    X = _build_feature_matrix(column_profiles, feature_keys)
    Xp = _preprocess_for_metric(X, metric)
    if metric == "cosine":
        db = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine")
    else:
        db = DBSCAN(eps=eps, min_samples=min_samples, metric="euclidean")
    labels = db.fit_predict(Xp)  # noise = -1
    clusters = {}
    for cp, lab in zip(column_profiles, labels):
        clusters.setdefault(int(lab), []).append(cp["unique_id"])
    if -1 in clusters:
        # keep noise in -1 cluster for consistency
        pass
    return clusters, Xp



def _pick_k_by_knee(ks, inertias):
    """
    Classic elbow heuristic:
    - Normalize points (k, inertia)
    - Compute distance of each point to the line between first and last
    - Choose k with maximum distance (the "knee")
    """
    x = np.array(ks, dtype=float)
    y = np.array(inertias, dtype=float)

    # Normalize to [0,1] (avoid div-by-zero)
    x_norm = (x - x.min()) / (x.max() - x.min() + 1e-12)
    y_norm = (y - y.min()) / (y.max() - y.min() + 1e-12)

    p1 = np.array([x_norm[0], y_norm[0]])
    p2 = np.array([x_norm[-1], y_norm[-1]])
    line_vec = p2 - p1
    line_len = np.linalg.norm(line_vec) + 1e-12
    line_unit = line_vec / line_len

    dists = []
    for xi, yi in zip(x_norm, y_norm):
        p  = np.array([xi, yi])
        v  = p - p1
        # Perpendicular distance from point to line
        proj_len = np.dot(v, line_unit)
        proj_pt  = p1 + proj_len * line_unit
        dist     = np.linalg.norm(p - proj_pt)
        dists.append(dist)

    idx = int(np.argmax(dists))
    return int(ks[idx])

def find_best_k_by_elbow(column_profiles, k_min=2, k_max=12, standardize=True,
                         results_dir: Path | None = None, plot=False, plot_filename="elbow.png"):
    """
    Returns (best_k, diag) where diag has ks/inertias and optional plot path.
    """
    from utils.clustering import get_numeric_feature_keys

    feature_keys = get_numeric_feature_keys(column_profiles)
    if not feature_keys:
        raise ValueError("No numeric feature keys found for elbow method.")

    X = _build_feature_matrix(column_profiles, feature_keys)
    if standardize:
        X = StandardScaler().fit_transform(X)

    ks = list(range(k_min, k_max + 1))
    inertias = []
    for k in ks:
        km = KMeans(n_clusters=k, n_init="auto", random_state=42, max_iter=300)
        km.fit(X)
        inertias.append(float(km.inertia_))

    best_k = _pick_k_by_knee(ks, inertias)

    plot_path = None
    if plot:
        # save plot if requested
        if results_dir is not None:
            results_dir.mkdir(parents=True, exist_ok=True)
            plot_path = results_dir / plot_filename
        # Make the plot
        plt.figure()
        plt.plot(ks, inertias, marker="o")
        plt.xlabel("k (number of clusters)")
        plt.ylabel("Inertia (within-cluster SSE)")
        plt.title("Elbow Method")
        if plot_path:
            plt.savefig(plot_path, bbox_inches="tight")
            plt.close()
        else:
            # show inline if no path (useful in notebooks; harmless in scripts)
            plt.show()

    diag = {"ks": ks, "inertias": inertias, "plot_path": str(plot_path) if plot_path else None}
    return best_k, diag
# silhouette

from sklearn.metrics import silhouette_score

def save_silhouette_plot(diag, out_path: Path):
    ks = diag["ks"]; sils = [v if v is not None else np.nan for v in diag["silhouettes"]]
    plt.figure()
    plt.plot(ks, sils, marker="o")
    plt.xlabel("k (number of clusters)")
    plt.ylabel("Silhouette score")
    plt.title("Silhouette vs k")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()

def _safe_k_range(n_samples: int, k_min: int, k_max: int):
    # silhouette requires 2 <= k <= n_samples-1
    upper = max(2, min(k_max, n_samples - 1))
    lower = max(2, min(k_min, upper))
    if upper < 2:
        raise ValueError("Not enough samples for clustering/silhouette.")
    return list(range(lower, upper + 1))

def compute_elbow_and_silhouette(column_profiles, k_min=2, k_max=100, metric="euclidean"):
    feature_keys = get_numeric_feature_keys(column_profiles)
    if not feature_keys:
        raise ValueError("No numeric feature keys found.")
    X = _build_feature_matrix(column_profiles, feature_keys)
    Xp = _preprocess_for_metric(X, metric)

    ks = _safe_k_range(len(Xp), k_min, k_max)
    inertias, silhouettes = [], []

    for k in ks:
        km = KMeans(n_clusters=k, n_init="auto", random_state=42, max_iter=300)
        labels = km.fit_predict(Xp)
        inertias.append(float(km.inertia_))
        try:
            if len(set(labels)) >= 2:
                s = float(silhouette_score(Xp, labels, metric=("cosine" if metric == "cosine" else "euclidean")))
            else:
                s = np.nan
        except Exception:
            s = np.nan
        silhouettes.append(s)

    return {"ks": ks, "inertias": inertias, "silhouettes": silhouettes}

def find_best_k_by_elbow(column_profiles, k_min=2, k_max=12, metric="euclidean",
                         results_dir: Path | None = None, plot=False, plot_filename="elbow.png"):
    feature_keys = get_numeric_feature_keys(column_profiles)
    if not feature_keys:
        raise ValueError("No numeric feature keys found for elbow method.")
    X = _build_feature_matrix(column_profiles, feature_keys)
    Xp = _preprocess_for_metric(X, metric)

    ks = list(range(k_min, k_max + 1))
    inertias = []
    for k in ks:
        km = KMeans(n_clusters=k, n_init="auto", random_state=42, max_iter=300)
        km.fit(Xp)
        inertias.append(float(km.inertia_))

    best_k = _pick_k_by_knee(ks, inertias)

    plot_path = None
    if plot:
        if results_dir is not None:
            results_dir.mkdir(parents=True, exist_ok=True)
            plot_path = results_dir / plot_filename
        plt.figure()
        plt.plot(ks, inertias, marker="o")
        plt.xlabel("k (number of clusters)")
        plt.ylabel("Inertia (within-cluster SSE)")
        plt.title(f"Elbow Method ({metric})")
        if plot_path:
            plt.savefig(plot_path, bbox_inches="tight")
            plt.close()
        else:
            plt.show()

    diag = {"ks": ks, "inertias": inertias, "plot_path": str(plot_path) if plot_path else None}
    return best_k, diag

def _save_cluster_scatter(column_profiles, Xp, labels, out_path: Path, title: str):
    """
    PCA to 2D on the (already preprocessed) feature matrix Xp, then scatter by cluster label.
    Noise/unassigned (-1) plotted in grey.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pca = PCA(n_components=2, random_state=42)
    Z = pca.fit_transform(Xp)

    # Build a stable palette
    unique_labs = sorted(set(labels))
    # Put -1 last to keep colors for real clusters stable
    if -1 in unique_labs:
        unique_labs = [l for l in unique_labs if l != -1] + [-1]

    plt.figure()
    for lab in unique_labs:
        mask = (labels == lab)
        if lab == -1:
            plt.scatter(Z[mask, 0], Z[mask, 1], s=28, alpha=0.7, label="noise (-1)", color="gray")
        else:
            plt.scatter(Z[mask, 0], Z[mask, 1], s=28, alpha=0.8, label=f"cluster {lab}")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.title(title)
    # keep legends readable
    if len(unique_labs) <= 20:
        plt.legend(loc="best", fontsize=8, frameon=True)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def save_elbow_silhouette_combo(diag, out_path: Path, chosen_k: int | None = None):
    """
    Draws inertia (left y-axis) and silhouette (right y-axis) vs k on one chart.
    - diag: dict with keys 'ks', 'inertias', 'silhouettes' (None where unavailable)
    - out_path: where to save the PNG
    - chosen_k: optional vertical line to highlight the selected k
    """
    ks = np.array(diag["ks"], dtype=int)
    inertias = np.array(diag["inertias"], dtype=float)
    # Replace None silhouettes with NaN so matplotlib skips them
    silhouettes = np.array([np.nan if s is None else float(s) for s in diag["silhouettes"]], dtype=float)
    mask = np.isfinite(silhouettes)

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    # Elbow (inertia) on left Y
    ax1.plot(ks, inertias, marker="o", label="Inertia (WCSS)")
    ax1.set_xlabel("k (number of clusters)")
    ax1.set_ylabel("Inertia (within-cluster SSE)")

    # Silhouette on right Y
    # ax2.plot(ks, silhouettes, marker="s", linestyle="--", label="Silhouette score")
    # ax2.set_ylabel("Silhouette score")
    if mask.any():
        ax2.plot(ks[mask], silhouettes[mask], marker="s", linestyle="--", color = "green", label="Silhouette score")
        ax2.scatter(ks[mask], silhouettes[mask], marker="s")
    else:
        print("[PLOT] No finite silhouette scores to plot.")
    ax2.set_ylabel("Silhouette score", color="green")
    ax2.tick_params(axis="y", colors="green")
    ax2.spines["right"].set_color("green")
    ax2.set_ylim(-1, 1)  # typical range; adjust if you prefer (0,1)

    # Optional vertical marker at chosen k
    if chosen_k is not None:
        ax1.axvline(chosen_k, linestyle=":", linewidth=1.5)
        ax1.text(chosen_k, ax1.get_ylim()[1], f"  k={chosen_k}", va="top")

    # Build a joint legend
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="best")

    ax1.set_title("Elbow (Inertia) & Silhouette vs k")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

def pick_k_auto(diag, strategy="both"):
    ks = np.array(diag["ks"])
    inertias = np.array(diag["inertias"], dtype=float)
    sils = np.array([(-1 if v is None else v) for v in diag["silhouettes"]], dtype=float)

    elbow_k = _pick_k_by_knee(ks, inertias)

    sil_ok = sils > -0.5
    if sil_ok.any():
        sil_k = int(ks[sil_ok][np.argmax(sils[sil_ok])])
    else:
        sil_k = elbow_k

    if strategy == "elbow":
        return elbow_k
    if strategy == "silhouette":
        return sil_k

    window = 2
    mask = (ks >= (elbow_k - window)) & (ks <= (elbow_k + window))
    if mask.any():
        local_idx = np.argmax(sils[mask])
        return int(ks[mask][local_idx])
    return elbow_k

def collect_seeds_from_dictionary_rules(column_profiles, centroid_key="column_name"):
    """
    Pull seed names from SIMPLE_RULE_PROFILES and keep only those present
    in the loaded profiles (matched by centroid_key).
    """
    candidate_fields = (
        "sample_columns", "samples", "examples", "column_examples",
        "seed_columns", "seed", "sample_column", "columns", "column_name"
    )
    candidates = []
    for _, spec in dict(SIMPLE_RULE_PROFILES).items():
        if isinstance(spec, dict):
            for field in candidate_fields:
                if field in spec:
                    val = spec[field]
                    if isinstance(val, str):
                        candidates.append(val.strip())
                    elif isinstance(val, (list, tuple)):
                        candidates.extend([v.strip() for v in val if isinstance(v, str) and v.strip()])

    available = {c.get(centroid_key) for c in column_profiles if c.get(centroid_key)}
    seeds, seen = [], set()
    for s in candidates:
        if s in available and s not in seen:
            seeds.append(s)
            seen.add(s)
    return seeds


def print_cluster_overview(method_label, clusters, column_profiles):
    """
    Print summary + cluster members; members are assumed to be 'unique_id's.
    """
    clustered = set(c for members in clusters.values() for c in members)
    all_cols = set(cp["unique_id"] for cp in column_profiles)
    missing = sorted(all_cols - clustered)

    print(f"\n=== {method_label} ===")
    print(f"Clusters: {len(clusters)}")
    print(f"Clustered cols: {len(clustered)} / {len(all_cols)}")
    if missing:
        preview = ", ".join(missing[:10])
        print(f"Missing (noise/unassigned): {len(missing)} -> {preview}{' ...' if len(missing) > 10 else ''}")
    for cid, members in clusters.items():
        print(f"  Cluster {cid} ({len(members)}): {sorted(members)}")

import shutil

def _parse_rule_entry(entry):
    """
    Make 'get_shared_rules_per_cluster_with_sample_cloumn' outputs robust:
    Accept str / tuple / dict shapes and extract rule name + sample column if present.
    """
    rule_name, sample_col = None, None
    if isinstance(entry, dict):
        rule_name = entry.get("rule") or entry.get("rule_name") or entry.get("name") or entry.get("id")
        sample_col = (
            entry.get("sample_column") or entry.get("sample") or
            entry.get("column") or entry.get("example") or entry.get("seed")
        )
    elif isinstance(entry, (list, tuple)):
        if len(entry) >= 1: rule_name = entry[0]
        if len(entry) >= 2: sample_col = entry[1]
    else:
        rule_name = str(entry)
    return (rule_name or "UNKNOWN_RULE", sample_col)

def save_clustering_outputs(base_dir: Path,
                            method_label: str,
                            clusters: dict[int, list[str]],
                            column_profiles: list[dict],
                            shared_rules: dict[int, list],
                            include_rule_descriptions: bool = True):
    """
    Writes three CSVs into results/.../<method_label>/ :
      - cluster_assignments.csv
      - cluster_rules.csv
      - cluster_summary.csv
    """
    out_dir = base_dir / method_label
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Map column id -> (dataset_name, column_name)
    meta = {
        cp["unique_id"]: {
            "dataset_name": cp.get("dataset_name"),
            "column_name":  cp.get("column_name"),
        }
        for cp in column_profiles
    }

    # --- assignments (one row per column) ---
    assigned = set()
    rows_assign = []
    for cid, members in clusters.items():
        for uid in members:
            m = meta.get(uid, {})
            rows_assign.append({
                "method": method_label,
                "cluster_id": cid,
                "unique_id": uid,
                "dataset_name": m.get("dataset_name"),
                "column_name": m.get("column_name"),
            })
            assigned.add(uid)

    # Add unassigned as cluster_id = -1 (useful for DBSCAN noise)
    all_ids = set(meta.keys())
    for uid in sorted(all_ids - assigned):
        m = meta.get(uid, {})
        rows_assign.append({
            "method": method_label,
            "cluster_id": -1,
            "unique_id": uid,
            "dataset_name": m.get("dataset_name"),
            "column_name": m.get("column_name"),
        })

    df_assign = pd.DataFrame(rows_assign).sort_values(["cluster_id", "dataset_name", "column_name"])
    df_assign.to_csv(out_dir / f"cluster_assignments_{ts}.csv", index=False)

    # --- rules (one row per (cluster, rule)) ---
    rows_rules = []
    for cid, rulelist in (shared_rules or {}).items():
        for entry in (rulelist or []):
            rule_name, sample_col = _parse_rule_entry(entry)
            desc = desc_resolver(rule_name) if include_rule_descriptions else None
            rows_rules.append({
                "method": method_label,
                "cluster_id": cid,
                "rule": rule_name,
                "sample_column": sample_col,
                "description": desc,
            })

    df_rules = pd.DataFrame(rows_rules)
    if not df_rules.empty:
        df_rules = df_rules.sort_values(["cluster_id", "rule"])
    df_rules.to_csv(out_dir / f"cluster_rules_{ts}.csv", index=False)

    # --- summary (cluster size + member list + rule count) ---
    summary_rows = []
    for cid, members in clusters.items():
        member_list = ",".join(sorted(members))
        rule_count = len([r for r in rows_rules if r["cluster_id"] == cid])
        summary_rows.append({
            "method": method_label,
            "cluster_id": cid,
            "size": len(members),
            "members": member_list,
            "rule_count": rule_count,
        })
    # include an entry for unassigned if any
    unassigned = sorted(all_ids - assigned)
    if unassigned:
        summary_rows.append({
            "method": method_label,
            "cluster_id": -1,
            "size": len(unassigned),
            "members": ",".join(unassigned),
            "rule_count": 0,
        })

    pd.DataFrame(summary_rows).sort_values("cluster_id").to_csv(out_dir / f"cluster_summary_{ts}.csv", index=False)

    print(f"[SAVE] Wrote { (out_dir / f'cluster_assignments_{ts}.csv').resolve() }")
    print(f"[SAVE] Wrote { (out_dir / f'cluster_rules_{ts}.csv').resolve() }")
    print(f"[SAVE] Wrote { (out_dir / f'cluster_summary_{ts}.csv').resolve() }")


def verify_single_dataset_paths(dataset_group, dataset_name) -> tuple[Path, Path]:
    """
    Ensure dirty.csv and clean.csv exist. Supports legacy fallback without group.
    Returns (dirty_path, clean_path) if found, raises otherwise.
    """
    grouped = Path("datasets") / dataset_group / dataset_name
    dirty_p, clean_p = grouped / "dirty.csv", grouped / "clean.csv"
    if dirty_p.exists() and clean_p.exists():
        return dirty_p, clean_p

    legacy = Path("datasets") / dataset_name
    l_dirty, l_clean = legacy / "dirty.csv", legacy / "clean.csv"
    if l_dirty.exists() and l_clean.exists():
        print(f"[WARN] Using legacy dataset path (no group): {legacy}")
        return l_dirty, l_clean

    raise FileNotFoundError(
        f"Missing CSVs for '{dataset_name}'. Tried:\n"
        f" - {dirty_p}\n - {clean_p}\n - {l_dirty}\n - {l_clean}"
    )

def desc_resolver(rule_name: str) -> str:
    spec = SIMPLE_RULE_PROFILES.get(rule_name, {})
    return spec.get("description") or "ERROR"

# ---------------------------
# Main execution
# ---------------------------
def main(
    mode="single",
    dataset_name=None,
    dataset_group=None,
    eps_value=0.5,
    min_samples=1,
    kmeans_k=5,
    similarity="cosine",
    threshold=0.8,
    kmeans_max_iter=100,
    seeds_csv="",
    seeds_file="",
):
    results_path = Path("results")
    datasets_path = Path("datasets")

    dataset_profiles: list[str] = []
    dataset_names: list[str] = []

    # Where to store elbow.png if requested
    if mode == "single" and dataset_group and dataset_name:
        elbow_dir = results_path / dataset_group / dataset_name
    elif mode == "multi" and dataset_group:
        elbow_dir = results_path / dataset_group / "_multi"
    else:
        elbow_dir = results_path / "_misc"

    # -------------------
    # Discover profiles
    # -------------------
    if mode == "single":
        if not dataset_name or not dataset_group:
            print("No dataset name or dataset group provided")
            return
        single_profile = results_path / dataset_group / dataset_name / "column_profile.dictionary"
        if single_profile.exists():
            dataset_profiles.append(str(single_profile))
            dataset_names.append(dataset_name)
        else:
            print(f"Profile not found for single dataset: {dataset_name}")
            return

    elif mode == "multi":
        if not dataset_group:
            print("No dataset group provided")
            return

        group_results_path = results_path / dataset_group
        group_data_path = datasets_path / dataset_group

        for folder in group_results_path.iterdir():
            profile_file = folder / "column_profile.dictionary"
            if folder.is_dir() and profile_file.exists():
                dataset_profiles.append(str(profile_file))
                dataset_names.append(folder.name)

        if not dataset_profiles:
            print(f"No dataset profiles found in results/{dataset_group}")
            return

        # Optional: which tables have ground truth
        multi_table_names = [
            folder.name for folder in group_data_path.iterdir()
            if folder.is_dir()
            and (folder / "dirty.csv").exists()
            and (folder / "clean.csv").exists()
        ]
    else:
        print(f"Unknown mode: {mode}")
        return

    # -------------------------------
    # Load column profiles and tag IDs
    # -------------------------------
    column_profiles = []
    for path, dname in zip(dataset_profiles, dataset_names):
        dataset_column_profiles = load_pickle(path)
        for col in dataset_column_profiles:
            col["dataset_name"] = dname
            clean_col_name = col["column_name"]
            col["column_name"] = clean_col_name
            col["unique_id"] = f"{clean_col_name}"
        column_profiles.extend(dataset_column_profiles)

    # Preflight path check for single mode (clearer errors)
    if mode == "single":
        try:
            _ = verify_single_dataset_paths(dataset_group, dataset_name)
        except FileNotFoundError as e:
            print("[ERROR]", e)
            return

    # -------------------------------
    # Load rules
    # -------------------------------
    rules = load_all_rules()

    # -------------------------------
    # Build clusters for all methods
    # -------------------------------
    comparison: dict[str, dict[int, list[str]]] = {}
    config_base = f"Eps:{eps_value} | MinSamples:{min_samples} | K:{kmeans_k}"
    elbow_dir.mkdir(parents=True, exist_ok=True)
    print(f"[AUTO-K] Combined plot: {(elbow_dir / 'combo.png').resolve()}")

    # 1) DBSCAN
    start = time.perf_counter()
    clusters_dbscan, Xp_db = run_dbscan(
        column_profiles, eps=eps_value, min_samples=min_samples, metric=args.metric
    )
    elapsed = time.perf_counter() - start
    print(f"[TIME] DBSCAN clustering took {elapsed:.3f} seconds (metric={args.metric})")
    comparison["DBSCAN"] = clusters_dbscan

    # 2) Plain K-Means (optionally auto-select k via elbow/silhouette)
    start = time.perf_counter()
    if args.auto_k:
        try:
            diag = compute_elbow_and_silhouette(
                column_profiles, k_min=args.k_min, k_max=args.k_max, metric=args.metric
            )
            print(f"[AUTO-K] Tested ks={diag['ks']}")
            print(f"[AUTO-K] Inertias={np.round(diag['inertias'], 2).tolist()}")
            print(
                f"[AUTO-K] Silhouettes={[None if (v is None or np.isnan(v)) else round(v, 3) for v in diag['silhouettes']]}")

            kmeans_k = pick_k_auto(diag, strategy=args.auto_k_strategy)
            print(f"[AUTO-K] Selected k={kmeans_k} (strategy={args.auto_k_strategy}, metric={args.metric})")

            if args.elbow_plot:
                _, _ = find_best_k_by_elbow(
                    column_profiles,
                    k_min=args.k_min, k_max=args.k_max,
                    metric=args.metric,
                    results_dir=elbow_dir, plot=True, plot_filename="elbow.png"
                )
                print(f"[AUTO-K] Elbow plot: {(elbow_dir / 'elbow.png').resolve()}")
            if args.silhouette_plot:
                sil_path = elbow_dir / "silhouette.png"
                save_silhouette_plot(diag, sil_path)
                print(f"[AUTO-K] Silhouette plot: {sil_path.resolve()}")
            if args.combo_plot:
                combo_path = elbow_dir / "combo.png"
                save_elbow_silhouette_combo(diag, combo_path, chosen_k=kmeans_k)
                print(f"[AUTO-K] Combined plot: {combo_path.resolve()}")

        except Exception as e:
            print(f"[AUTO-K] Failed to auto-select k ({e}); falling back to k={kmeans_k}")

    clusters_kmeans, Xp_km = run_plain_kmeans(
        column_profiles, n_clusters=kmeans_k, metric=args.metric
    )
    elapsed = time.perf_counter() - start
    print(f"[TIME] K-Means (k={kmeans_k}) clustering took {elapsed:.3f} seconds (metric={args.metric})")
    comparison[f"KMEANS_k={kmeans_k}"] = clusters_kmeans

    # 3) Seeded K-Means
    seeds: list[str] = []
    if seeds_csv:
        seeds.extend([s.strip() for s in seeds_csv.split(",") if s.strip()])
    if seeds_file:
        with open(seeds_file, "r") as sf:
            seeds.extend([ln.strip() for ln in sf if ln.strip()])
    if not seeds:
        seeds = collect_seeds_from_dictionary_rules(column_profiles, centroid_key="column_name")
        if seeds:
            print(f"[KMEANS_SEEDED] Using seeds from dictionary rules: {seeds}")
        else:
            print("[KMEANS_SEEDED] No seeds provided and none inferred; skipping seeded K-Means.")

    Xp_seeded = None
    if seeds:
        feature_keys = get_numeric_feature_keys(column_profiles)
        start = time.perf_counter()
        clusters_seeded = cluster_columns_kmeans_by_samples(
            column_profiles,
            seeds,
            feature_keys=feature_keys,
            id_key="unique_id",
            centroid_key="column_name",
            similarity=similarity,  # "cosine" or "euclidean" from args
            membership_threshold=None if (threshold is None or threshold <= 0) else float(threshold),
            max_iter=kmeans_max_iter,
        )
        elapsed = time.perf_counter() - start
        print(f"[TIME] Seeded K-Means clustering took {elapsed:.3f} seconds (similarity={similarity})")
        comparison["KMEANS_SEEDED"] = clusters_seeded

        # Build X preprocessed to match seeded similarity for plotting
        X = _build_feature_matrix(column_profiles, feature_keys)
        Xp_seeded = _preprocess_for_metric(X, "cosine" if similarity == "cosine" else "euclidean")

    # -------------------------------
    # Compare + Evaluate each method
    # -------------------------------
    for method_label, clusters in comparison.items():
        print_cluster_overview(method_label, clusters, column_profiles)

        print("\nIdentifying shared rules per cluster...")
        shared_rules = get_shared_rules_per_cluster_with_sample_cloumn(
            rules, column_profiles, clusters
        )
        for cid, rulelist in shared_rules.items():
            print(f"[{method_label}] Cluster {cid} shares rules: {rulelist}")

        if mode == "single":
            base_dir = Path("results") / dataset_group / dataset_name
        elif mode == "multi":
            base_dir = Path("results") / dataset_group / "_multi"
        else:
            base_dir = Path("results") / "_misc"

        # ---- Plot clusters (if requested) ----
        if args.plot_clusters:
            # choose the right preprocessed X for the method
            if method_label == "DBSCAN":
                Xp = Xp_db
            elif method_label.startswith("KMEANS_k="):
                Xp = Xp_km
            elif method_label == "KMEANS_SEEDED":
                Xp = Xp_seeded
            else:
                Xp = None

            if Xp is not None:
                labels_vec = _labels_from_clusters(clusters, column_profiles)
                plot_path = base_dir / method_label / "clusters_2d.png"
                _save_cluster_scatter(column_profiles, Xp, labels_vec, plot_path,
                                      title=f"{method_label} clusters (metric={args.metric if method_label != 'KMEANS_SEEDED' else similarity})")
                print(f"[PLOT] Saved cluster scatter: {plot_path.resolve()}")

        save_clustering_outputs(base_dir, method_label, clusters, column_profiles, shared_rules)

        config = f"{config_base} | Method:{method_label}"

        # Single dataset evaluation
        if mode == "single":
            # evaluate_one_dataset_only writes TXT + (with patches) CSV exports
            evaluate_one_dataset_only(
                rules, shared_rules, clusters, column_profiles,
                dataset_group, dataset_name, config,
                method_label=method_label,
            )

        # Multi-dataset evaluation (only if we detected any GT tables earlier)
        if mode == "multi":
            evaluate_multiple_datasets(
                rules, shared_rules, clusters, column_profiles,
                dataset_group, config,
                method_label=method_label,
            )

    # -------------------------------
    # (Optional) Print overall GT counts (single run convenience)
    # -------------------------------
    if mode == "single":
        raw_dataset, clean_dataset_dict = {}, {}
        # prefer grouped path; fall back handled in verify
        try:
            dirty_p, clean_p = verify_single_dataset_paths(dataset_group, dataset_name)
            raw_dataset[dataset_name] = pd.read_csv(dirty_p)
            clean_dataset_dict[dataset_name] = pd.read_csv(clean_p)
            actual_errors_by_column = compute_actual_errors(clean_dataset_dict, raw_dataset)
            print("\nGround Truth Error Counts Per Column:")
            for (table, column), row_indices in actual_errors_by_column.items():
                if table == dataset_name:
                    print(f"Table: {table} | Column: {column} | Error Count: {len(row_indices)}")
            print(f"Total GT error cells: {sum(len(v) for v in actual_errors_by_column.values())}")
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run clustering (DBSCAN, K-Means, Seeded K-Means) and evaluate."
    )
    parser.add_argument("--mode", choices=["single", "multi"], default="single")
    parser.add_argument("--dataset_name", type=str, help="Dataset name (single mode)")
    parser.add_argument("--dataset_group", type=str, help="Dataset group folder")

    # DBSCAN
    parser.add_argument("--eps", type=float, default=0.5, help="DBSCAN epsilon")
    parser.add_argument("--min_samples", type=int, default=1, help="DBSCAN min samples")

    # Plain K-Means
    parser.add_argument("--kmeans_k", type=int, default=5, help="K-Means number of clusters")

    # Seeded K-Means
    parser.add_argument("--similarity", choices=["cosine", "euclidean"], default="cosine")
    parser.add_argument("--threshold", type=float, default=0.8,
                        help="Seeded membership: cosine>=thr / euclidean<=thr; <=0 disables")
    parser.add_argument("--kmeans_max_iter", type=int, default=100)
    parser.add_argument("--seeds", dest="seeds_csv", type=str, default="",
                        help="Comma-separated seed column names")
    parser.add_argument("--seeds_file", type=str, default="", help="Path to file with one seed per line")

    # Elbow / auto-k
    # parser.add_argument("--auto_k", action="store_true",
    #                     help="Use elbow method to select K for plain K-Means")
    # parser.add_argument("--k_min", type=int, default=2, help="Min k to try for elbow")
    # parser.add_argument("--k_max", type=int, default=12, help="Max k to try for elbow")
    # parser.add_argument("--elbow_plot", action="store_true", help="Save elbow plot")

    parser.add_argument("--auto_k", action="store_true",
                        help="Auto-select K for plain K-Means using elbow/silhouette")
    parser.add_argument("--k_min", type=int, default=2)
    parser.add_argument("--k_max", type=int, default=100)
    parser.add_argument("--auto_k_strategy", choices=["elbow", "silhouette", "both"], default="both",
                        help="How to pick k when auto-selecting")
    parser.add_argument("--elbow_plot", action="store_true", help="Save elbow.png")
    parser.add_argument("--silhouette_plot", action="store_true", help="Save silhouette.png")
    parser.add_argument("--combo_plot", action="store_true",
                        help="Save a combined inertia+silhouette plot (combo.png)")
    parser.add_argument("--metric", choices=["euclidean", "cosine"], default="euclidean",
                        help="Distance/similarity used across methods (K-Means uses L2-normalized features to approximate cosine)")
    parser.add_argument("--plot_clusters", action="store_true",
                        help="Save a 2D scatter plot of clusters for each method (clusters_2d.png)")

    args = parser.parse_args()

    main(
        mode=args.mode,
        dataset_name=args.dataset_name,
        dataset_group=args.dataset_group,
        eps_value=args.eps,
        min_samples=args.min_samples,
        kmeans_k=args.kmeans_k,
        similarity=args.similarity,
        threshold=args.threshold,
        kmeans_max_iter=args.kmeans_max_iter,
        seeds_csv=args.seeds_csv,
        seeds_file=args.seeds_file,
    )

    # Examples:
    # python3 main.py --mode single --dataset_group Quintet --dataset_name hospital
    # python3 main.py --mode multi  --dataset_group Quintet --min_samples 2 --kmeans_k 6
    # python3 main.py --mode single --dataset_group Quintet --dataset_name hospital --seeds "email,zip_code" --threshold 0.85

    # # Single dataset, auto-pick k from 2..12 and save elbow.png
    # python3 main.py --mode single --dataset_group Quintet --dataset_name hospital \
    #   --auto_k --k_min 2 --k_max 12 --elbow_plot
    #
    # # Multi-dataset, try wider range
    # python3 main.py --mode multi --dataset_group Quintet \
    #   --auto_k --k_min 2 --k_max 15 --elbow_plot

# # Elbow + silhouette, pick using the 'both' strategy, save both plots
# python3 main.py --mode single --dataset_group Quintet --dataset_name hospital \
#   --auto_k --k_min 2 --k_max 15 --auto_k_strategy both \
#   --elbow_plot --silhouette_plot
#
# # Silhouette-only selection
# python3 main.py --mode multi --dataset_group Quintet \
#   --auto_k --auto_k_strategy silhouette --k_min 3 --k_max 20 --silhouette_plot

# python3 main.py --mode single --dataset_group Quintet --dataset_name hospital \
#   --auto_k --k_min 2 --k_max 15 --auto_k_strategy both --combo_plot

# python3 main.py --mode multi --dataset_group Quintet \
#   --auto_k --k_min 2 --k_max 60 --auto_k_strategy both --combo_plot

# Use elbow only
# python3 main.py --mode single --dataset_group Quintet --dataset_name hospital \
#   --auto_k --auto_k_strategy elbow
#
# # Default (both: elbow primary + silhouette refine), plus combo plot
# python3 main.py --mode single --dataset_group Quintet --dataset_name hospital \
#   --auto_k --k_min 2 --k_max 15 --auto_k_strategy both --combo_plot
#

