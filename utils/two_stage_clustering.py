# utils/two_stage_clustering.py
# pip install sentence-transformers scikit-learn
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MinMaxScaler

try:
    from sentence_transformers import SentenceTransformer
except ImportError as e:
    raise ImportError(
        "Missing dependency: sentence-transformers. Install with:\n"
        "  pip install sentence-transformers"
    ) from e


@dataclass
class TableClusteringResult:
    table_labels: Dict[str, int]               # table_name -> table_cluster_id
    table_clusters: Dict[int, List[str]]       # table_cluster_id -> [table_name, ...]
    column_clusters: Dict[int, List[str]]      # final column clusters (global IDs) -> [unique_id, ...]


# -------------------------------
# Stage 0: utilities
# -------------------------------

def _ensure_unique_id(col: dict) -> str:
    """
    Make sure the 'unique_id' is actually unique across datasets.
    If the caller didn't set one, fall back to 'dataset::column'.
    """
    if col.get("unique_id"):
        return col["unique_id"]
    ds = str(col.get("dataset_name", "") or "")
    cn = str(col.get("column_name", "") or "")
    return f"{ds}::{cn}" if ds else cn


def _numeric_feature_keys(columns: List[dict]) -> List[str]:
    return sorted({
        k for col in columns
        for k, v in col.items()
        if isinstance(v, (int, float))
    })


def _build_profile_matrix(columns: List[dict], feature_keys: List[str]) -> Tuple[np.ndarray, List[str]]:
    rows, ids = [], []
    for col in columns:
        vec = []
        for f in feature_keys:
            val = col.get(f, 0.0)
            if val is None or (isinstance(val, float) and math.isnan(val)):
                val = 0.0
            vec.append(float(val))
        rows.append(vec)
        ids.append(_ensure_unique_id(col))
    X = np.asarray(rows, dtype=float) if rows else np.zeros((0, len(feature_keys)), dtype=float)
    return X, ids


# -------------------------------
# Stage 1: cluster TABLE names with BERT
# -------------------------------

def _embed_table_names(
    table_names: List[str],
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 64
) -> np.ndarray:
    model = SentenceTransformer(model_name)
    # Normalize+encode; sentence-transformers returns float32 numpy
    Z = model.encode(table_names, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(Z, dtype=np.float32)


def cluster_tables_by_name(
    table_names: List[str],
    *,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    eps_tables: float = 0.20,                 # tighten/loosen table similarity
    min_samples_tables: int = 2
) -> Dict[int, List[str]]:
    """
    Returns {table_cluster_id: [table_name, ...]} using DBSCAN with cosine distance on BERT embeddings.
    """
    if not table_names:
        return {}

    Z = _embed_table_names(table_names, model_name=model_name)
    # DBSCAN with cosine distance (1 - cosine_similarity) because embeddings are L2-normalized
    db = DBSCAN(eps=eps_tables, min_samples=min_samples_tables, metric="cosine")
    labels = db.fit_predict(Z)

    clusters: Dict[int, List[str]] = defaultdict(list)
    noise_bucket: List[str] = []
    for name, lab in zip(table_names, labels):
        if lab == -1:
            noise_bucket.append(name)
        else:
            clusters[lab].append(name)

    # Make noise tables singletons so nothing is dropped
    next_id = (max(clusters.keys()) + 1) if clusters else 0
    for name in noise_bucket:
        clusters[next_id] = [name]
        next_id += 1

    # Reindex cluster ids to 0..K-1 for cleanliness
    reindexed: Dict[int, List[str]] = {}
    for new_id, old_id in enumerate(sorted(clusters.keys())):
        reindexed[new_id] = clusters[old_id]
    return reindexed


# -------------------------------
# Stage 2: cluster COLUMNS within each table cluster
# -------------------------------

def two_stage_clustering(
    columns: List[dict],
    *,
    # TABLE name clustering
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    eps_tables: float = 0.20,
    min_samples_tables: int = 2,
    # COLUMN profile clustering
    eps_columns: float = 0.50,
    min_samples_columns: int = 5,
    scale_profiles: bool = True,
    # Behavior
    group_mode: str = "by_table_cluster",    # "by_table_cluster" or "per_table"
) -> TableClusteringResult:
    """
    Two-stage:
      1) Cluster tables by BERT name semantics.
      2) For each table cluster, cluster columns using numeric profile features.

    Returns:
      - table_labels: table_name -> cluster_id (stage 1)
      - table_clusters: cluster_id -> [table_name, ...]
      - column_clusters: global column cluster ids -> [unique_id, ...]
    """
    # --- Collect unique table names from column dicts
    table_names = sorted({str(c.get("dataset_name", "")) for c in columns if c.get("dataset_name") is not None})
    table_clusters = cluster_tables_by_name(
        table_names,
        model_name=model_name,
        eps_tables=eps_tables,
        min_samples_tables=min_samples_tables
    )

    # Map table_name -> table_cluster_id
    table_labels: Dict[str, int] = {}
    for cid, names in table_clusters.items():
        for n in names:
            table_labels[n] = cid

    # --- Precompute profile matrix for all columns once
    feature_keys = _numeric_feature_keys(columns)
    P_all, ids_all = _build_profile_matrix(columns, feature_keys)
    if P_all.shape[0] == 0:
        return TableClusteringResult(table_labels=table_labels, table_clusters=table_clusters, column_clusters={})
    if scale_profiles:
        scaler = MinMaxScaler()
        P_all = scaler.fit_transform(P_all)

    # Index columns by (dataset_name)
    idx_by_table: Dict[str, List[int]] = defaultdict(list)
    for i, col in enumerate(columns):
        ds = str(col.get("dataset_name", ""))
        idx_by_table[ds].append(i)

    # Helper to run DBSCAN over a subset of rows
    def _cluster_subset(row_indices: List[int]) -> List[List[str]]:
        if not row_indices:
            return []
        subX = P_all[row_indices]
        sub_ids = [ids_all[i] for i in row_indices]

        if len(row_indices) < max(min_samples_columns, 2):
            # too small to cluster; return a single group
            return [sub_ids]

        db = DBSCAN(eps=eps_columns, min_samples=min_samples_columns)
        labs = db.fit_predict(subX)

        clusters_local: Dict[int, List[str]] = defaultdict(list)
        noise_local: List[str] = []
        for sid, lab in zip(sub_ids, labs):
            if lab == -1:
                noise_local.append(sid)
            else:
                clusters_local[lab].append(sid)

        # each noise point becomes its own singleton group
        out = [ids_ for _, ids_ in sorted(clusters_local.items(), key=lambda kv: kv[0])]
        out.extend([[nid] for nid in noise_local])
        return out

    # --- Stage 2 execution
    column_clusters: Dict[int, List[str]] = {}
    next_cluster_id = 0

    if group_mode == "per_table":
        # cluster columns independently inside each table (no cross-table grouping)
        for table in table_names:
            groups = _cluster_subset(idx_by_table.get(table, []))
            for g in groups:
                column_clusters[next_cluster_id] = g
                next_cluster_id += 1
    else:
        # Default: cluster columns within each *table cluster* (i.e., across similar tables)
        for tcid, tnames in table_clusters.items():
            # combine indices of all tables in this table-cluster
            row_indices: List[int] = []
            for t in tnames:
                row_indices.extend(idx_by_table.get(t, []))
            groups = _cluster_subset(row_indices)
            for g in groups:
                column_clusters[next_cluster_id] = g
                next_cluster_id += 1

    return TableClusteringResult(
        table_labels=table_labels,
        table_clusters=table_clusters,
        column_clusters=column_clusters
    )
