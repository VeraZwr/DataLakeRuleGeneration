# clustering.py

import math
from typing import Iterable, List, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler, normalize

__all__ = [
    "get_numeric_feature_keys",
    "build_feature_matrix",
    "plot_k_distance",
    "cluster_columns",                         # DBSCAN
    "k_means_cluster_columns",                 # K-Means (auto features)
    "k_means_cluster_columns_with_keys",       # K-Means (explicit features)
    "cluster_columns_kmeans_seeded",           # Seeded K-Means (dict keyed by seed)
    "cluster_columns_kmeans_by_samples",       # Seeded K-Means (int keys wrapper)
    "encode_semantic",
    "encode_data_type",
]

# ------------------------------------------------------------
# Helpers: collect numeric feature keys & build feature matrix
# ------------------------------------------------------------
def get_numeric_feature_keys(
    columns: List[dict],
    exclude: Tuple[str, ...] = ("dataset_name", "column_name", "unique_id"),
) -> List[str]:
    """
    Collect numeric keys found across column profiles (excluding id/meta fields).
    """
    feature_keys = sorted({
        k for col in columns
        for k, v in col.items()
        if (
            k not in exclude
            and isinstance(v, (int, float, np.integer, np.floating))
            and not (isinstance(v, float) and math.isnan(v))
        )
    })
    return feature_keys


def build_feature_matrix(
    columns: List[dict],
    feature_keys: List[str],
    id_key: str = "unique_id",
) -> Tuple[np.ndarray, List[str]]:
    """
    Build a feature matrix X (N x D) using feature_keys and a parallel list of IDs.
    """
    X, ids = [], []
    for col in columns:
        vec = []
        for f in feature_keys:
            val = col.get(f, 0.0)
            if val is None or (isinstance(val, float) and math.isnan(val)):
                val = 0.0
            vec.append(float(val))
        X.append(vec)
        ids.append(col.get(id_key) or col.get("column_name"))
    return np.asarray(X, dtype=float), ids


# ------------------------------------------------------------
# Visualization: k-distance for DBSCAN eps selection
# ------------------------------------------------------------
def plot_k_distance(data: np.ndarray, k: int = 4) -> None:
    nbrs = NearestNeighbors(n_neighbors=k)
    nbrs.fit(data)
    distances, _ = nbrs.kneighbors(data)
    k_distances = np.sort(distances[:, k - 1])

    plt.figure(figsize=(10, 5))
    plt.plot(k_distances)
    plt.ylabel(f"Distance to {k}-th Nearest Neighbor")
    plt.xlabel("Points sorted by distance")
    plt.title("k-Distance Plot for DBSCAN eps estimation")
    plt.show()


# ------------------------------------------------------------
# DBSCAN clustering (existing)
# ------------------------------------------------------------
def cluster_columns(
    columns: List[dict],
    eps: float = 0.5,
    min_samples: int = 5,
    plot_eps: bool = False,
) -> Dict[int, List[str]]:
    """
    Cluster columns using DBSCAN over normalized numeric features.
    Returns {cluster_id: [unique_id]} (noise is omitted).
    """
    # Collect all numeric feature keys
    feature_keys = sorted({
        k for col in columns
        for k, v in col.items()
        if isinstance(v, (int, float, np.integer, np.floating))
    })

    data, valid_columns = [], []
    for col in columns:
        vec = []
        for f in feature_keys:
            val = col.get(f, 0.0)
            if val is None or (isinstance(val, float) and math.isnan(val)):
                val = 0.0
            vec.append(float(val))
        data.append(vec)
        valid_columns.append(col["unique_id"])

    if not data:
        return {}

    data = np.asarray(data, dtype=float)

    # Normalize to [0,1]
    scaler = MinMaxScaler()
    data = scaler.fit_transform(data)

    # Optional k-distance plot
    if plot_eps and min_samples > 1:
        plot_k_distance(data, k=min_samples - 1)

    # Run DBSCAN
    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(data)

    # Build clusters (skip noise = -1)
    clusters: Dict[int, List[str]] = {}
    for label, col_name in zip(labels, valid_columns):
        if label == -1:
            continue
        clusters.setdefault(label, []).append(col_name)

    # (Optional) pseudo-centers (kept silent for now)
    # cluster_vectors: Dict[int, List[np.ndarray]] = {}
    # for label, vec in zip(labels, data):
    #     if label == -1:
    #         continue
    #     cluster_vectors.setdefault(label, []).append(vec)
    # for cluster_id, vectors in cluster_vectors.items():
    #     center = np.mean(vectors, axis=0)
    #     # center_named = dict(zip(feature_keys, center))

    return clusters


# ------------------------------------------------------------
# Plain K-Means (auto feature detection)
# ------------------------------------------------------------
def k_means_cluster_columns(
    columns: List[dict],
    n_clusters: int = 5,
    id_key: str = "unique_id",
) -> Dict[int, List[str]]:
    """
    K-Means clustering using all numeric fields discovered automatically.
    Returns {cluster_id: [ids]} where ids come from id_key (fallback to column_name).
    """
    feature_keys = get_numeric_feature_keys(columns)
    if not feature_keys:
        return {}

    X, ids = build_feature_matrix(columns, feature_keys, id_key=id_key)

    if len(X) == 0:
        return {}

    # Handle small cases
    if len(X) <= n_clusters:
        return {i: [ids[i]] for i in range(len(X))}

    model = KMeans(n_clusters=n_clusters, random_state=42)
    labels = model.fit_predict(X)

    clusters: Dict[int, List[str]] = {}
    for label, col_id in zip(labels, ids):
        clusters.setdefault(label, []).append(col_id)

    return clusters


# ------------------------------------------------------------
# Plain K-Means (explicit feature keys)
# ------------------------------------------------------------
def k_means_cluster_columns_with_keys(
    columns: List[dict],
    feature_keys: List[str],
    n_clusters: int = 5,
    id_key: str = "column_name",
) -> Dict[int, List[str]]:
    """
    K-Means clustering using a provided set of feature_keys.
    Returns {cluster_id: [ids]} where ids come from id_key.
    """
    X, ids = build_feature_matrix(columns, feature_keys, id_key=id_key)

    if len(X) == 0:
        return {}

    # Adjust clusters if too few unique points
    unique_points = np.unique(X, axis=0)
    if len(unique_points) == 0:
        return {}
    if len(unique_points) < n_clusters:
        n_clusters = len(unique_points)
    if len(X) <= n_clusters:
        return {i: [ids[i]] for i in range(len(X))}

    model = KMeans(n_clusters=n_clusters, random_state=42)
    labels = model.fit_predict(X)

    clusters: Dict[int, List[str]] = {}
    for label, col_id in zip(labels, ids):
        clusters.setdefault(label, []).append(col_id)

    return clusters


# ------------------------------------------------------------
# Seeded K-Means using sample columns as initial centroids
# ------------------------------------------------------------
def cluster_columns_kmeans_seeded(
    columns: List[dict],
    sample_columns: Iterable[str],               # seed names (match column_name OR unique_id)
    feature_keys: Optional[List[str]] = None,
    id_key: str = "unique_id",                  # how to label members; falls back to column_name if missing
    centroid_key: str = "column_name",          # how to find seeds in 'columns' list
    normalize_features: bool = True,
    similarity: str = "cosine",                 # "cosine" or "euclidean"
    membership_threshold: Optional[float] = None,
    max_iter: int = 100,
    random_state: int = 42,
) -> Dict[str, List[str]]:
    """
    Each seed becomes one cluster center. We init KMeans with those seed vectors.
    Optionally filter members by threshold:
      - cosine: keep if cosine >= membership_threshold
      - euclidean: keep if distance <= membership_threshold
    Returns: {seed_name: [member_ids]} where seed_name is the seed's 'centroid_key' value.
    """
    seeds = [s.strip() for s in sample_columns if s and s.strip()]
    if not seeds:
        raise ValueError("Provide at least one seed in sample_columns.")

    # 1) Feature keys & matrix
    if feature_keys is None:
        feature_keys = get_numeric_feature_keys(columns)
        if not feature_keys:
            raise ValueError("No numeric feature keys found to build vectors.")
    X, ids = build_feature_matrix(columns, feature_keys, id_key=id_key)

    # 2) Build seed -> vector (lookup by centroid_key)
    name_to_row = {col.get(centroid_key): i for i, col in enumerate(columns)}
    missing = [s for s in seeds if s not in name_to_row]
    if missing:
        raise ValueError(f"Seed(s) not found by '{centroid_key}': {missing}")

    init_centers = np.vstack([X[name_to_row[s]] for s in seeds])

    # 3) Normalize for cosine
    if similarity == "cosine":
        if normalize_features:
            X = normalize(X)
            init_centers = normalize(init_centers)
    elif similarity != "euclidean":
        raise ValueError("similarity must be 'cosine' or 'euclidean'.")

    # 4) K-Means with fixed init
    kmeans = KMeans(
        n_clusters=len(seeds),
        init=init_centers,
        n_init=1,                 # we supply init; don't reinit
        max_iter=max_iter,
        random_state=random_state,
        algorithm="lloyd",
    )
    labels = kmeans.fit_predict(X)
    centers = kmeans.cluster_centers_

    # 5) Optional thresholding against assigned centroid
    if membership_threshold is not None:
        assigned_centers = centers[labels]
        if similarity == "cosine":
            scores = np.sum(X * assigned_centers, axis=1)        # cos since normalized
            keep_mask = scores >= membership_threshold
        else:
            dists = np.linalg.norm(X - assigned_centers, axis=1)
            keep_mask = dists <= membership_threshold
    else:
        keep_mask = np.ones(len(ids), dtype=bool)

    # 6) Assemble clusters keyed by the seed name
    clusters_by_seed: Dict[str, List[str]] = {s: [] for s in seeds}
    for i, col_id in enumerate(ids):
        if not keep_mask[i]:
            continue
        seed_for_i = seeds[labels[i]]
        clusters_by_seed[seed_for_i].append(col_id)

    # Ensure every seed appears in its own cluster
    for s in seeds:
        seed_row = name_to_row[s]
        seed_id = ids[seed_row]
        if seed_id not in clusters_by_seed[s]:
            clusters_by_seed[s].append(seed_id)

    return clusters_by_seed


# ------------------------------------------------------------
# Wrapper to match {int_id: [names]} like your DBSCAN output
# ------------------------------------------------------------
def cluster_columns_kmeans_by_samples(
    columns: List[dict],
    sample_columns: Iterable[str],
    **kwargs,
) -> Dict[int, List[str]]:
    """
    Returns {0: [...], 1: [...], ...} with deterministic order of seeds.
    """
    seeded = cluster_columns_kmeans_seeded(columns, sample_columns, **kwargs)
    return {i: sorted(members) for i, (_, members) in enumerate(seeded.items())}


# ------------------------------------------------------------
# Encoders for optional semantic/data type features
# ------------------------------------------------------------
def encode_semantic(semantic: str) -> int:
    mapping = {"identifier": 1, "price": 2, "description": 3}
    return mapping.get(semantic, 0)


def encode_data_type(data_type: str) -> int:
    mapping = {"integer": 1, "float": 2, "string": 3, "date": 4}
    return mapping.get(data_type, 0)
