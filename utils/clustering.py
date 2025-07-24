import math
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt
import pandas as pd

def plot_k_distance(data, k=4):
    nbrs = NearestNeighbors(n_neighbors=k)
    nbrs.fit(data)
    distances, indices = nbrs.kneighbors(data)
    k_distances = distances[:, k - 1]
    k_distances = np.sort(k_distances)

    plt.figure(figsize=(10, 5))
    plt.plot(k_distances)
    plt.ylabel(f"Distance to {k}-th Nearest Neighbor")
    plt.xlabel("Points sorted by distance")
    plt.title("k-Distance Plot for DBSCAN eps estimation")
    plt.show()

def cluster_columns(columns, eps=0.5, min_samples=5,plot_eps=False):
    # Collect all numeric feature keys
    feature_keys = sorted({
        k for col in columns
        for k, v in col.items()
        if isinstance(v, (int, float))
    })

    data = []
    valid_columns = []

    for col in columns:
        vec = []
        for f in feature_keys:
            val = col.get(f, 0.0)
            if val is None or (isinstance(val, float) and math.isnan(val)):
                val = 0.0  # or use a better default
            vec.append(float(val))
        data.append(vec)
        valid_columns.append(col["column_name"])

    if len(data) == 0:
        return {}

    data = np.array(data)
    #normalize
    scaler = MinMaxScaler()
    data = scaler.fit_transform(data)
    # k-distance plot
    if plot_eps:
        plot_k_distance(data, k=min_samples - 1)
        # You may want to exit here to inspect manually, or run DBSCAN after

    # Run DBSCAN
    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(data)

    clusters = {}
    for label, col_name in zip(labels, valid_columns):
        if label == -1:
            # -1 means noise, skip if you don't want to keep unclustered columns
            continue
        clusters.setdefault(label, []).append(col_name)

    # ---- Pseudo-centers ----
    cluster_vectors = {}
    for label, vec in zip(labels, data):
        if label == -1:
            continue
        cluster_vectors.setdefault(label, []).append(vec)

    #print("\nCluster centers (pseudo-centroids):")
    for cluster_id, vectors in cluster_vectors.items():
        center = np.mean(vectors, axis=0)
        center_named = dict(zip(feature_keys, center))
        #print(f"Cluster {cluster_id}:")
        #for k, v in center_named.items():
        #    print(f"  {k}: {v:.4f}")
        print()

    return clusters

from sklearn.cluster import KMeans

def k_means_cluster_columns(columns, n_clusters=5):
    # Collect all numeric feature keys
    feature_keys = sorted({
        k for col in columns
        for k, v in col.items()
        if isinstance(v, (int, float))
    })

    data = []
    valid_columns = []

    for col in columns:
        vec = [float(col.get(f, 0.0) or 0.0) for f in feature_keys]
        data.append(vec)
        valid_columns.append(col["column_name"])

    if len(data) == 0:
        return {}

    if len(data) <= n_clusters:
        return {i: [name] for i, name in enumerate(valid_columns)}

    model = KMeans(n_clusters=n_clusters, random_state=42)
    labels = model.fit_predict(np.array(data))

    clusters = {}
    for label, col_name in zip(labels, valid_columns):
        clusters.setdefault(label, []).append(col_name)

    return clusters

def k_means_cluster_columns(columns, feature_keys, n_clusters=5):
    data = []
    valid_columns = []

    # Build the feature matrix
    for col in columns:
        vec = []
        valid = True
        for f in feature_keys:
            val = col.get(f)
            if val is None:
                valid = False
                break
            try:
                vec.append(float(val))
            except ValueError:
                valid = False
                break
        if valid:
            data.append(vec)
            valid_columns.append(col["column_name"])

    if len(data) == 0:
        return {}  # No usable columns

    data = np.array(data)

    # Adjust number of clusters based on unique points
    unique_points = np.unique(data, axis=0)
    if len(unique_points) < n_clusters:
        n_clusters = len(unique_points)
        if n_clusters == 0:
            return {}  # Nothing unique to cluster

    if len(data) <= n_clusters:
        return {i: [name] for i, name in enumerate(valid_columns)}

    # Apply clustering
    model = KMeans(n_clusters=n_clusters, random_state=42)
    labels = model.fit_predict(data)

    # Map cluster_id to column names
    clusters = {}
    for label, col_name in zip(labels, valid_columns):
        clusters.setdefault(label, []).append(col_name)

    return clusters

# use data profile for different type of clusters
# content-based, structure-based, quality-based -> semantic, structural, statistical
# Step 1: Build feature dataframe
def encode_semantic(semantic):
    mapping = {'identifier': 1, 'price': 2, 'description': 3}
    return mapping.get(semantic, 0)

def encode_data_type(data_type):
    mapping = {'integer': 1, 'float': 2, 'string': 3, 'date': 4}
    return mapping.get(data_type, 0)


# Modified function: Three strategy clustering based on numeric features using DBSCAN
import numpy as np
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import DBSCAN


def cluster_columns_by_strategy(columns, eps=0.5, min_samples=5, plot_eps=False):
    # Strategy 1: ID-like columns (high unique_ratio, low null_ratio)
    # Strategy 2: Numeric fields (price, amount, measurable quantities)
    # Strategy 3: Text-like fields (character ratios, word length)

    strategy_clusters = {'id_like': [], 'numeric': [], 'text_like': [], 'unclustered': []}

    for col in columns:
        name = col.get('column_name', 'unknown')
        data_type = col.get('basic_data_type', 'unknown')
        unique_ratio = col.get('unique_ratio', 0)
        null_ratio = col.get('null_ratio', 1)
        char_alpha = col.get('characters_alphabet', 0)
        char_numeric = col.get('characters_numeric', 0)
        avg_word_len = col.get('words_length_avg', 0)

        # Strategy 1: ID-like columns
        if data_type in ['integer', 'string'] and unique_ratio >= 0.95 and null_ratio <= 0.1:
            strategy_clusters['id_like'].append(name)

        # Strategy 2: Numeric measurable fields
        elif data_type in ['integer', 'float'] and unique_ratio < 0.95 and null_ratio <= 0.5:
            strategy_clusters['numeric'].append(name)

        # Strategy 3: Text-like fields (more alphabetic characters, longer word length)
        elif data_type == 'string' and char_alpha >= 0.5 and avg_word_len >= 3:
            strategy_clusters['text_like'].append(name)

        else:
            strategy_clusters['unclustered'].append(name)

    return strategy_clusters


# Example usage on mock data
