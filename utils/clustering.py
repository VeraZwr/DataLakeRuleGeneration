import math

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

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

    return clusters

""""KMeans-------------
import numpy as np
from sklearn.cluster import KMeans

def cluster_columns(columns, n_clusters=5):
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

def cluster_columns(columns, feature_keys, n_clusters=5):
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
"""