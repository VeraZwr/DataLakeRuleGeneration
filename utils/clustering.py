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
""""
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