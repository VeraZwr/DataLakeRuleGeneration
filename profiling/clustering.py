def cluster_columns(columns, feature_keys, n_clusters=5):
    import numpy as np
    from sklearn.cluster import KMeans

    # Build the feature matrix
    data = []
    valid_columns = []

    for col in columns:
        try:
            vec = [float(col.get(f, 0.0) or 0.0) for f in feature_keys]
            data.append(vec)
            valid_columns.append(col["column_name"])
        except Exception:
            continue  # Skip columns with missing or non-numeric values

    if len(data) == 0:
        return {}  # Nothing to cluster

    if len(data) <= n_clusters:
        # Not enough samples — each column gets its own cluster
        return {i: [name] for i, name in enumerate(valid_columns)}

    # Apply clustering
    model = KMeans(n_clusters=n_clusters, random_state=42)
    labels = model.fit_predict(np.array(data))

    # Build result dictionary: cluster_id -> list of column names
    clusters = {}
    for label, col_name in zip(labels, valid_columns):
        clusters.setdefault(label, []).append(col_name)

    return clusters

