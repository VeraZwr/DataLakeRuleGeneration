import matplotlib.pyplot as plt

def plot_feature_distribution(column_profiles, feature_key):
    values = [col.get(feature_key) for col in column_profiles if col.get(feature_key) is not None]
    plt.hist(values, bins=20)
    plt.title(f"Distribution of {feature_key}")
    plt.xlabel(feature_key)
    plt.ylabel("Count")
    plt.grid(True)
    plt.show()
