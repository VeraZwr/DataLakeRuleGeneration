import pickle
import os

# Path to your .dictionary file, adjust accordingly
file_path = os.path.join("../results/Quintet", "beers", "column_profile.dictionary")

with open(file_path, "rb") as f:
    dataset_profile = pickle.load(f)

print(dataset_profile)

