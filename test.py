import os
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans

from column_features.data_type_features import DataTypeFeatures
from column_features.column_name_features import ColumnNameFeature
from utils import extract_all_tokens  # if you created utils.py

def read_files(root_folder):
    file_count = 0
    max_files = 3
    all_column_names = []

    for subfolder in os.listdir(root_folder):
        if file_count >= max_files:
            break

        subfolder_path = os.path.join(root_folder, subfolder)

        if os.path.isdir(subfolder_path):
            csv_file = os.path.join(subfolder_path, 'clean.csv')
            if os.path.exists(csv_file):
                file_count += 1
                print(f"\n--- First 10 rows of {subfolder}/clean.csv ---\n")
                df = pd.read_csv(csv_file)
                print(df.head(10))

                all_column_names.extend(df.columns.tolist())

                dtype_transformer = DataTypeFeatures()
                transformed_df = dtype_transformer.fit_transform(df)
                print(f"\n--- Data Type Features for {subfolder}/clean.csv ---\n")
                print(transformed_df)

            else:
                print(f"\n[WARNING] No 'clean.csv' found in {subfolder}\n")

    print("\n--- Learning keywords from column names across files ---")
    tokens = extract_all_tokens(all_column_names)

    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(tokens)

    num_clusters = 20
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    labels = kmeans.fit_predict(embeddings)

    df_clusters = pd.DataFrame({'token': tokens, 'cluster': labels})
    grouped = df_clusters.groupby('cluster')['token'].apply(list)
    learned_keywords = [tokens[0] for tokens in grouped]

    print("Learned keywords:", learned_keywords)

    colname_transformer = ColumnNameFeature(keywords=learned_keywords)
    features_df = colname_transformer.transform(all_column_names)
    print("\n--- Column Name Features ---")
    print(features_df)

    return all_column_names


if __name__ == "__main__":
    root_folder = "/Users/veraz/PycharmProjects/DataLakeRuleGeneration/datasets/Quintet"  # change this
    read_files(root_folder)
