import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder


def analyze_csv(file_path):
    # Load the CSV file into a pandas DataFrame
    df = pd.read_csv(file_path)

    # 1. Display the first few rows to get a quick look at the data
    print("First 5 rows of the dataset:")
    print(df.head())

    # 2. Get basic info about the dataset (number of rows, columns, data types)
    print("\nBasic Info:")
    print(df.info())

    # 3. Basic statistics for numeric columns
    print("\nBasic Statistics for Numeric Columns:")
    print(df.describe())

    # 4. Check for missing values
    print("\nMissing Values:")
    print(df.isnull().sum())

    # 5. Display column data types
    print("\nColumn Data Types:")
    print(df.dtypes)

    # 6. Convert categorical columns to numeric using LabelEncoder (for ordinal data)
    categorical_columns = df.select_dtypes(include=[object]).columns
    le = LabelEncoder()

    for col in categorical_columns:
        if df[col].dtype == object:
            print(f"\nConverting column '{col}' to numeric values...")
            df[col] = le.fit_transform(df[col])

    # 7. Visualizations (optional but helpful)
    # Plot distribution for numeric columns
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    if len(numeric_columns) > 0:
        print("\nHistograms for Numeric Columns:")
        df[numeric_columns].hist(figsize=(12, 8), bins=20)
        plt.tight_layout()
        plt.show()

        # Pairplot for numerical features
        print("\nPairplot for Numeric Columns:")
        sns.pairplot(df[numeric_columns])
        plt.show()

    # 8. Correlation matrix only for numeric columns
    print("\nCorrelation Matrix:")
    numeric_df = df.select_dtypes(include=[np.number])  # Only select numeric columns
    correlation_matrix = numeric_df.corr()
    print(correlation_matrix)

    # Heatmap for correlation matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Correlation Matrix')
    plt.show()

    # 9. Handle missing values (if any)
    # Optionally fill or drop missing values
    if df.isnull().sum().sum() > 0:
        print("\nHandling Missing Values:")
        df_filled = df.fillna(df.mean())  # Replace missing values with the mean of the column
        print(df_filled.head())

    return df


# Example usage
file_path = '/Users/veraz/PycharmProjects/DataLakeRuleGeneration/datasets/Quintet/hospital/dirty.csv' # Update with your CSV file path
df = analyze_csv(file_path)

