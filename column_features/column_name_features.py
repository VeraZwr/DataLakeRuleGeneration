import re
import os
# for ollama import subprocess
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertTokenizer, BertModel
import torch
from scipy.special import softmax
# Load stopwords
stop_words = stopwords.words('english')

# Initialize BERT tokenizer and model once for embeddings
tokenizer_bert = BertTokenizer.from_pretrained('bert-base-uncased')
model_bert = BertModel.from_pretrained('bert-base-uncased')
model_bert.eval()

def clean_column_name(name):
    name = re.sub(r'[_\d]+', ' ', name)  # remove digits and underscores
    return name.strip().lower()

def embed_text(text):
    """Generate BERT CLS token embedding for given text."""
    cleaned = clean_column_name(text)
    inputs = tokenizer_bert(cleaned, return_tensors='pt', truncation=True, padding=True, max_length=16)
    with torch.no_grad():
        outputs = model_bert(**inputs)
        cls_embedding = outputs.last_hidden_state[:, 0, :].numpy()
    return cls_embedding
'''

def query_ollama(prompt):
    """Send prompt to local Ollama LLaMA3 model via subprocess and return text response."""
    result = subprocess.run(
        ['ollama', 'run', 'llama3'],
        input=prompt.encode(),
        stdout=subprocess.PIPE
    )
    return result.stdout.decode()
'''
'''
def classify_columns_with_scores(column_names, prototype_embeddings):
    col_embeddings = np.vstack([embed_text(c) for c in column_names])
    labels = list(prototype_embeddings.keys())
    proto_embs = np.vstack([prototype_embeddings[label] for label in labels])
    similarities = cosine_similarity(col_embeddings, proto_embs)

    results = []
    for col, sim_scores in zip(column_names, similarities):
        soft_scores = softmax(sim_scores)
        score_dict = {label: round(float(score), 3) for label, score in zip(labels, soft_scores)}
        results.append((col, score_dict))
    return results
'''
def classify_columns_with_labels(column_names, prototype_embeddings):
    """
    Assign a single best label to each column based on cosine similarity to category prototypes.
    Returns a list of (column_name, best_category, confidence_score) tuples.
    """
    col_embeddings = np.vstack([embed_text(c) for c in column_names])
    labels = list(prototype_embeddings.keys())
    proto_embs = np.vstack([prototype_embeddings[label] for label in labels])
    similarities = cosine_similarity(col_embeddings, proto_embs)

    results = []
    for col, sim_scores in zip(column_names, similarities):
        scores = softmax(sim_scores)  # turn into probability-like scores
        best_idx = np.argmax(scores)
        best_label = labels[best_idx]
        best_score = round(float(scores[best_idx]), 3)
        results.append((col, best_label)) #(col, best_label, best_score)
    return results
COLUMN_CATEGORY_PROTOTYPES = {
        'id': ['id', 'identifier', 'user id', 'uuid', 'primary key', 'pk', 'account id', 'customer id', 'order id', 'record id', 'unique id', 'tuple id'],
        'name': ['name', 'fullname', 'first name', 'last name', 'surname', 'nickname', 'username', 'contact name', 'person name', 'entity name'],
        'number': ['number', 'num', 'count', 'quantity', 'total', 'amount', 'score', 'index', 'age', 'price', 'value', 'figure'],
        'email': ['email', 'email address', 'contact email', 'user email', 'primary email', 'work email'],
        'phone': ['phone', 'phone number', 'mobile', 'cell number', 'telephone', 'fax', 'contact number'],
        'address': ['address', 'street', 'city', 'state', 'zipcode', 'postal code', 'country', 'location', 'mailing address', 'physical address'],
        'date': ['date', 'datetime', 'timestamp', 'created at', 'updated at', 'birthdate', 'dob', 'start date', 'end date', 'registration date', 'transaction date'],
        'status': ['status', 'state', 'condition', 'flag', 'active', 'enabled', 'verified', 'pending', 'completed', 'is active', 'is enabled'],
        'description': ['description', 'comment', 'note', 'details', 'remarks', 'info', 'text'],
        'category': ['category', 'type', 'label', 'code', 'tag', 'kind', 'class'],
        'title': ['title', 'job title', 'prefix', 'designation'],
        'city': ['city', 'town', 'municipality'],
        'country': ['country', 'nation', 'region'],
        'postal code': ['postal code', 'zip code'],
        'time': ['time', 'duration', 'hour', 'minute', 'second', 'scheduled time', 'actual time', 'departure time', 'arrival time', 'dep time', 'arr time'],
        'age': ['age', 'years old'],
        'amount': ['amount', 'sum', 'total cost'],
        'price': ['price', 'cost', 'unit price', 'rate'],
        'quantity': ['quantity', 'count', 'how many'],
        'source': ['src', 'source', 'origin', 'system', 'file', 'data source', 'platform'] # Added a 'source' category for 'src'
    }

'''
# hard keyword extraction
def compute_prototype_embeddings(prototypes):
    """Compute mean embedding vector per category based on prototype keywords."""
    proto_embeddings = {}
    for label, keywords in prototypes.items():
        embeddings = np.vstack([embed_text(k) for k in keywords])
        proto_embeddings[label] = np.mean(embeddings, axis=0)
    return proto_embeddings
'''

class ColumnNameFeature(BaseEstimator, TransformerMixin):
    """
    Scikit-learn transformer that classifies column names based on embedding similarity
    to a list of categories.
    """
    def __init__(self, category_prototypes): # Use the global constant as default
        self.category_prototypes = category_prototypes
        self.prototype_embeddings_ = {}

    def fit(self, X=None, y=None):
        # Compute mean embedding vector per category based on prototype keywords.
        for label, keywords in self.category_prototypes.items():
            if keywords:  # Ensure there are keywords to embed
                embeddings = np.vstack([embed_text(k) for k in keywords])
                self.prototype_embeddings_[label] = np.mean(embeddings, axis=0)
            else:  # Handle categories with no keywords provided
                print(f"[WARNING] Category '{label}' has no prototype keywords provided. Skipping embedding.")
        return self

    def transform(self, X):
        # X is expected to be a list of column names
        if not isinstance(X, list):
            raise ValueError("Input X must be a list of column names.")
        if not self.prototype_embeddings_:
            raise RuntimeError("ColumnNameFeature must be fitted before transforming.")
        labeled_results = classify_columns_with_labels(X, self.prototype_embeddings_)
        #return pd.DataFrame(labeled_results, columns=["column_name", "predicted_label"]).set_index("column_name")
        predicted_labels = [label for _, label in labeled_results]
        return predicted_labels
        #return pd.DataFrame(labeled_results, columns=["column_name", "predicted_label"]).set_index("column_name") #columns=["column_name", "predicted_label", "confidence"]
        '''
        scored_results = classify_columns_with_scores(X, self.prototype_embeddings_)
        data = []
        for col, score_dict in scored_results:
            row = {'column_name': col}
            row.update(score_dict)
            data.append(row)
        return pd.DataFrame(data).set_index('column_name')
        '''



def read_files(root_folder):
    file_count = 0
    max_files = 1
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

            else:
                print(f"\n[WARNING] No 'clean.csv' found in {subfolder}\n")

    # Deduplicate and remove stopwords
    #unique_columns = list(set(all_column_names))
    #unique_columns = [col for col in unique_columns if col.lower() not in stop_words]

    return all_column_names


if __name__ == "__main__":
    root_folder = "/Users/veraz/PycharmProjects/DataLakeRuleGeneration/datasets/Quintet"  # change this path as needed
    all_columns_from_files = read_files(root_folder)

    colname_transformer = ColumnNameFeature(category_prototypes=COLUMN_CATEGORY_PROTOTYPES )
    colname_transformer.fit()
    classified_df = colname_transformer.transform(all_columns_from_files)
    print("\n--- Column Name Classification (embedding similarity) ---")
    print(colname_transformer.transform(all_columns_from_files))

'''from transformers import pipeline

classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

result = classifier("birthdate", candidate_labels=["date", "string", "integer", "float", "boolean"])
print(result)
'''
