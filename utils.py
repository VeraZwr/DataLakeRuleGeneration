import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
# Initialize the model once at the top-level
model = SentenceTransformer('all-MiniLM-L6-v2')
def extract_all_tokens(column_names):
    all_tokens = []
    for col in column_names:
        tokens = tokenize_column_name(col)
        all_tokens.extend(tokens)
    return list(set(all_tokens))

def try_strip_prefix(token, max_prefix_len=3, threshold=0.75):
    token = token.lower()
    full_emb = model.encode([token])

    for prefix_len in range(1, min(max_prefix_len+1, len(token))):
        stripped_token = token[prefix_len:]
        if len(stripped_token) < 2:  # don't strip to too short token
            continue
        stripped_emb = model.encode([stripped_token])
        sim = cosine_similarity(full_emb, stripped_emb)[0][0]
        if sim > threshold:
            return stripped_token  # stripped version likely better
    return token  # no good prefix found

def tokenize_column_name(col_name):
    # Normalize underscores and non-alphanumeric chars to spaces
    col_name = re.sub(r'[_\W]+', ' ', col_name)
    # Split camelCase words
    camel_case_split = re.sub('([a-z])([A-Z])', r'\1 \2', col_name)
    # Tokenize to alpha-only tokens with length >= 1
    tokens = re.findall(r'[a-zA-Z]{1,}', camel_case_split)
    tokens = [token.lower() for token in tokens]

    # For each token, check if first letter is a prefix and remove it if yes
    cleaned_tokens = []
    for token in tokens:
        cleaned = try_strip_prefix(token)
        cleaned_tokens.append(cleaned)
    return cleaned_tokens