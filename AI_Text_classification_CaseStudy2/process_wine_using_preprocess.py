

import os
import json
import pandas as pd
import numpy as np

# sklearn vectorizers
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


import PreProcess as pp  # PreProcess.Refine will be used


# Path to CSV
INPUT_CSV = "Wine.csv"
OUTPUT_CSV = INPUT_CSV  # overwrite as requested; change if you want a new file

def tokens_to_string(tokens):
    """Join tokens into a single space-separated string for vectorizers."""
    if isinstance(tokens, (list, tuple)):
        return " ".join(tokens)
    elif pd.isna(tokens):
        return ""
    else:
        return str(tokens)

def main():
    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(f"{INPUT_CSV} not found. Put Wine.csv at this path.")

    # 1. Read CSV
    df = pd.read_csv(INPUT_CSV)

    # Ensure 'description' column exists
    if 'description' not in df.columns:
        raise KeyError("Column 'description' not found in CSV.")

    # 2. Use PreProcess.Refine to get pre-processed tokens for each description
    #    PreProcess.Refine returns a list of tokens (as in your PreProcess.py).
    #    We'll store both the token list and the joined-string (for vectorizers).
    refined_tokens_list = []
    refined_joined_texts = []

    for idx, text in df['description'].fillna("").iteritems():
        try:
            tokens = pp.Refine(str(text))            # returns list of tokens
        except Exception as e:
            # In case Refine expects well-formed text or NLTK data missing, handle gracefully
            print(f"Warning: Refine failed at row {idx}, reason: {e}")
            tokens = []
        refined_tokens_list.append(tokens)
        refined_joined_texts.append(tokens_to_string(tokens))

    # Add columns to dataframe
    df['Refined-Description-Tokens'] = refined_tokens_list         # list of tokens
    df['Refined-Description'] = refined_joined_texts               # joined string for vectorizers

    # 3. Vectorization
    # CountVectorizer (counts) and TfidfVectorizer (tfidf values)
    # Fit on the whole Refined-Description column
    count_vect = CountVectorizer()
    tfidf_vect = TfidfVectorizer()

    # Fit & transform (resulting matrices are sparse)
    count_matrix = count_vect.fit_transform(df['Refined-Description'])
    tfidf_matrix = tfidf_vect.fit_transform(df['Refined-Description'])

    # 4. Convert each row vector to a plain Python list (so it can be stored in CSV as JSON)
   
    def sparse_row_to_list(sparse_row):
        # sparse_row is a (1, n_features) sparse matrix
        return sparse_row.toarray().ravel().tolist()

    # Convert matrices to lists row-by-row
    count_vectors = [sparse_row_to_list(count_matrix[i]) for i in range(count_matrix.shape[0])]
    tfidf_vectors = [sparse_row_to_list(tfidf_matrix[i]) for i in range(tfidf_matrix.shape[0])]

    # 5. Attach vectors to dataframe (store as JSON strings to keep CSV valid)
    df['CountVectorizer'] = [json.dumps(v) for v in count_vectors]
    df['TF-IDF Vectorizer'] = [json.dumps(v) for v in tfidf_vectors]

    # Save the vocabularies (so user knows which index corresponds to which feature)
    count_vocab = count_vect.get_feature_names_out().tolist()
    tfidf_vocab = tfidf_vect.get_feature_names_out().tolist()

    # We can't easily store large lists in CSV meta, so we'll save vocabs as separate JSON files
    vocab_dir = "vector_vocab"
    os.makedirs(vocab_dir, exist_ok=True)
    with open(os.path.join(vocab_dir, "count_vocab.json"), "w", encoding="utf-8") as f:
        json.dump(count_vocab, f, ensure_ascii=False, indent=2)
    with open(os.path.join(vocab_dir, "tfidf_vocab.json"), "w", encoding="utf-8") as f:
        json.dump(tfidf_vocab, f, ensure_ascii=False, indent=2)

    # 6. Save the modified CSV
   
    df.to_csv(OUTPUT_CSV, index=False)

    # Summary print
    print("Processing complete.")
    print(f"Rows processed: {len(df)}")
    print(f"Count vocabulary size: {len(count_vocab)}")
    print(f"TF-IDF vocabulary size: {len(tfidf_vocab)}")
    print(f"Saved updated CSV to: {OUTPUT_CSV}")
    print(f"Saved vocabularies to: {vocab_dir}/count_vocab.json and tfidf_vocab.json")
    print("Columns added: 'Refined-Description-Tokens', 'Refined-Description', 'CountVectorizer', 'TF-IDF Vectorizer'")

if __name__ == "__main__":
    main()
