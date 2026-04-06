#!/usr/bin/env python3


import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# -------------------------
# Configuration / paths
# -------------------------
INPUT_CSV = "Eopinions.csv"   # uploaded file
TRAIN_CSV = "train.csv"
TEST_CSV = "test.csv"
CLASS_FREQ_PLOT = "class_frequencies.png"
CONF_MATRIX_PLOT = "confusion_matrix.png"
ROC_PLOT = "roc_curve.png"

TEST_SIZE = 0.2
RANDOM_STATE = 42
MAX_FEATURES = 5000   # for CountVectorizer
CLASSIFIER = "logreg" # choices: "logreg" or "mnb"


nltk_needed = ["stopwords", "wordnet", "omw-1.4", "punkt"]
for res in nltk_needed:
    try:
        nltk.data.find(res)
    except Exception:
        nltk.download(res)

STOP_WORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()


def preprocess_text(text: str) -> str:
   
    if pd.isna(text):
        return ""
    txt = str(text).lower()
    # remove urls
    txt = re.sub(r"http\S+|www\S+|https\S+", " ", txt)
    # remove non-alphanumeric (keep spaces)
    txt = re.sub(r"[^a-z0-9\s]", " ", txt)
    # collapse whitespace
    txt = re.sub(r"\s+", " ", txt).strip()
    # tokenize (simple split or nltk)
    tokens = nltk.word_tokenize(txt)
    tokens = [t for t in tokens if (t not in STOP_WORDS and len(t) > 1)]
    tokens = [LEMMATIZER.lemmatize(t) for t in tokens]
    return " ".join(tokens)


def main():
    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(f"Input file not found at {INPUT_CSV}. Place Eopinions.csv at that path.")

    # 1) Read
    df = pd.read_csv(INPUT_CSV)
    if "class" not in df.columns or "text" not in df.columns:
        raise ValueError("Eopinions.csv must contain 'class' and 'text' columns.")
    print("Loaded dataset:", INPUT_CSV, "shape:", df.shape)

    # 2) Label encode 'class'
    le = LabelEncoder()
    df["Label"] = le.fit_transform(df["class"].astype(str))
    print("Class labels:", list(le.classes_))
    print("Label mapping:", dict(zip(le.classes_, le.transform(le.classes_))))

    # 3) Plot bar graph for class frequencies
    plt.figure(figsize=(8,5))
    order = df["class"].value_counts().index
    sns.countplot(x="class", data=df, order=order)
    plt.title("Class frequencies")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(CLASS_FREQ_PLOT)
    plt.close()
    print("Saved class frequency plot to:", CLASS_FREQ_PLOT)

    # 4) Preprocess text
    print("Preprocessing text... (may take a few seconds)")
    df["text_clean"] = df["text"].astype(str).apply(preprocess_text)

    # 5) Vectorize text with CountVectorizer
    vectorizer = CountVectorizer(max_features=MAX_FEATURES, ngram_range=(1,1))
    X = vectorizer.fit_transform(df["text_clean"])
    y = df["Label"].values
    print("Vectorized text shape:", X.shape)

    # 6) Stratified split (preserve class proportions)
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, df.index, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    # Save train/test CSVs (include original columns for reference)
    train_df = df.loc[idx_train].reset_index(drop=True)
    test_df = df.loc[idx_test].reset_index(drop=True)
    train_df.to_csv(TRAIN_CSV, index=False)
    test_df.to_csv(TEST_CSV, index=False)
    print(f"Saved train ({len(train_df)}) -> {TRAIN_CSV}, test ({len(test_df)}) -> {TEST_CSV}")

    # 7) Train classifier
    if CLASSIFIER == "logreg":
        clf = LogisticRegression(solver="liblinear", random_state=RANDOM_STATE)
    elif CLASSIFIER == "mnb":
        clf = MultinomialNB()
    else:
        raise ValueError("Unsupported CLASSIFIER. Use 'logreg' or 'mnb'.")

    clf.fit(X_train, y_train)
    print("Trained classifier:", CLASSIFIER)

    # 8) Evaluate on test set
    y_pred = clf.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion matrix (rows=true, cols=pred):\n", cm)
    print("\nClassification report:\n", classification_report(y_test, y_pred, target_names=le.classes_))

    # Save confusion matrix heatmap
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=le.classes_, yticklabels=le.classes_)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(CONF_MATRIX_PLOT)
    plt.close()
    print("Saved confusion matrix to:", CONF_MATRIX_PLOT)

    # 9) ROC Curve (binary only)
    unique_test_labels = np.unique(y_test)
    if len(unique_test_labels) == 2:
        # Try predict_proba else decision_function else use labels (coarse)
        try:
            y_scores = clf.predict_proba(X_test)[:, 1]
        except Exception:
            try:
                y_scores = clf.decision_function(X_test)
                if y_scores.ndim > 1:
                    y_scores = y_scores[:, 1]
            except Exception:
                # fallback to predicted labels (0/1)
                y_scores = y_pred.astype(float)

        fpr, tpr, thresholds = roc_curve(y_test, y_scores, pos_label=1)
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(6,5))
        plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
        plt.plot([0,1],[0,1], linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(ROC_PLOT)
        plt.close()
        print("Saved ROC plot to:", ROC_PLOT, f"(AUC={roc_auc:.4f})")
    else:
        print("ROC curve skipped: test labels not binary. To compute multiclass ROC, use one-vs-rest approach.")

if __name__ == "__main__":
    main()
