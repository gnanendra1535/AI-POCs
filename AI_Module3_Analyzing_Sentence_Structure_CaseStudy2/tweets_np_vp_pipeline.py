# tweets_np_vp_pipeline.py

import re
import os
import csv
import sys
import pandas as pd
import spacy
import matplotlib.pyplot as plt
from collections import defaultdict


# Configuration / Constants

INPUT_CSV = "Tweets.csv"
OUTPUT_DIR = "data"
SENTIMENTS = ["positive", "negative", "neutral"]


# Helper: load spaCy model

def load_spacy_model():
    try:
        nlp = spacy.load("en_core_web_sm")
    except Exception as e:
        print("ERROR: spaCy model 'en_core_web_sm' is not installed or could not be loaded.")
        print("Install it by running:\n    python -m spacy download en_core_web_sm\nThen re-run this script.")
        raise
    return nlp


# Helper: verb-phrase extraction

def extract_verb_phrases(doc):
   
    vps = []
    for token in doc:
        if token.pos_ == "VERB" or token.tag_.startswith("VB"):
            indices = {token.i}
            # include relevant children and their subtrees
            for child in token.children:
                if child.dep_ in {"aux", "auxpass", "neg", "prt", "advmod", "dobj", "attr", "oprd", "pobj", "dative", "xcomp", "ccomp", "prep"}:
                    # include child's subtree indices
                    for t in child.subtree:
                        indices.add(t.i)
            # also include token subtree's right/left edges for multiword verbs (rare)
            for t in token.subtree:
                indices.add(t.i)
            min_i = min(indices)
            max_i = max(indices)
            # Expand to nearest punctuation boundary on the right if it helps (stop before .,!?)
            # but keep phrase compact — don't expand over sentence punctuation.
            span = doc[min_i : max_i + 1]
            # Normalize whitespace
            phrase = span.text.strip()
            # Filter too short phrases (single auxiliaries etc.) and duplicates in adjacency
            if len(phrase) > 0:
                vps.append(phrase)
    return vps


# Main pipeline

def main():
    # Check input file
    if not os.path.exists(INPUT_CSV):
        print(f"Input CSV not found at: {INPUT_CSV}")
        print("Place Tweets.csv at that path and re-run.")
        return

    # Load spaCy
    try:
        nlp = load_spacy_model()
    except Exception:
        return

    # Read CSV
    try:
        df = pd.read_csv(INPUT_CSV, encoding="utf-8")
    except Exception as e:
        print("Failed to read CSV:", e)
        return

    # Ensure columns exist
    if "text" not in df.columns or "airline_sentiment" not in df.columns:
        print("CSV must contain 'text' and 'airline_sentiment' columns.")
        print("Columns found:", df.columns.tolist())
        return

    # 1) Extract all @tags (keep duplicates as occurrences), save to References.txt
    all_tags = []
    tag_pattern = re.compile(r"@[\w_]+")  # includes underscores
    for text in df["text"].astype(str):
        all_tags.extend(tag_pattern.findall(text))

    references_path = os.path.join(OUTPUT_DIR, "References.txt")
    with open(references_path, "w", encoding="utf-8") as f:
        for tag in all_tags:
            f.write(tag + "\n")
    print(f"Wrote {len(all_tags)} @-tags to: {references_path}")

    # Also write unique sorted references (optional)
    unique_refs_path = os.path.join(OUTPUT_DIR, "References_unique_sorted.txt")
    with open(unique_refs_path, "w", encoding="utf-8") as f:
        for tag in sorted(set(all_tags), key=lambda s: s.lower()):
            f.write(tag + "\n")
    print(f"Wrote {len(set(all_tags))} unique @-tags to: {unique_refs_path}")

    # Containers for counts (for pie charts)
    counts = {}

    # 2 & 3) Extract NP and VP per sentiment and save
    for sentiment in SENTIMENTS:
        subset = df[df["airline_sentiment"] == sentiment]
        noun_phrases = []
        verb_phrases = []

        # Process each tweet text
        # Use nlp.pipe for performance
        texts = subset["text"].astype(str).tolist()
        for doc in nlp.pipe(texts, disable=["ner"]):  # NER not needed here
            # Noun phrases via spaCy's noun_chunks (span objects)
            for nc in doc.noun_chunks:
                phrase = nc.text.strip()
                if phrase:
                    noun_phrases.append(phrase)

            # Verb phrases via rule-based extractor
            vps = extract_verb_phrases(doc)
            # Optionally filter extremely short verbs (like just 'was')? We'll keep them.
            verb_phrases.extend(vps)

        # Save to files (one phrase per line, preserving duplicates)
        np_filename = f"Noun Phrases for {sentiment} Review.txt"
        np_path = os.path.join(OUTPUT_DIR, np_filename)
        with open(np_path, "w", encoding="utf-8") as f:
            for p in noun_phrases:
                f.write(p + "\n")

        vp_filename = f"Verb Phrases for {sentiment} Review.txt"
        vp_path = os.path.join(OUTPUT_DIR, vp_filename)
        with open(vp_path, "w", encoding="utf-8") as f:
            for p in verb_phrases:
                f.write(p + "\n")

        print(f"Saved {len(noun_phrases)} noun phrases to: {np_path}")
        print(f"Saved {len(verb_phrases)} verb phrases to: {vp_path}")

        counts[sentiment] = {"np": len(noun_phrases), "vp": len(verb_phrases)}

    # 4) Make pie charts for each sentiment
    for sentiment in SENTIMENTS:
        np_count = counts[sentiment]["np"]
        vp_count = counts[sentiment]["vp"]
        labels = ["Noun Phrases", "Verb Phrases"]
        sizes = [np_count, vp_count]
        # avoid plotting empty pie
        if sum(sizes) == 0:
            print(f"No phrases for sentiment '{sentiment}' — skipping pie chart.")
            continue

        plt.figure(figsize=(6,6))
        plt.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=90)
        plt.title(f"Distribution of NP vs VP ({sentiment.capitalize()})")
        plt.axis("equal")
        chart_path = os.path.join(OUTPUT_DIR, f"{sentiment}_NP_VP_pie.png")
        plt.tight_layout()
        plt.savefig(chart_path)
        plt.close()
        print(f"Saved pie chart to: {chart_path} (NP={np_count}, VP={vp_count})")

    print("\nAll done. Output files are in:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
