"""
NLTK Tasks Script
Performs:
1. import nltk
2. list corpora modules available
3. import twitter_samples
4. print files in twitter_samples
5. download 'swadesh'
6. list Gutenberg files
7. print contents of 'shakespeare-macbeth.txt'
8. save that content to 'Macbeth.txt'
9. count articles (a, an, the), remove them, save 'Macbeth-ArticlesRemoved.txt'
10. remove punctuation, save 'Macbeth-punctuationsRemoved.txt'
11. download 'names' corpus
12. print total number of words in names corpus
13. frequency of names by starting alphabet
14-16. plot frequency bar graph with labels/title and save PNG
"""

import os
import re
import string
import pkgutil
import nltk
from nltk.corpus import twitter_samples, gutenberg, names
import matplotlib.pyplot as plt

# ---------- CONFIG ----------
OUT_DIR = os.getcwd()
MACBETH_TXT = os.path.join(OUT_DIR, "Macbeth.txt")
MACBETH_ART_REMOVED = os.path.join(OUT_DIR, "Macbeth-ArticlesRemoved.txt")
MACBETH_PUNCT_REMOVED = os.path.join(OUT_DIR, "Macbeth-punctuationsRemoved.txt")
BAR_GRAPH_PNG = os.path.join(OUT_DIR, "FrequencyOfNamesOfEachAlphabet.png")
# ----------------------------

def ensure_nltk_resource(resource_name):
    """Download resource if not present."""
    try:
        nltk.data.find(resource_name)
    except LookupError:
        print(f"Downloading NLTK resource: {resource_name} ...")
        nltk.download(resource_name.split('/')[-1])

def main():
    # 1. Import nltk (already imported above)
    print("1) NLTK imported. Version:", nltk.__version__)

    # 2. List all document modules present in nltk.corpus (top-level)
    print("\n2) Top-level corpus modules available in nltk.corpus (module names):")
    corpus_pkg = nltk.corpus
    modules = [m.name for m in pkgutil.iter_modules(corpus_pkg.__path__)]
    print(", ".join(modules))

    # Additionally: list a bunch of fileids across installed corpora (useful)
    print("\n(Also showing fileids for some standard corpora if present)")

    # ensure gutenberg and names are downloadable later
    ensure_nltk_resource('corpora/gutenberg')
    ensure_nltk_resource('corpora/names')

    # 3. Import 'twitter_samples' document
    # Ensure twitter_samples is downloaded
    ensure_nltk_resource('corpora/twitter_samples')
    from nltk.corpus import twitter_samples as tw  # local alias
    print("\n3) twitter_samples imported as 'tw'.")

    # 4. Print all the files present in 'twitter_samples'
    print("\n4) Files in twitter_samples:")
    try:
        print(tw.fileids())
    except Exception as e:
        print("  Could not list twitter_samples fileids:", e)

    # 5. Download 'swadesh' corpora
    ensure_nltk_resource('corpora/swadesh')
    print("\n5) 'swadesh' corpora ensured/downloaded.")

    # 6. Print all files present in nltk Gutenberg corpora
    print("\n6) Gutenberg fileids:")
    try:
        print(gutenberg.fileids())
    except Exception as e:
        print("  Could not list Gutenberg fileids:", e)

    # 7. Print the contents of 'shakespeare-macbeth.txt'
    target_gid = 'shakespeare-macbeth.txt'
    print(f"\n7) Printing first 1000 characters of '{target_gid}':")
    try:
        macbeth_raw = gutenberg.raw(target_gid)
        print(macbeth_raw[:1000])  # print first 1000 chars for console brevity
    except Exception as e:
        print("  Could not read", target_gid, ":", e)
        macbeth_raw = ""

    # 8. Save 'shakespeare-macbeth.txt' contents to 'Macbeth.txt'
    if macbeth_raw:
        with open(MACBETH_TXT, "w", encoding="utf-8") as f:
            f.write(macbeth_raw)
        print(f"\n8) Saved Macbeth content to: {MACBETH_TXT}")
    else:
        print("\n8) Macbeth content unavailable; skipping save.")

    # 9. Count number of articles ('a', 'an', 'the') in Macbeth.txt and save file with them removed
    if macbeth_raw:
        # Count occurrences as whole words (case-insensitive)
        articles_pattern = re.compile(r'\b(a|an|the)\b', flags=re.IGNORECASE)
        matches = articles_pattern.findall(macbeth_raw)
        num_articles = len(matches)
        print(f"\n9) Number of articles (a, an, the) found: {num_articles}")

        # Remove articles (replace with single space to avoid word joins)
        macbeth_no_articles = articles_pattern.sub(" ", macbeth_raw)

        # Normalize multiple spaces to single spaces (optional)
        macbeth_no_articles = re.sub(r'\s+', ' ', macbeth_no_articles)

        with open(MACBETH_ART_REMOVED, "w", encoding="utf-8") as f:
            f.write(macbeth_no_articles.strip())
        print(f"   Saved file without articles to: {MACBETH_ART_REMOVED}")
    else:
        print("\n9) Macbeth text not available - skipping article removal.")

    # 10. Remove all punctuations from the file and save
    if macbeth_raw:
        # define punctuation set (use string.punctuation)
        punct_pattern = re.compile(r'[{}]'.format(re.escape(string.punctuation)))
        macbeth_punct_removed = punct_pattern.sub("", macbeth_raw)

        # collapse multiple whitespace characters
        macbeth_punct_removed = re.sub(r'\s+', ' ', macbeth_punct_removed)

        with open(MACBETH_PUNCT_REMOVED, "w", encoding="utf-8") as f:
            f.write(macbeth_punct_removed.strip())
        print(f"\n10) Saved file without punctuations to: {MACBETH_PUNCT_REMOVED}")
    else:
        print("\n10) Macbeth text not available - skipping punctuation removal.")

    # 11. Download 'names' corpora using nltk.download
    ensure_nltk_resource('corpora/names')
    print("\n11) 'names' corpus ensured/downloaded.")

    # 12. Print total number of words in 'names' corpora of nltk
    try:
        name_words = names.words()
        total_names = len(name_words)
        print(f"\n12) Total number of name entries in names corpus: {total_names}")
    except Exception as e:
        print("  Could not access names.words():", e)
        name_words = []

    # 13. For each alphabet print the frequency of names present in 'names' corpora
    from collections import Counter
    freq_by_alpha = Counter()
    for nm in name_words:
        if not nm: 
            continue
        first = nm[0].upper()
        if 'A' <= first <= 'Z':
            freq_by_alpha[first] += 1
        else:
            freq_by_alpha['Other'] += 1

    print("\n13) Frequency of names by starting alphabet:")
    for ch in list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"):
        print(f"  {ch}: {freq_by_alpha.get(ch, 0)}")

    # 14-16. Plot bar graph of frequency of names for each alphabet and save PNG
    alphabets = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    counts = [freq_by_alpha.get(ch, 0) for ch in alphabets]

    plt.figure(figsize=(12,6))
    plt.bar(alphabets, counts)
    plt.title("Frequency of Names Starting With Each Alphabet")
    plt.xlabel("Alphabet (First letter of name)")
    plt.ylabel("Number of names")
    plt.tight_layout()

    # Save the figure to file
    plt.savefig(BAR_GRAPH_PNG, dpi=300)
    plt.close()
    print(f"\n14-16) Bar graph saved to: {BAR_GRAPH_PNG}")

if __name__ == "__main__":
    main()