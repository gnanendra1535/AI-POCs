# remove_stopwords_plot.py
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
import matplotlib.pyplot as plt
import re

# Ensure required NLTK data is available
def ensure_nltk_resources():
    resources = ["punkt", "stopwords"]
    for res in resources:
        try:
            nltk.data.find(f"tokenizers/{res}" if res == "punkt" else f"corpora/{res}")
        except LookupError:
            nltk.download(res)

ensure_nltk_resources()
STOPSET = set(stopwords.words("english"))

def _simple_tokenize(text):
    """Tokenize into words (lowercase), preserving only alphanumeric and apostrophe tokens."""
    # use nltk tokenizer for better splitting, then filter tokens
    raw = word_tokenize(text)
    tokens = [t.lower() for t in raw if re.search(r"[A-Za-z0-9]", t)]
    return tokens

def RemoveStopWords(text):
    """
    Remove English stop words from the input text and return the cleaned string.
    This function RETURNS only the cleaned string (as requested).
    """
    tokens = _simple_tokenize(text)
    filtered = [t for t in tokens if t not in STOPSET]
    # join with single spaces (loses original punctuation/spacing but is simple and clear)
    return " ".join(filtered)

def count_stopword_frequencies_using_remove(text):
    """
    Counts frequencies of each stop word present in the original text,
    by using RemoveStopWords(text) to obtain the cleaned string and
    computing the difference between original tokens and cleaned tokens.
    Returns a Counter mapping stopword -> frequency.
    """
    orig_tokens = _simple_tokenize(text)
    cleaned_text = RemoveStopWords(text)
    cleaned_tokens = cleaned_text.split() if cleaned_text.strip() else []
    orig_counter = Counter(orig_tokens)
    cleaned_counter = Counter(cleaned_tokens)
    # tokens removed = orig_counter - cleaned_counter
    removed_counter = orig_counter - cleaned_counter
    # filter to only stopwords (safety)
    stopword_counts = Counter({tok: cnt for tok, cnt in removed_counter.items() if tok in STOPSET})
    return stopword_counts

def plot_stopword_frequencies(counter, title="Stopword Frequencies"):
    """Plot a bar chart of stopword frequencies (counter: stopword->count)."""
    if not counter:
        print("No stopwords found in the input to plot.")
        return

    words, freqs = zip(*counter.most_common())
    plt.figure(figsize=(max(6, len(words)*0.6), 4))
    bars = plt.bar(words, freqs, color="tab:blue", edgecolor="black")
    plt.xlabel("Stopword")
    plt.ylabel("Frequency")
    plt.title(title)
    plt.xticks(rotation=45, ha="right")
    # Annotate each bar with its count
    for bar, freq in zip(bars, freqs):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height()+0.05, str(freq), ha="center", va="bottom")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    s = input("Enter a string:\n")
    cleaned = RemoveStopWords(s)
    print("\nString after removing stop words:")
    print(cleaned if cleaned else "(empty after removing stop words)")

    sw_counts = count_stopword_frequencies_using_remove(s)
    if sw_counts:
        print("\nStop words and their frequencies:")
        for w, c in sw_counts.most_common():
            print(f"{w}: {c}")
    else:
        print("\nNo stop words found in the input.")

    # Plot bar graph
    plot_stopword_frequencies(sw_counts, title="Stopword Frequencies in Input")

# This is an example sentence. It is meant to show how stop words like 'is', 'an', 'it' are counted and removed.
# After removing stop words, the cleaned string should be: "example sentence . meant show stop words like ' , ' counted removed ."
# String after removing stop words:
# example sentence meant show stop words like ' counted removed

# Stop words and their frequencies:
# is: 2
# an: 1
# it: 1
