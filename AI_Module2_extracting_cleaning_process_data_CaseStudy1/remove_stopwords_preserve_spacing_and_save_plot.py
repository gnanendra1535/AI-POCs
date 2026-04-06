# remove_stopwords_preserve_spacing_and_save_plot.py
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
import matplotlib.pyplot as plt

# Ensure required NLTK data is available
def ensure_nltk_resources():
    for res, kind in [("punkt", "tokenizers"), ("stopwords", "corpora")]:
        try:
            nltk.data.find(f"{kind}/{res}")
        except LookupError:
            nltk.download(res)

ensure_nltk_resources()
STOPSET = set(stopwords.words("english"))

def RemoveStopWords(text: str) -> str:
   
    # Pattern to match words including simple contractions (e.g., "don't", "we're")
    word_re = re.compile(r"\b([A-Za-z]+(?:'[A-Za-z]+)?)\b")

    def repl(m):
        token = m.group(1)
        if token.lower() in STOPSET:
            # Replace the word characters with empty string (preserve surrounding chars)
            return ""
        else:
            return token

    # Use sub to remove only the matched word text if it's a stopword
    cleaned = word_re.sub(repl, text)
    return cleaned

def count_stopword_frequencies(text: str) -> Counter:
   
    raw_tokens = word_tokenize(text)
    tokens = [t.lower() for t in raw_tokens if any(ch.isalnum() for ch in t)]
    sw_counts = Counter(tok for tok in tokens if tok in STOPSET)
    return sw_counts

def plot_and_save_stopword_frequencies(counter: Counter,
                                       png_fname="stopword_freqs.png",
                                       svg_fname="stopword_freqs.svg",
                                       title="Stopword Frequencies"):
    
    if not counter:
        print("No stopwords found to plot.")
        return

    words, freqs = zip(*counter.most_common())
    plt.figure(figsize=(max(6, len(words)*0.6), 4))
    bars = plt.bar(words, freqs, color="tab:blue", edgecolor="black")
    plt.xlabel("Stopword")
    plt.ylabel("Frequency")
    plt.title(title)
    plt.xticks(rotation=45, ha="right")

    # Annotate
    for bar, freq in zip(bars, freqs):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height()+0.05, str(freq),
                 ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    # Save files
    plt.savefig(png_fname, dpi=300)
    plt.savefig(svg_fname)
    print(f"Saved bar chart to: {png_fname} and {svg_fname}")
    plt.show()

if __name__ == "__main__":
    s = input("Enter a string:\n")
    cleaned = RemoveStopWords(s)

    print("\nString after removing stop words (original punctuation & spacing preserved):")
    # Show result visibly (surround with markers so multiple spaces are visible)
    print("-----START-----")
    print(cleaned)
    print("------END------")

    sw_counts = count_stopword_frequencies(s)
    if sw_counts:
        print("\nStop words and their frequencies:")
        for w, c in sw_counts.most_common():
            print(f"{w}: {c}")
    else:
        print("\nNo stop words found in the input.")

    # Save plot to both PNG and SVG (filenames can be changed)
    plot_and_save_stopword_frequencies(sw_counts,
                                       png_fname="stopword_freqs.png",
                                       svg_fname="stopword_freqs.svg",
                                       title="Stopword Frequencies in Input")
