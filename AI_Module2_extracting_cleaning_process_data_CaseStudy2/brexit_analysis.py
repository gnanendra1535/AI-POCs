# brexit_analysis.py

import docx
import nltk
from collections import Counter
import matplotlib.pyplot as plt
import re
import spacy
import os

# ---------- Ensure required NLTK data (uncomment & run once if needed) ----------
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('omw-1.4')

# ---------- Helpers ----------
def read_docx(path):
    """Read text from .docx and return combined string."""
    doc = docx.Document(path)
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip() != ""]
    return "\n".join(paragraphs)

def tokenize_text(text):
    """Tokenize into words (simple word_tokenize)."""
    tokens = nltk.word_tokenize(text)
    return tokens

def normalize_token(tok):
   
    tok = tok.strip()
    if re.fullmatch(r'[\W_]+', tok):  
        return None
    return tok

# ---------- Task 1: GetNGrams ----------
def GetNGrams(text, n):
   
    tokens = tokenize_text(text)
   
    cleaned = [t for t in [normalize_token(t) for t in tokens] if t is not None]
    ngrams = []
    for i in range(len(cleaned) - n + 1):
        ngrams.append(tuple(cleaned[i:i+n]))
    return ngrams

# ---------- POS counting functions (Task 2) ----------
# POS tag groups mapping (Penn Treebank tags)
NOUN_TAGS = {'NN', 'NNS', 'NNP', 'NNPS'}
PRONOUN_TAGS = {'PRP', 'PRP$', 'WP', 'WP$'}
ADJ_TAGS = {'JJ', 'JJR', 'JJS'}
VERB_TAGS = {'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'}
ADV_TAGS = {'RB', 'RBR', 'RBS', 'WRB'}

def _pos_counts_for_text(text):
    tokens = tokenize_text(text)
    # filter out punctuation-only tokens for better POS accuracy
    cleaned = [t for t in tokens if normalize_token(t) is not None]
    pos_tags = nltk.pos_tag(cleaned)
    # return list of (token, tag)
    return pos_tags

def NounsCount(text):
    pos_tags = _pos_counts_for_text(text)
    return sum(1 for (_, tag) in pos_tags if tag in NOUN_TAGS)

def PronounsCount(text):
    pos_tags = _pos_counts_for_text(text)
    return sum(1 for (_, tag) in pos_tags if tag in PRONOUN_TAGS)

def AdjectivesCount(text):
    pos_tags = _pos_counts_for_text(text)
    return sum(1 for (_, tag) in pos_tags if tag in ADJ_TAGS)

def VerbsCount(text):
    pos_tags = _pos_counts_for_text(text)
    return sum(1 for (_, tag) in pos_tags if tag in VERB_TAGS)

def AdverbsCount(text):
    pos_tags = _pos_counts_for_text(text)
    return sum(1 for (_, tag) in pos_tags if tag in ADV_TAGS)

# ---------- Pie chart for POS distribution ----------
def plot_pos_pie(text, save_path="pos_pie.png"):
    counts = {
        'Nouns': NounsCount(text),
        'Pronouns': PronounsCount(text),
        'Verbs': VerbsCount(text),
        'Adverbs': AdverbsCount(text),
        'Adjectives': AdjectivesCount(text)
    }
    labels = list(counts.keys())
    sizes = list(counts.values())

    # Avoid zero-sum plotting error
    if sum(sizes) == 0:
        print("No POS tags found to plot.")
        return

    plt.figure(figsize=(7,7))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
    plt.title("POS distribution (Nouns, Pronouns, Verbs, Adverbs, Adjectives)")
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved POS distribution pie chart to: {save_path}")

# ---------- Task 3: NER counts using spaCy ----------
# We'll use spaCy's en_core_web_sm model. Make sure it's downloaded.
def load_spacy_model():
    try:
        nlp = spacy.load("en_core_web_sm")
    except Exception as e:
        raise RuntimeError(
            "spaCy model 'en_core_web_sm' not found. Install it with:\n"
            "  python -m spacy download en_core_web_sm\n"
            "and then re-run this script."
        ) from e
    return nlp

def GeoPoliticalCount(text, nlp=None):
    if nlp is None:
        nlp = load_spacy_model()
    doc = nlp(text)
    return sum(1 for ent in doc.ents if ent.label_ == "GPE")

def PersonsCount(text, nlp=None):
    if nlp is None:
        nlp = load_spacy_model()
    doc = nlp(text)
    return sum(1 for ent in doc.ents if ent.label_ == "PERSON")

def OrganizationsCount(text, nlp=None):
    if nlp is None:
        nlp = load_spacy_model()
    doc = nlp(text)
    return sum(1 for ent in doc.ents if ent.label_ in {"ORG", "GPE", "NORP"} and ent.label_ == "ORG" or ent.label_ == "ORG")
    # Above line intentionally restricts to ORG. (keeps Organizations only)

# Better OrganizationsCount implementation (clear)
def OrganizationsCount(text, nlp=None):
    if nlp is None:
        nlp = load_spacy_model()
    doc = nlp(text)
    return sum(1 for ent in doc.ents if ent.label_ == "ORG")

# ---------- Task 4: Most frequent items ----------
def most_frequent_bigram(text, top_n=1, filter_stopwords=True):
    tokens = tokenize_text(text)
    cleaned = [t.lower() for t in tokens if normalize_token(t) is not None]
    if filter_stopwords:
        stop = set(nltk.corpus.stopwords.words('english'))
        cleaned = [t for t in cleaned if t not in stop]
    bigrams = GetNGrams(" ".join(cleaned), 2)
    if not bigrams:
        return None
    counts = Counter(bigrams)
    return counts.most_common(top_n)

def most_frequent_noun(text, top_n=1):
    pos_tags = _pos_counts_for_text(text)
    nouns = [tok for (tok, tag) in pos_tags if tag in NOUN_TAGS]
    if not nouns:
        return None
    counts = Counter([n.lower() for n in nouns])
    return counts.most_common(top_n)

def most_frequent_entities_by_label(text, label, nlp=None, top_n=1):
    if nlp is None:
        nlp = load_spacy_model()
    doc = nlp(text)
    ents = [ent.text for ent in doc.ents if ent.label_ == label]
    if not ents:
        return None
    counts = Counter(ents)
    return counts.most_common(top_n)

# ---------- Main / Example execution ----------
def main():
    # Path to the uploaded docx file (provided by user)
    path = "/Brexit.docx"
    if not os.path.exists(path):
        print(f"File not found at {path}. Please place Brexit.docx at that location.")
        return

    # Read the file
    text = read_docx(path)
    # (File source: the uploaded Brexit.docx). See file reference: :contentReference[oaicite:1]{index=1}

    print("\n--- Basic stats ---")
    print("Characters:", len(text))
    print("Words (approx):", len([t for t in tokenize_text(text) if normalize_token(t) is not None]))

    # Task 1: n-grams example
    print("\n--- GetNGrams examples ---")
    print("First 10 tokens:", tokenize_text(text)[:10])
    print("Bigrams (first 10):", GetNGrams(text, 2)[:10])
    print("Trigrams (first 10):", GetNGrams(text, 3)[:10])

    # Task 2: POS counts
    nouns = NounsCount(text)
    pronouns = PronounsCount(text)
    adjectives = AdjectivesCount(text)
    verbs = VerbsCount(text)
    adverbs = AdverbsCount(text)

    print("\n--- POS counts ---")
    print("Nouns:", nouns)
    print("Pronouns:", pronouns)
    print("Adjectives:", adjectives)
    print("Verbs:", verbs)
    print("Adverbs:", adverbs)

    # Plot pie chart
    plot_pos_pie(text, save_path="pos_distribution_brexit.png")

    # Task 3: NER counts
    try:
        nlp = load_spacy_model()
    except RuntimeError as e:
        print("\nERROR:", e)
        print("Skipping NER counts. Install spaCy model and re-run.")
        nlp = None

    if nlp:
        gpe_count = GeoPoliticalCount(text, nlp=nlp)
        person_count = PersonsCount(text, nlp=nlp)
        org_count = OrganizationsCount(text, nlp=nlp)
        print("\n--- NER counts (via spaCy) ---")
        print("Geo-Political Entities (GPE):", gpe_count)
        print("Persons:", person_count)
        print("Organizations:", org_count)
    else:
        gpe_count = person_count = org_count = None

    # Task 4: Most frequent bi-gram, noun, GPE, Person
    print("\n--- Most frequent items ---")
    mf_bigram = most_frequent_bigram(text, top_n=1)
    mf_noun = most_frequent_noun(text, top_n=1)
    mf_gpe = most_frequent_entities_by_label(text, "GPE", nlp=nlp, top_n=1) if nlp else None
    mf_person = most_frequent_entities_by_label(text, "PERSON", nlp=nlp, top_n=1) if nlp else None

    print("Most frequent bi-gram (filtered stopwords):", mf_bigram)
    print("Most frequent noun:", mf_noun)
    print("Most frequent GeoPolitical Entity:", mf_gpe)
    print("Most frequent person:", mf_person)

    # Also print top-5 frequent nouns & bigrams for extra context
    print("\nTop 5 nouns:", most_frequent_noun(text, top_n=5))
    print("Top 5 bigrams:", most_frequent_bigram(text, top_n=5))

if __name__ == "__main__":
    main()
