# fifa_analysis.py

nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')

import nltk
from nltk.corpus import stopwords
from collections import Counter
import string


# Helper: POS Tagging


NOUN_TAGS = {"NN", "NNS", "NNP", "NNPS"}
VERB_TAGS = {"VB", "VBD", "VBG", "VBN", "VBP", "VBZ"}
PREP_TAGS = {"IN"}   # Prepositions = IN tag

# Delimiters (punctuation symbols)
DELIMITERS = set(string.punctuation)


def tokenize(text):
    
    return nltk.word_tokenize(text)


def pos_tag_text(text):
    tokens = tokenize(text)
    return nltk.pos_tag(tokens)



# A. N Most Frequent Nouns

def GetNMostFrequentNouns(text, n):
    tagged = pos_tag_text(text)
    nouns = [word.lower() for word, tag in tagged if tag in NOUN_TAGS]

    freq = Counter(nouns)
    return freq.most_common(n)



# B. N Most Frequent Verbs

def GetNMostFrequentVerbs(text, n):
    tagged = pos_tag_text(text)
    verbs = [word.lower() for word, tag in tagged if tag in VERB_TAGS]

    freq = Counter(verbs)
    return freq.most_common(n)



# C. N Most Frequent Delimiters

def GetNMostFrequentDelimiters(text, n):
    tokens = tokenize(text)
    delimiters_used = [t for t in tokens if t in DELIMITERS]

    freq = Counter(delimiters_used)
    return freq.most_common(n)



# D. N Most Frequent Prepositions

def GetNMostFrequentPrepositions(text, n):
    tagged = pos_tag_text(text)
    preps = [word.lower() for word, tag in tagged if tag in PREP_TAGS]

    freq = Counter(preps)
    return freq.most_common(n)



# Read the “FIFAWorldCup2018.txt” file & Print Results


def main():
    path = "FIFAWorldCup2018.txt"   # Make sure file exists here

    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    print("\n--- N Most Frequent Nouns ---")
    print(GetNMostFrequentNouns(text, 10))

    print("\n--- N Most Frequent Verbs ---")
    print(GetNMostFrequentVerbs(text, 10))

    print("\n--- N Most Frequent Delimiters ---")
    print(GetNMostFrequentDelimiters(text, 10))

    print("\n--- N Most Frequent Prepositions ---")
    print(GetNMostFrequentPrepositions(text, 10))


if __name__ == "__main__":
    main()
