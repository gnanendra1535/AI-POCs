# lemmas_and_stems_to_csv.py
import csv
import re
import nltk
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet

# Ensure required NLTK resources are available
def ensure_nltk_resources():
    needed = ["punkt", "wordnet", "omw-1.4", "averaged_perceptron_tagger"]
    for res in needed:
        try:
            nltk.data.find(res if res.startswith("punkt") else f"corpora/{res}" if res.startswith("wordnet") or res.startswith("omw") else f"taggers/{res}" )
        except LookupError:
            nltk.download(res)

ensure_nltk_resources()

lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

def get_wordnet_pos(treebank_tag):
   
    if treebank_tag.startswith("J"):
        return wordnet.ADJ
    elif treebank_tag.startswith("V"):
        return wordnet.VERB
    elif treebank_tag.startswith("N"):
        return wordnet.NOUN
    elif treebank_tag.startswith("R"):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def Lemmatize(text: str) -> str:
   
    raw_tokens = word_tokenize(text)
    
    tokens = [t for t in raw_tokens if any(ch.isalnum() for ch in t)]
    pos_tags = nltk.pos_tag(tokens)
    lemmas = []
    for tok, tag in pos_tags:
        wn_pos = get_wordnet_pos(tag)
        lemma = lemmatizer.lemmatize(tok.lower(), pos=wn_pos)
        lemmas.append(lemma)
    return " ".join(lemmas)

def Stemmed(text: str) -> str:
   
    raw_tokens = word_tokenize(text)
    tokens = [t for t in raw_tokens if any(ch.isalnum() for ch in t)]
    stems = [stemmer.stem(t.lower()) for t in tokens]
    return " ".join(stems)

def words_with_forms(text: str):
   
    raw_tokens = word_tokenize(text)
    tokens = [t for t in raw_tokens if any(ch.isalnum() for ch in t)]
   
    seen = set()
    ordered = []
    for t in tokens:
        tl = t.lower()
        if tl not in seen:
            seen.add(tl)
            ordered.append(tl)
   
    pos_tags = nltk.pos_tag(ordered)
    results = []
    for tok, tag in pos_tags:
        wn_pos = get_wordnet_pos(tag)
        lemma = lemmatizer.lemmatize(tok, pos=wn_pos)
        stem = stemmer.stem(tok)
        results.append((tok, lemma, stem))
    return results

def save_to_csv(rows, csv_fname="lemmas_stems.csv"):
    
    with open(csv_fname, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Original Word", "Lemmatized Form", "Stemmed Form"])
        for r in rows:
            writer.writerow(r)
    print(f"Saved CSV to: {csv_fname}")

if __name__ == "__main__":
   
    text = input("Enter a string:\n").strip()
   
    text = "This is an example sentence. It is meant to show how stop words like 'is', 'an', 'it' are counted and removed."

    # Produce results
    rows = words_with_forms(text)

    # Print the table
    print("\nOriginal | Lemmatized | Stemmed")
    print("---------------------------------")
    for orig, lemma, stem in rows:
        print(f"{orig:10} | {lemma:10} | {stem}")

    # Save CSV
    save_to_csv(rows, csv_fname="lemmas_stems.csv")

    #This is an example sentence. It is meant to show how stop words like 'is', 'an', 'it' are counted and removed.
    # Original | Lemmatized | Stemmed
    # --------------------------------- 
    # This      | this       | this
    # is        | be         | is       

    # an        | an         | an       
    # example   | example    | exampl                           
    # sentence  | sentence   | sentence  
    # .         | .          | .
