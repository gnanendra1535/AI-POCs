# chunking_fifa.py


import nltk
from nltk import word_tokenize, pos_tag, sent_tokenize
from nltk.chunk import RegexpParser
import os




# Chunking patterns


# 1. Proper noun(s) followed by verb(s)
GRAMMAR_VER1 = r"""
  CHUNK: {<NNP|NNPS>+<VB.*>+}
"""

# 2. Verb followed by adjective(s)
GRAMMAR_VER2 = r"""
  CHUNK: {<VB.*><JJ.*>+}
"""

# 3. Determiner followed by Noun(s)
GRAMMAR_VER3 = r"""
  CHUNK: {<DT><NN.*>+}
"""

# 4. Verb followed by Adverb(s)
GRAMMAR_VER4 = r"""
  CHUNK: {<VB.*><RB.*>+}
"""

# 5. Delimiter (comma/colon/semicolon), Adjectives, Nouns in that order
#    Punctuation tags in NLTK are literal ',' ':' ';'
GRAMMAR_VER5 = r"""
  CHUNK: {<,|:|;><JJ.*>+<NN.*>+}
"""

# 6. Noun phrases having Nouns and Adjectives, terminated with Noun
#    (e.g., NN ... JJ ... NN) — we capture sequences with final NN.* 
GRAMMAR_VER6 = r"""
  CHUNK: {<NN.*>+<JJ.*>*<NN.*>}
"""

# Compiled parsers
PARSER_VER1 = RegexpParser(GRAMMAR_VER1)
PARSER_VER2 = RegexpParser(GRAMMAR_VER2)
PARSER_VER3 = RegexpParser(GRAMMAR_VER3)
PARSER_VER4 = RegexpParser(GRAMMAR_VER4)
PARSER_VER5 = RegexpParser(GRAMMAR_VER5)
PARSER_VER6 = RegexpParser(GRAMMAR_VER6)



# Helper functions

def _pos_tag_sentence(sentence):
    tokens = word_tokenize(sentence)
    tagged = pos_tag(tokens)
    return tagged

def _extract_chunks(tagged, parser):
    
    tree = parser.parse(tagged)
    phrases = []
    for subtree in tree:
        # chunk nodes are Tree objects; leaves are tuples
        if isinstance(subtree, nltk.Tree) and subtree.label() == 'CHUNK':
            words = [tok for tok, tag in subtree.leaves()]
            phrase = " ".join(words)
            phrases.append(phrase)
    return phrases


# -----------------------
# Required functions
# -----------------------

def ChunkingVer1(text):
    """Phrases having Proper nouns followed by Verbs"""
    # Accepts a string (may contain multiple sentences). We will run on entire text,
    # but typical usage in this task is on the first sentence.
    sentences = sent_tokenize(text)
    results = []
    for sent in sentences:
        tagged = _pos_tag_sentence(sent)
        results.extend(_extract_chunks(tagged, PARSER_VER1))
    return results

def ChunkingVer2(text):
    """Verb phrases having Verbs followed by Adjectives"""
    sentences = sent_tokenize(text)
    results = []
    for sent in sentences:
        tagged = _pos_tag_sentence(sent)
        results.extend(_extract_chunks(tagged, PARSER_VER2))
    return results

def ChunkingVer3(text):
    """Noun Phrases having Determiners followed by Nouns"""
    sentences = sent_tokenize(text)
    results = []
    for sent in sentences:
        tagged = _pos_tag_sentence(sent)
        results.extend(_extract_chunks(tagged, PARSER_VER3))
    return results

def ChunkingVer4(text):
    """Verb Phrases having Verbs followed by Adverbs"""
    sentences = sent_tokenize(text)
    results = []
    for sent in sentences:
        tagged = _pos_tag_sentence(sent)
        results.extend(_extract_chunks(tagged, PARSER_VER4))
    return results

def ChunkingVer5(text):
    """Phrases having Delimiter, Adjectives and Nouns in that order."""
    sentences = sent_tokenize(text)
    results = []
    for sent in sentences:
        tagged = _pos_tag_sentence(sent)
        results.extend(_extract_chunks(tagged, PARSER_VER5))
    return results

def ChunkingVer6(text):
    """Noun Phrases having Nouns and Adjectives, terminated with Nouns."""
    sentences = sent_tokenize(text)
    results = []
    for sent in sentences:
        tagged = _pos_tag_sentence(sent)
        results.extend(_extract_chunks(tagged, PARSER_VER6))
    return results


# -----------------------
# Run all functions on the FIRST sentence of FIFAWorldCup2018.txt
# -----------------------
def main():
    path = "FIFAWorldCup2018.txt"
    if not os.path.exists(path):
        print(f"File not found: {path}\nPlease place FIFAWorldCup2018.txt in the script folder and re-run.")
        return

    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    # Extract first sentence
    sents = sent_tokenize(text)
    if not sents:
        print("No sentences found in the file.")
        return
    first_sentence = sents[0].strip()
    print("FIRST SENTENCE:\n", first_sentence, "\n")

    # Run chunkers (on the first sentence only)
    print("ChunkingVer1 (Proper Nouns followed by Verbs):")
    print(ChunkingVer1(first_sentence))
    print()

    print("ChunkingVer2 (Verbs followed by Adjectives):")
    print(ChunkingVer2(first_sentence))
    print()

    print("ChunkingVer3 (Determiners followed by Nouns):")
    print(ChunkingVer3(first_sentence))
    print()

    print("ChunkingVer4 (Verbs followed by Adverbs):")
    print(ChunkingVer4(first_sentence))
    print()

    print("ChunkingVer5 (Delimiter, Adjectives, Nouns):")
    print(ChunkingVer5(first_sentence))
    print()

    print("ChunkingVer6 (Nouns + optional Adjectives, terminated with Noun):")
    print(ChunkingVer6(first_sentence))
    print()


if __name__ == "__main__":
    main()
