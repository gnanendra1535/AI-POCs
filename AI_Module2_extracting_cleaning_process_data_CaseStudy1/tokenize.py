# variant_b.py
import nltk
from collections import Counter
import sys

def ensure_punkt():
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")

def Tokenize(text, use_nltk=True):
   
    if use_nltk:
        ensure_punkt()
        raw_tokens = nltk.word_tokenize(text)
        # Keep tokens that contain at least one alphanumeric character
        tokens = [t.lower() for t in raw_tokens if any(ch.isalnum() for ch in t)]
        return tokens
    else:
        # fallback to simple regex tokenizer if use_nltk is False
        import re
        return re.findall(r"\b[a-z0-9']+\b", text.lower())

def print_token_info(tokens):
    cnt = Counter(tokens)
    print("Token | Frequency")
    print("-----------------")
    for token, freq in sorted(cnt.items(), key=lambda x: (-x[1], x[0])):
        print(f"{token:10} | {freq}")

    least = sorted(cnt.items(), key=lambda x: (x[1], x[0]))[:5]
    print("\n5 least occurring tokens (token: frequency):")
    for token, freq in least:
        print(f"{token}: {freq}")

if __name__ == "__main__":
    s = input("Enter a string: ")
    # set use_nltk True to use NLTK tokenizer
    tokens = Tokenize(s, use_nltk=True)
    print("\nTokens:", tokens)
    print()
    print_token_info(tokens)

    #Hello, hello! This is a test. This test: tests, tester's tests.
# Tokens: ['hello', 'hello', 'this', 'is', 'a', 'test', 'this', 'test', 'tests', "tester's", 'tests']

# Token | Frequency
# -----------------
# test       | 2
# tests      | 2
# hello      | 2
# this       | 2
# a          | 1
# is         | 1
# tester's   | 1

# 5 least occurring tokens (token: frequency):
# a: 1
# is: 1
# tester's: 1
# hello: 2
# test: 2

