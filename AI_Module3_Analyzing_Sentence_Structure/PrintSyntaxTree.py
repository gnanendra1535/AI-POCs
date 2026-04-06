# syntax_tree.py
#
# Requirements:
#   pip install nltk
#   nltk.download('punkt')
#   nltk.download('averaged_perceptron_tagger')

import nltk
from nltk import pos_tag, word_tokenize, sent_tokenize
from nltk.tree import Tree


# Function: Print Syntax Tree of First Sentence


def PrintSyntaxTree(text):
    

    # Step 1: Sentence Tokenization
    sentences = sent_tokenize(text)
    if not sentences:
        print("No sentence found.")
        return

    first_sentence = sentences[0]
    print("\nFIRST SENTENCE:")
    print(first_sentence)
    print("\nSYNTAX TREE:\n")

    # Step 2: Word Tokenize + POS Tag
    tokens = word_tokenize(first_sentence)
    tagged = pos_tag(tokens)

    # Step 3: Simple Grammar (Chunking)
    grammar = r"""
        NP: {<DT|JJ|NN.*>+}          # Noun Phrase
        VP: {<VB.*><NP|PP|CLAUSE>*}  # Verb Phrase
        PP: {<IN><NP>}               # Prepositional Phrase
        CLAUSE: {<NP><VP>}           # Clause
    """

    parser = nltk.RegexpParser(grammar)
    tree = parser.parse(tagged)

    # Step 4: Print syntax tree in text format
    print(tree)

   
    try:
        tree.draw()
    except:
        pass



# Run function on FIFAWorldCup2018.txt

def main():
    path = "FIFAWorldCup2018.txt"   # Place file in same folder

    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    PrintSyntaxTree(text)


if __name__ == "__main__":
    main()
