# generate_cfg.py


import os
import nltk
from collections import Counter
import string



# POS tag groups
NOUN_TAGS = {"NN", "NNS", "NNP", "NNPS"}
VERB_TAGS = {"VB", "VBD", "VBG", "VBN", "VBP", "VBZ"}
PREP_TAGS = {"IN"}  # preposition
DELIMITERS = set(string.punctuation)  # punctuation chars

def read_file(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def tokenize_and_tag(text):
    tokens = nltk.word_tokenize(text)
    tagged = nltk.pos_tag(tokens)
    return tokens, tagged

def top_k_of_pos(tagged, tags_set, k=2, lower=True):
    items = []
    for tok, tag in tagged:
        if tag in tags_set:
            items.append(tok.lower() if lower else tok)
    freq = Counter(items)
    most = [item for item, cnt in freq.most_common(k)]
    # If fewer than k items, pad with placeholder or repeat last
    if len(most) < k:
        # fill with what's available
        while len(most) < k:
            most.append(most[-1] if most else "<NONE>")
    return most

def top_k_delimiters(tokens, k=2):
    dels = [t for t in tokens if t in DELIMITERS]
    freq = Counter(dels)
    most = [item for item, cnt in freq.most_common(k)]
    if len(most) < k:
        while len(most) < k:
            most.append(most[-1] if most else "<NONE>")
    return most

def safe_terminal(token):
    
    if token == "<NONE>":
        return "'<NONE>'"
    token_escaped = token.replace("\\", "\\\\")
    if "'" not in token_escaped:
        return f"'{token_escaped}'"
    else:
        return f'"{token_escaped}"'

def build_cfg(top_dels, top_verbs, top_preps, top_nouns):
    
    # create terminals
    V_terms = " | ".join(safe_terminal(t) for t in top_verbs)
    N_terms = " | ".join(safe_terminal(t) for t in top_nouns)
    P_terms = " | ".join(safe_terminal(t) for t in top_preps)
    DEL_terms = " | ".join(safe_terminal(t) for t in top_dels)

    lines = []
    lines.append("# CFG generated from FIFAWorldCup2018.txt (top-2 tokens per category)")
    lines.append("S -> NP VP")
    lines.append("") 
    lines.append("# VP: either Verb NP  OR Verb NP PP")
    lines.append("VP -> V NP | V NP PP")
    lines.append("") 
    lines.append("# NP: either Delimiter N  OR N")
    lines.append("NP -> DEL N | N")
    lines.append("")
    lines.append("# PP: Preposition followed by NP")
    lines.append("PP -> P NP")
    lines.append("")
    # terminal expansions
    lines.append("# Terminals (V: verbs, N: nouns, P: prepositions, DEL: delimiters)")
    lines.append(f"V -> {V_terms}")
    lines.append(f"N -> {N_terms}")
    lines.append(f"P -> {P_terms}")
    lines.append(f"DEL -> {DEL_terms}")
    lines.append("")
    lines.append("# End of CFG")

    return "\n".join(lines)

def main():
    file_path = "FIFAWorldCup2018.txt"
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}. Please place FIFAWorldCup2018.txt in current directory.")
        return

    text = read_file(file_path)
    tokens, tagged = tokenize_and_tag(text)

    top2_dels = top_k_delimiters(tokens, 2)
    top2_verbs = top_k_of_pos(tagged, VERB_TAGS, 2, lower=True)
    top2_preps = top_k_of_pos(tagged, PREP_TAGS, 2, lower=True)
    top2_nouns = top_k_of_pos(tagged, NOUN_TAGS, 2, lower=True)

    print("Top-2 Delimiters:", top2_dels)
    print("Top-2 Verbs:", top2_verbs)
    print("Top-2 Prepositions:", top2_preps)
    print("Top-2 Nouns:", top2_nouns)

    cfg_text = build_cfg(top2_dels, top2_verbs, top2_preps, top2_nouns)

    out_path = "CFG.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(cfg_text)

    print(f"\nCFG saved to {out_path}\n")
    print("---- CFG content preview ----")
    print(cfg_text)

if __name__ == "__main__":
    main()
