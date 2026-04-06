def MakeCorpus(sentences):
    corpus = set()  # use set to avoid duplicates

    for sentence in sentences:
        words = sentence.strip().split()  # split words
        for w in words:
            corpus.add(w)

    return list(corpus)  # return as list