from Corpus import MakeCorpus

def PresenceAbsenceVectorization(sentences):
    corpus = MakeCorpus(sentences)  # get the list of unique words
    vectors = []

    for sentence in sentences:
        words = sentence.strip().split()
        vector = []

        for term in corpus:
            # Presence = 1, Absence = 0
            vector.append(1 if term in words else 0)

        vectors.append(vector)

    return corpus, vectors