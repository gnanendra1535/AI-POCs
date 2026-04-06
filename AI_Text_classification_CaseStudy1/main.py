from Vectorization import PresenceAbsenceVectorization

# Input 3 sentences
sentences = []
for i in range(3):
    s = input(f"Enter sentence {i+1}: ")
    sentences.append(s)

# Get corpus and vectors
corpus, vectors = PresenceAbsenceVectorization(sentences)

print("\nCorpus (Union of all words):")
print(corpus)

print("\nVectors:")
for i, v in enumerate(vectors, 1):
    print(f"S{i} → {v}")

# Example usage:
# Corpus:
# ['India', 'England', 'Australia', 'won', 'the', 'match', 'cricket', 'final']

# Vectors:
# S1 → [1,0,0,1,1,1,0,0]
# S2 → [0,1,0,1,1,1,1,0]
# S3 → [0,0,1,1,1,1,0,1]