import pandas as pd
import numpy as np
import re
from collections import Counter
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


# LOAD TRAIN DATA

train_path = "Tweets-train.csv"
test_path = "Tweets-test.csv"

df = pd.read_csv(train_path)

# Keep only required columns
df = df[['airline_sentiment', 'text']]
df.columns = ['sentiment', 'text']
df.head()


# RANDOMLY OBSERVE 10 TWEETS PER SENTIMENT

for s in df['sentiment'].unique():
    print(f"\n===== Sentiment: {s} =====\n")
    sample = df[df['sentiment']==s].sample(10, random_state=42)
    for t in sample['text']:
        print(t)
        print("-"*50)


# CLEANING FUNCTION


def clean_tweet(text):
    text = str(text).lower()

    # remove @mentions
    text = re.sub(r"@\w+", " ", text)

    # remove urls
    text = re.sub(r"http\S+|https\S+|www\.\S+", " ", text)

    # remove emoticons
    emoticons_pattern = r'[:;=][-~]?[)(DPp]'
    text = re.sub(emoticons_pattern, " ", text)

    # remove punctuation
    text = re.sub(r"[^a-z\s]", " ", text)

    # remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


df['clean_text'] = df['text'].apply(clean_tweet)
df.head()


# FUNCTION TO GET TOP 15 WORDS

def top_words(series, n=15):
    all_words = " ".join(series).split()
    return Counter(all_words).most_common(n)



# TOP 15 WORDS PER SENTIMENT (BEFORE STOPWORDS REMOVAL)

print("\n==== Top words BEFORE stopword removal ====\n")

for s in df['sentiment'].unique():
    words = top_words(df[df['sentiment']==s]['clean_text'])
    print(f"\nSentiment = {s}")
    print(words)


# REMOVE STOPWORDS

stopwords = set(ENGLISH_STOP_WORDS)

def remove_stopwords(text):
    return " ".join([w for w in text.split() if w not in stopwords])

df['clean_no_stop'] = df['clean_text'].apply(remove_stopwords)


# TOP 15 WORDS AFTER STOPWORDS REMOVAL

print("\n==== Top words AFTER stopword removal ====\n")

top_words_to_remove = {}

for s in df['sentiment'].unique():
    words = top_words(df[df['sentiment']==s]['clean_no_stop'])
    top_words_to_remove[s] = [w for w, c in words]
    print(f"\nSentiment = {s}")
    print(words)


# REMOVE THESE TOP WORDS FROM ALL TWEETS

all_words_remove = set()
for lst in top_words_to_remove.values():
    all_words_remove.update(lst)

def remove_top_common_words(text):
    return " ".join([w for w in text.split() if w not in all_words_remove])

df['final_clean'] = df['clean_no_stop'].apply(remove_top_common_words)

df.head()

# Save cleaned dataset if needed
df.to_csv("Tweets-train-cleaned.csv", index=False)
print("\nSaved cleaned file: Tweets-train-cleaned.csv")

