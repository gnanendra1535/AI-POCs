# PreProcess.py


import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# 1. Tokenize Function

def Tokenize(text):
    
    return word_tokenize(text)



# 2. RemoveStopWords Function

def RemoveStopWords(tokens):
   
    stop_words = set(stopwords.words("english"))
    filtered = [word for word in tokens if word.lower() not in stop_words]
    return filtered



# 3. Lemmatize Function

def Lemmatize(tokens):
   
    lemmatizer = WordNetLemmatizer()
    lemmatized = [lemmatizer.lemmatize(word) for word in tokens]
    return lemmatized



# 4. Refine Function

def Refine(text):
    
    # Step 1: Tokenize
    tokens = Tokenize(text)

    # Step 2: Remove Stopwords
    filtered_tokens = RemoveStopWords(tokens)

    # Step 3: Lemmatize
    final_tokens = Lemmatize(filtered_tokens)

    return final_tokens



if __name__ == "__main__":
    sample = "The striped bats are hanging on their feet for best"
    print("Original:", sample)
    print("Refined:", Refine(sample))
