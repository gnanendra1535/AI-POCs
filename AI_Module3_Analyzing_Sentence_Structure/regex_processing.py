# regex_processing.py

import re
import string


# a. Remove all punctuations using regex

def TextAfterRemovingPunctuations(text):
    
    return re.sub(rf"[{re.escape(string.punctuation)}]", "", text)



# b. Remove all digits from text

def TextAfterRemovingDigits(text):
   
    return re.sub(r"\d+", "", text)



# c. All words beginning with Capital letter

def AllCapitalizedWordsFromText(text):
   
    return re.findall(r"\b[A-Z][a-zA-Z]*\b", text)



# d. Extract all emails from text

def AllEmailsFromText(text):
    
    pattern = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[A-Za-z]{2,}"
    return re.findall(pattern, text)



# Run functions on FIFAWorldCup2018.txt

def main():
    file_path = "FIFAWorldCup2018.txt"   # keep file in same folder

    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    print("\n--- Text After Removing Punctuations ---")
    print(TextAfterRemovingPunctuations(text)[:500], "...")  # show first 500 chars

    print("\n--- Text After Removing Digits ---")
    print(TextAfterRemovingDigits(text)[:500], "...")

    print("\n--- All Capitalized Words ---")
    print(AllCapitalizedWordsFromText(text))

    print("\n--- All Emails Found ---")
    print(AllEmailsFromText(text))


if __name__ == "__main__":
    main()
