import pandas as pd
from nltk.stem import WordNetLemmatizer

test_text = """
The bag-of-words language model is a simple-yet-powerful tool to have up your sleeve when working on natural 
language processing (NLP). The model has many, many use cases including
"""


def preprocess_text(text):
    characters = "-"
    string = [text.replace(characters[x], " ") for x in range(len(characters))]

    words = ["".join(filter(str.isalnum, word)) for word in " ".join(string).lower().split()]
    lemmatizer = WordNetLemmatizer()
    lemmatized_words = [lemmatizer.lemmatize(word, "v") for word in words]
    return lemmatized_words


def bag_of_words(words_in_text):
    bag_of_words = {}
    for word in words_in_text:
        if word in bag_of_words:
            bag_of_words[word] += 1
        else:
            bag_of_words[word] = 1

    df_bow = pd.DataFrame([[key, bag_of_words[key]] for key in bag_of_words.keys()], columns=["Word", "Count"])

    return df_bow, bag_of_words


df_bow, bag_of_words = bag_of_words(preprocess_text(test_text))
print(df_bow)
