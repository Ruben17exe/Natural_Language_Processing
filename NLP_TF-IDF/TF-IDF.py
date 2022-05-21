import numpy as np


documents = ["the the universe has very many stars",
             "the galaxy contains many stars",
             "the cold breeze of winter it very cold outside"]

########################################################################################################################
bag_of_words = {}
for index, sentence in enumerate(documents):
    tokenizedWords = sentence.split(" ")
    bag_of_words[index] = [(word, tokenizedWords.count(word)) for word in tokenizedWords]

for i in range(0, len(documents)):
    no_duplicates = []
    for word in bag_of_words[i]:
        if word not in no_duplicates:
            no_duplicates.append(word)
    bag_of_words[i] = no_duplicates

len_sentence = []
for index in range(0, len(documents)):
    sentence = bag_of_words[index]
    count_words = 0
    for bow_word in sentence:
        word, num = bow_word
        count_words += num
    sentence.append(("length_sentence", count_words))

TF_per_sentence = {}
for index in range(0, len(documents)):
    length, num_len = bag_of_words[index][-1]
    sentence = bag_of_words[index]
    sentence.pop()
    frequency_list = []
    for words in sentence:
        word, num_word = words
        frequency_list.append((word, round(num_word/num_len, 4)))
    TF_per_sentence[index] = frequency_list

########################################################################################################################
########################################################################################################################
inverse_document_frequency_per_word = {}
IDF_per_sentence = {}
for sentence in documents:
    for word in sentence.split():
        if word not in inverse_document_frequency_per_word:
            inverse_document_frequency_per_word[word] = 0

for index in range(len(documents)):
    sentences = TF_per_sentence[index]
    words_per_sentence = [words for words, values in sentences]
    for term in inverse_document_frequency_per_word:
        if term in words_per_sentence:
            inverse_document_frequency_per_word[term] += 1

for key, values in inverse_document_frequency_per_word.items():
    inverse_document_frequency_per_word[key] = round(np.log(len(documents) / values), 4)  # np.log10

for index in range(len(documents)):
    sentence = TF_per_sentence[index]
    IDF_per_sentence[index] = [(word, inverse_document_frequency_per_word[word]) for word, value in sentence]

########################################################################################################################
########################################################################################################################
TF_IDF_per_sentence = {}
for index in range(len(documents)):
    TF_sentence = TF_per_sentence[index]
    IDF_sentence = IDF_per_sentence[index]
    TF_IDF_per_sentence[index] = [(TF_per_sentence[index][i][0],
                                  round((TF_per_sentence[index][i][1] * IDF_per_sentence[index][i][1]), 4))
                                  for i in range(len(IDF_sentence))]

print(TF_per_sentence)
print(IDF_per_sentence)
print(TF_IDF_per_sentence)

########################################################################################################################
# Validate with NLTK library ----- Open text.py inside the NLTK library
# Go to line 716 where you find the method tf(self, term, text) that has to be modified
# Uncomment lines 719 and 720 and comment out line 721
# This way, the source code use the words instead the characters
#   Line 719: newText = text.split(" ")
#   Line 720: return text.count(term) / len(newText)

# from nltk.text import TextCollection
# documents_nltk = TextCollection(documents)
# print(documents_nltk.tf_idf("contains", documents[1]))
