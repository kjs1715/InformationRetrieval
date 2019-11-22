import sys
import os
import re
import nltk
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

def lemmatization():
    english_punctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%']

    # read file
    f = open("/Users/kim/Desktop/text2.txt", 'r', encoding='utf-8')
    words = f.read()
    f.close()
    words = list(filter(None, re.split(r"[\n|\s|,|!|.|\"|)|(]", words)))
    temp_words = ''
    for w in words:
        temp_words += w + ' '
    words = nltk.pos_tag(nltk.word_tokenize(temp_words))
    words = [word for word in words if word not in english_punctuations]
    word_dict = {}
    print(words)
    lemmatize_all(words)


def lemmatize_all(words):
    lm = WordNetLemmatizer()
    for word, tag in words:
        if tag.startswith('NN'):
            w = lm.lemmatize(str(word), pos='n')
        elif tag.startswith('VB'):
            w = lm.lemmatize(str(word), pos='v')
        elif tag.startswith('JJ'):
            w = lm.lemmatize(str(word), pos='a')
        elif tag.startswith('R'):
            w = lm.lemmatize(str(word), pos='r')
        else:
            w = word
        if w is not word:
            print(word, " : ", w) 

def sortbyvalue(dict):
    items = dict.items()
    backitems = [[v[1], v[0]] for v in items]
    backitems.sort()
    return [[backitems[i][1], backitems[i][0]] for i in range(0, len(backitems))]


if __name__ == '__main__':
    lemmatization()