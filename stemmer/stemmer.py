import sys
import os
import re
import nltk
from nltk.stem import PorterStemmer
nltk.download('punkt')

def stemming():
    ps = PorterStemmer()
    english_punctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%', '\"', '\"', '`', '\'']

    # read file
    f = open("/Users/kim/Desktop/text2.txt", 'r', encoding='utf-8')
    words = f.read()
    f.close()
    # words = list(filter(None, re.split(r"[\n|\s|,|!|.|\"|)|(]", words)))
    words = nltk.word_tokenize(words)
    words = [word for word in words if word not in english_punctuations]
    word_dict = {}
    print(words)
    for w in words:
        word = ps.stem(w)
        if word_dict.get(word) is not None:
            word_dict[word] += 1
        else:
            word_dict[word] = 1

    word_dict = sortbyvalue(word_dict)
    # print(word_dict)
    for i in range(0, len(word_dict)):
        print(word_dict[i][0], " : ", word_dict[i][1])


def sortbyvalue(dict):
    items = dict.items()
    backitems = [[v[1], v[0]] for v in items]
    backitems.sort()
    return [[backitems[i][1], backitems[i][0]] for i in range(0, len(backitems))]


if __name__ == '__main__':
    stemming();