import sys
import os
import time 
import matplotlib.pyplot as plt
import numpy as np
import numba
import gensim
from gensim.models import Word2Vec
from pathlib import Path

''' 
    2019/11/22
    Using gensim.word2vec to train a model for RMRB corpus

'''


# global
fileName = '/Users/kim/Desktop/rmrb/rmrb'
punctuation = "！ ？ ｡ ＂ ＃ ＄ ％ ＆ ＇ （ ） ＊ ＋ ， － ／ ： ； ＜ ＝ ＞ ＠ ［ ＼ ］ ＾ ＿ ｀ ｛ ｜ ｝ ～ ｟ ｠ ｢ ｣ 、 ，〃 》 「 」 『 』 【 】 〔 〕 〖 〗 〘 〙 〚 〛 〜 〝 〞 〟 ； 〰 〾 〿 – — ‘ ’ ‛ “ ” „ ‟ … ‧ ﹏ . "
punctuation = punctuation.split(' ')

'''
    First I need to load corpus list (sentences) for line_count = 20000,
    in order to compare with performance with lsi analysis.
    
    :return 
        sentences : [[]]
'''
def get_corpus():
    sentence = []
    sentences = []
    line_count = 0
    with open(fileName + '2_10.txt') as f:
        line = f.readline()
        while line:
            text = line.strip('\n').split()
            for term in text:
                sentence.append(term.split('\\')[0])
            sentences.append(sentence)
            sentence = []
            line_count += 1

            if line_count == 20000:
                break
            line = f.readline()

    for sentence in sentences:
         for term in sentence:
             if term in punctuation:
                sentence.remove(term)
    
    return sentences


'''
    Training model with corpus(sentences)
'''
def model_train(sentences):
    model = Word2Vec(sentences, sg=1, size=300, window=5)
    model.save('us.model')
    return model

if __name__ == '__main__':
    sentences = get_corpus()
    try:
        file = Path('corpus.model')
        if file.is_file():
            print('Model was already trained...loading model')
            model = Word2Vec.load('corpus.model')
        else:
            model = model_train(sentences)
            print('Model trained...')
    except Exception as e:
        print(e)

    print(model['绿化'])
    print(model.similarity('中国','北京'))
    print(model.similarity('中国','国外'))
    print(model.most_similar(positive=['经济','北京'], negative=['天津']))
    