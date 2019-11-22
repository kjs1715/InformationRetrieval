import sys
import os
import time 
import matplotlib.pyplot as plt
import numpy as np
import numba

# global
fileName = '/Users/kim/Desktop/rmrb/rmrb'
DOC_INTERVAL = 30

punctuation = "！ ？ ｡ ＂ ＃ ＄ ％ ＆ ＇ （ ） ＊ ＋ ， － ／ ： ； ＜ ＝ ＞ ＠ ［ ＼ ］ ＾ ＿ ｀ ｛ ｜ ｝ ～ ｟ ｠ ｢ ｣ 、 ，〃 》 「 」 『 』 【 】 〔 〕 〖 〗 〘 〙 〚 〛 〜 〝 〞 〟 ； 〰 〾 〿 – — ‘ ’ ‛ “ ” „ ‟ … ‧ ﹏ . "
punctuation = punctuation.split(' ')

''' Count terms and seperate into several docs for term-doc-matrix
    temp = []

    :params
        fileName, fileNum
    :return 
        term_list
'''
def count_terms(fileName, fileNum):
    line_count = 0
    term_count = 0
    term_list = []
    temp = ''
    doc_list = []
    new_term_list = []


    for i in fileNum:
        with open(fileName + i + '.txt', 'r') as f:
            line = f.readline()
            while line:
                text = line.strip('\n').split(' ')
                for x in text:
                    term_list.append(x.split('\\')[0])
                temp += line
            
                line_count += 1
                term_count += len(text)

                if line_count % DOC_INTERVAL == 0:
                    doc_list.append(temp)
                    temp = ''
                
                if line_count == 20000:
                    break
                line = f.readline()
        f.close()
        new_term_list = sortListByIndex(term_list, new_term_list)
        for punc in punctuation:
            try:
                new_term_list.remove(punc)
            except:
                pass
        return new_term_list, doc_list

def sortListByIndex(raw, new):
    new = list(set(raw))
    new.sort(key=raw.index)
    return new

    delete_key = []

# statistics for words
def get_dic(term_count, term_list, doc_list):
    print("getdic")
    doc_dic = {}
    doc_num = 0
    temp_doc = ''
    doc_term_list = []
    delete_key = []
    punctuation = "！ ？ ｡ ＂ ＃ ＄ ％ ＆ ＇ （ ） ＊ ＋ ， － ／ ： ； ＜ ＝ ＞ ＠ ［ ＼ ］ ＾ ＿ ｀ ｛ ｜ ｝ ～ ｟ ｠ ｢ ｣ 、 ，〃 》 「 」 『 』 【 】 〔 〕 〖 〗 〘 〙 〚 〛 〜 〝 〞 〟 ； 〰 〾 〿 – — ‘ ’ ‛ “ ” „ ‟ … ‧ ﹏ . "
    punctuation = punctuation.split(' ')

    print('doc count' + str(len(doc_list)))

    for doc in doc_list:
        terms = doc.split(' ')
        for term in terms:
            # delete marks like : "开发\\v"
            temp = term.split('\\')[0]
            if temp in term_list:
                if doc_dic.__contains__(temp):
                    doc_dic[temp] += 1
                else:
                    doc_dic[temp] = 1

        # delete punctuation marks
        for key in doc_dic:
            punc = key.split('\\')
            if punc[0] in punctuation:
                delete_key.append(key)

        for key in delete_key:
            doc_dic.pop(key)
        # sorterd with frequency (high to low)
        sorted_dic = sorted(doc_dic.items(), key=lambda x: x[1], reverse=True)
        doc_term_list.append(sorted_dic)
        doc_dic.clear()
        delete_key = []
        
    return doc_term_list

# convert into term_doc matrix
# term_doc_list -> [(term, count)]

def get_term_doc_matrix(term_list, term_doc_list):
    print('getMatrix')
    # need to delete ' '
    # term_list.remove('')

    doc_count = len(term_doc_list)
    term_count = len(term_list)
    
    print(term_count, doc_count)

    matrix = np.zeros((term_count, doc_count))

    doc_num = 0
    for doc in term_doc_list:
        for term in doc:
            try:
                if term_list.index(term[0]):
                    matrix[term_list.index(term[0])][doc_num] = term[1]
            except:
                pass
        doc_num += 1 

    return matrix, term_list

def plot_term_doc(T, D, term_list, n):
    print(term_list[:30])
    term_x = [x[0] for x in T][:30]
    term_y = [x[1] for x in T][:30]

    print('x-----')
    print(term_x)
    print(term_y)

    doc_x = []
    doc_y = []

    for i in range(int(D.shape[1])):
        doc_x.append(D[0][i])
        doc_y.append(D[1][i])

    doc_x = doc_x[:20]
    doc_y = doc_y[:20]

    print('y------')
    print(doc_x)
    print(doc_y)

    plt.figure(num=n, figsize=(16, 10))
    # range
    # plt.xticks(np.arange(-1, 1, 0.1))
    # plt.yticks(np.arange(-1, 1, 0.1))
    # plt.axis([-1, 1, -1, 1])
    # draw
    plt.scatter(term_x, term_y, color='red')
    plt.scatter(doc_x, doc_y, color='blue')
    for i in range(len(term_x)):
        plt.text(term_x[i], term_y[i], str(i))
    for j in range(len(doc_x)):
        plt.text(doc_x[j], doc_y[j], str(j))
    plt.xlabel('x')
    plt.ylabel('y')

def kvalue_matrix(U, sigma, VT, k):
    return (U[:,0:k]).dot(np.diag(sigma[0:k])).dot(VT[0:k,:])

def cos(v1, v2):
    # print(v1, v2)
    num = v1.reshape(1, v1.shape[1] * v1.shape[0]).dot(v2.reshape(v2.shape[1] * v2.shape[0], 1))
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)
    return 0.5 + 0.5 * (num / denom)

if __name__ == '__main__':
    term_list, doc_list = count_terms(fileName, ['2_10'])
    term_doc_list = get_dic(len(term_list), term_list, doc_list)
    matrix, term_list = get_term_doc_matrix(term_list, term_doc_list)
    U, sigma, VT = np.linalg.svd(matrix[:5000, :300])
    
    # Q1
    print('Q1')
    k_matrix = [] # k = 1, 2, 3, 4
    k_matrix.append(matrix[:5000,:300])
    for k in range(1, 5):
        k_matrix.append(kvalue_matrix(U, sigma, VT, k))
    for k in k_matrix:
        print(k.shape)
    for k in [(1, 2), (1, 3), (1, 4), (1, 5)]:
        print(cos(k_matrix[k[0]-1], k_matrix[k[1]-1]))  

    # Q2
    print('Q2')
    # T = (U[:,0:2]).dot(np.diag(sigma[0:2]))
    # D = np.diag(sigma[0:2]).dot(VT[0:2,:])
    T = U[:,0:2]
    D = VT[0:2,:]
    print('sigma',np.diag(sigma[0:2]))
    print('T', T)
    print('D', VT[0:2,:])
    # print(T, D)
    print(matrix[:30,:20])
    plot_term_doc(T, D, term_list, 1)

    plt.show()