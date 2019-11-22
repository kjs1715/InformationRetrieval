# coding=utf-8
import os
import numpy as np
import sys
import time
import seaborn
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import scipy.spatial as sp
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

DOC_INTERVAL = 50000


# statistics for words

def count_terms(fileName, fileNum):
    line_count = 0
    term_count = 0
    temp = ''
    term_list = []
    doc_list = []
    new_term_list = []

    for t in fileNum:
        # with open('0.txt', 'r') as f:
        with open(fileName + t + '.txt', 'r') as f:
            line = f.readline()
            while line:
                text = line.strip('\n').split(' ')
                for x in text:
                    term_list.append(x.split('\\')[0])
                line_count += 1
                term_count += len(text)
                temp += line
                
                # separate documents
                if line_count % DOC_INTERVAL == 0:
                    doc_list.append(temp)
                    temp = ''

                if line_count == 500000:
                    break
                line = f.readline()
        f.close()

    # delete duplicate terms

    term_set = set(term_list)
    # print(new_term_list)
    # print(line_count, term_count, len(term_set))
    return line_count, term_count, term_set, doc_list


# statistics for words
def get_dic(term_count, term_set, doc_list):
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
            if temp in term_set:
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

# get top 15 frequent terms
# :list : [(x, y),]
def get_top15_terms(term_doc_list):
    top5_each_doc_list = []
    for doc in term_doc_list:
        temp = doc[:5]
        for item in temp:
            top5_each_doc_list.append(item)
    top5_sorted = sorted(top5_each_doc_list, key=lambda x: x[1], reverse=True)
    # top5_sorted = top5_each_doc_list # now it is not top 15 anymore(random)

    return top5_sorted[:30]


def cos(v1, v2):
    # print(v1, v2)
    num = v1.reshape(1, v1.shape[1] * v1.shape[0]).dot(v2.reshape(v2.shape[1] * v2.shape[0], 1))
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)
    return 0.5 + 0.5 * (num / denom)

def cos_vec(v1, v2):
    num = 0
    for i in range(len(v1)):
        num += v1[i] * v2[i]
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)
    if denom == 0.0:
        return 0
    return 0.5 + 0.5 * (num / denom)

def kvalue_matrix(U, sigma, VT, k):
    return (U[:,0:k]).dot(np.diag(sigma[0:k])).dot(VT[0:k,:])
   
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

    doc_x = doc_x[:10]
    doc_y = doc_y[:10]

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

# plot term_term_matrix and doc_doc matrix
def plot_others(T, D, term_list, n):
    # term_term_matrix
    term_x = [x[0] for x in T][:30]
    term_y = [x[1] for x in T][:30]

    plt.figure(num=n, figsize=(16, 10))
    # plt.axis([-1, 1, -1, 1])
    # plt.xticks(np.arange(-1, 1, 0.1))
    # plt.yticks(np.arange(-1, 1, 0.1))
    plt.scatter(term_x, term_y, color='red')

    for i in range(len(term_x)):
        plt.text(term_x[i], term_y[i], str(i))
    plt.xlabel('x')
    plt.ylabel('y')

    # doc_doc_matrix
    doc_x = []
    doc_y = []

    for i in range(int(D.shape[1])):
        doc_x.append(D[0][i])
        doc_y.append(D[1][i])

    doc_x = doc_x[:10]
    doc_y = doc_y[:10]

    plt.figure(num=n+1, figsize=(16, 10))
    # plt.axis([-1, 1, -1, 1])
    # plt.xticks(np.arange(-1, 1, 0.1))
    # plt.yticks(np.arange(-1, 1, 0.1))
    plt.scatter(doc_x, doc_y, color='blue')


    for j in range(len(doc_x)):
        plt.text(doc_x[j], doc_y[j], str(j))

    plt.xlabel('x')
    plt.ylabel('y')

# convert into term_doc matrix
# term_doc_list -> [(term, count)]
def get_term_doc_matrix(term_set, term_doc_list):
    print('getMatrix')
    # need to delete ' '
    term_set.remove('')

    # term_count = len(term_set)
    doc_count = len(term_doc_list)

    # top5_terms_list = get_top15_terms(term)
    term_list = list(term_set)[:30]
    term_count = 30
    # term_count = len(term_list)
    
    print(term_count, doc_count)

    
    # two approaches: 1. with top terms 2. random (sorted(reverse=False))
    # 1.
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

# convert into term_term_ matrix
# it is different with get_term_doc_matrix, (first parameter)
def get_term_term_matrix(term_list, term_doc_list):
    term_count = 30
    doc_count = len(term_doc_list)
    
    matrix = np.zeros(term_count, doc_count)

    doc_num = 0
    for doc in term_doc_list:
        for term in doc:
            try:
                if term_list.index(term[0]):
                    matrix[term_list.index(term[0])][doc_num] = term[1]
            except:
                pass
        doc_num += 1


if __name__ == '__main__':
    fileName = '/Users/kim/Desktop/rmrb/rmrb'
    line_count, term_count, term_set, doc_list= count_terms(fileName, ['2_10'])
    term_doc_list = get_dic(term_count, term_set, doc_list)
    matrix, term_list = get_term_doc_matrix(term_set, term_doc_list)
    U, sigma, VT = np.linalg.svd(matrix)

    print('term_count', len(term_set))

    # print first 10 docs
    for i in range(10):
        print(str(i) + ': ', term_doc_list[i][:100])

    # Q1
    print('Q1')
    k_matrix = [] # k = 1, 2, 3, 4
    k_matrix.append(matrix)
    for k in range(1, 5):
        k_matrix.append(kvalue_matrix(U, sigma, VT, k))
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

    # Q3
    term_term_matrix = matrix.dot(np.transpose(matrix))
    doc_doc_matrix = np.transpose(matrix).dot(matrix)

    print(term_term_matrix[:15,:15])
    print(doc_doc_matrix[:15,:15])

    T_sigma = T.dot(np.diag(sigma[0:2]))
    D_sigma = np.diag(sigma[0:2]).dot(D)
    # print('T_sigma', T_sigma)
    # print('D_sigma', D_sigma)
    plot_others(T_sigma, D_sigma, term_list, 2)
    print('term_term_cos')

    f1 = open('cos1.txt', 'w')
    for i in range(30):
        j = i + 1
        for j in range(30):
            f1.write(str((i, j)) + ' ' + str(cos_vec(T_sigma[i], T_sigma[j])) + '\n')

    D_sigma = np.transpose(D_sigma)
    # print(D_sigma.shape)
            
    for i in range(10):
        j = i + 1
        for j in range(10):
            f1.write(str((i, j)) + ' ' + str(cos_vec(D_sigma[i], D_sigma[j])) + '\n')

    f1.close()

    # Q4
    # d_d needs to be transformed once for pca
    T1 = U[:,0:3]
    D1 = VT[0:3,:]
    sigma1 = sigma[0:3]
    pca = PCA(n_components=2)
    T1_pca = pca.fit_transform(T1)
    D1_pca = pca.fit_transform(np.transpose(D1))
    D1_pca = np.transpose(D1_pca)
    T1_pca_sigma = pca.fit_transform(T1.dot(np.diag(sigma1)))
    D1_sigma = np.diag(sigma1).dot(D1)
    D1_pca_sigma = pca.fit_transform(np.transpose(D1_sigma))
    D1_pca_sigma = np.transpose(D1_pca_sigma)
    plot_term_doc(T1_pca, D1_pca, term_list, 4)

    plot_others(T1_pca_sigma, D1_pca_sigma, term_list, 5)

    # print("--------------")
    # print('T1', T1)
    # print('D1', D1)

    # print('D1_sigma', D1_sigma)

    # print('T1_pca', T1_pca_sigma)
    # print('D1_pca', D1_pca_sigma)

    f2 = open('cos2.txt', 'w')
    for i in range(30):
        j = i + 1
        for j in range(30):
            f2.write(str((i, j)) + ' ' + str(cos_vec(T1_pca_sigma[i], T1_pca_sigma[j])) + '\n')

    D1_pca = np.transpose(D1_pca_sigma)
    print(D1_pca.shape)
            
    for i in range(10):
        j = i + 1
        for j in range(10):
            f2.write(str((i, j)) + ' ' + str(cos_vec(D1_pca[i], D1_pca[j])) + '\n')

    f2.close()
    print('k = 2')
    print(np.diag(sigma[0:2]))
    print('k = 3')
    print(np.diag(sigma[0:3]))




    plt.show()





    



    