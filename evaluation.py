#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 22:40:33 2018

@author: dylanz
"""

from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
import numpy as np

# load data
#group_path = '../datasets/HepTh/group.txt'
#group_path = '../datasets/zhihu/group.txt'
#group_path = '../datasets/cora/group.txt'
group_path = './datasets/cora/group.txt'
emb_path = 'embed_cora_cane.txt'


def evaluate(train_vec, test_vec, train_y, test_y, classifierStr='SVM', normalize=0):

    if classifierStr == 'KNN':
        print('Training NN classifier...')
        classifier = KNeighborsClassifier(n_neighbors=1)
    else:
        print('Training SVM classifier...')

        classifier = LinearSVC()

    if(normalize == 1):
        print('Normalize data')
        allvec = list(train_vec)
        allvec.extend(test_vec)
        allvec_normalized = preprocessing.normalize(allvec, norm='l2', axis=1)
        train_vec = allvec_normalized[0:len(train_y)]
        test_vec = allvec_normalized[len(train_y):]


    classifier.fit(train_vec, train_y)
    y_pred = classifier.predict(test_vec)
    cm = confusion_matrix(test_y, y_pred)

    #print(cm)
    acc = accuracy_score(test_y, y_pred)
    #print(acc)

    #macro_f1 = f1_score(test_y, y_pred,pos_label=None, average='macro')
    #micro_f1 = f1_score(test_y, y_pred,pos_label=None, average='micro')

    macro_f1 = f1_score(test_y, y_pred,pos_label=None, average='macro')
    micro_f1 = f1_score(test_y, y_pred,pos_label=None, average='micro')

    per = len(train_y) * 1.0 /(len(test_y)+len(train_y))
#    print('Classification method:'+classifierStr+'(train, test, Training_Percent): (%d, %d, %f)' % (len(train_y),len(test_y), per ))
    print('Classification Accuracy=%f, macro_f1=%f, micro_f1=%f' % (acc, macro_f1, micro_f1))
    #print(metrics.classification_report(test_y, y_pred))

    return acc, macro_f1, micro_f1



def eval_emb(data_dir):
    train_size = 0.15
    random_state = 418
    
    # X = {}
    # f = open(emb_path, 'rb')
    # for i, j in enumerate(f):
    #     if j != '\n':
    #         X[i] = map(float, j.strip().split(' '))
    X = np.load(data_dir)
            
    labels = {}        
    f = open(group_path, 'rb')
    for i, j in enumerate(f):
        if j != '\n':
            labels[i] = map(int, j.strip().split(' '))
            
    X_ = []
    labels_ = []        
    for i in labels.keys():
        X_.append(np.array(X[i,:]))
        labels_.append(np.array(labels[i]))
        
    train_vec, test_vec = train_test_split(X_, train_size=train_size, random_state=random_state)
    train_y, test_y = train_test_split(labels_, train_size=train_size, random_state=random_state)
    
    acc, macro_f1, micro_f1 = evaluate(train_vec, test_vec, train_y, test_y, 'SVM', 0)
    return micro_f1
    


if __name__ == "__main__":
    eval_emb('./save_dir/node_att.npy')
