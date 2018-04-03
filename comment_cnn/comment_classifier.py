#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# comment_classifier.py
#
# Vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
# Python source code - replace this with a description
# of the code and write the code below this text
#

import numpy as np
from collections import Counter
import jieba
import xgboost as xgb
import pandas as pd
import tensorflow as tf
import pickle
import random
from matplotlib import pyplot as plt
import keras
from keras.models import Sequential  
from keras.layers import Dense,Flatten,Dropout  
from keras.layers.convolutional import Conv2D,MaxPooling2D  
from keras.utils.np_utils import to_categorical 
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB 
"""
'I'm super man'
tokenize:
['I', ''m', 'super', 'man']
"""

"""
词形还原(lemmatizer)，即把一个任何形式的英语单词还原到一般形式，
与词根还原不同(stemmer)，后者是抽取一个单词的词根。
"""



pos_file = 'pos.txt'
neg_file = 'neg.txt'
class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch':[], 'epoch':[]}
        self.accuracy = {'batch':[], 'epoch':[]}
        self.val_loss = {'batch':[], 'epoch':[]}
        self.val_acc = {'batch':[], 'epoch':[]}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('acc'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_acc'))

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc'))

    def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))
        plt.figure()
        # acc
        plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')
        # loss
        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
        if loss_type == 'epoch':
            # val_acc
            plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
            # val_loss
            plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('acc-loss')
        plt.legend(loc="upper right")
        plt.show()
def my_CNN():
    model = Sequential()
    model.add(Conv2D(32,(1,5),strides=(1,1),input_shape=(1,len(lex),1),padding='valid',activation='relu',kernel_initializer='uniform'))
    model.add(MaxPooling2D(pool_size=(1,2),strides=(2,2)))
    model.add(Conv2D(64,(1,4),padding='same',activation='relu'))
    model.add(MaxPooling2D(pool_size=(1,2),strides=(2,2)))
    model.add(Conv2D(128,(1,3),padding='same',activation='relu'))
    model.add(MaxPooling2D(pool_size=(1,2),strides=(2,2)))
    model.add(Conv2D(256,(1,3),padding='same',activation='relu'))
    model.add(MaxPooling2D(pool_size=(1,2),strides=(2,2)))
    model.add(Conv2D(512,(1,4),padding='same',activation='relu'))
    model.add(Flatten()) 
    model.add(Dropout(0.4))  
    model.add(Dense(2,activation='softmax'))  
    model.compile(loss='categorical_crossentropy',optimizer='rmsprop',metrics=['accuracy'])  
    model.summary()  
    return model
# 创建词汇表
def create_lexicon(pos_file, neg_file):
    lex = []
    # 读取文件
    def process_file(txtfile):
        with open(txtfile, 'r',encoding='utf8') as f:
            lex = []
            lines = f.readlines()
            #print(lines)
            for line in lines:
                words = jieba.lcut(line[5:])
                lex += words
            return lex

    lex += process_file(pos_file)
    lex + process_file(neg_file)
    #print(len(lex))
#    lemmatizer = WordNetLemmatizer()
#    lex = [lemmatizer.lemmatize(word) for word in lex] # 词形还原(cats -> cat)

    word_count = Counter(lex)
    #print(word_count)
    lex = []
    for word in word_count:
        if word_count[word] < 2000 and word_count[word] > 20:
            lex.append(word)
    return lex

#lex 里保存了文本中出现过的单词




def normalize_dataset(lex):
    dataset = []
    # lex:词汇表；review:评论；clf:评论对应的分类，
    # [0,1]代表负面评论 [1,0]代表正面评论
    def string_to_vector(lex, review, clf):
        words = jieba.lcut(review[5:])
        features = np.zeros(len(lex))
        for word in words:
            if word in lex:
                features[lex.index(word)] = 1
        return [features, clf]

    with open(pos_file, 'r', encoding='utf8') as f:
        lines = f.readlines()
        for line in lines:
            one_sample = string_to_vector(lex, line, [0])
            dataset.append(one_sample)
    with open(neg_file, 'r', encoding='utf8') as f:
        lines = f.readlines()
        for line in lines:
            one_sample = string_to_vector(lex, line, [1])
            dataset.append(one_sample)


    return dataset

def data_2_array(dataset):
    data_x = []
    data_y = []
    for i in range(len(dataset)):
        data_x.append(dataset[i][0])
        data_y.append(dataset[i][1])
    return np.array(data_x),np.array(data_y)
lex = create_lexicon(pos_file, neg_file)
dataset = normalize_dataset(lex)
random.shuffle(dataset)


#把整理好的数据保存到文件，方便使用。到此完成了数据的整理工作
with open('save.pickle', 'wb') as f:
	pickle.dump(dataset, f)
#with open('save.pickle', 'rb') as f:
#    dataset = pickle.load(f)
# 取样本的10%作为测试数据

test_size = int(len(dataset) * 0.1)
data_x, data_y = data_2_array(dataset)
train_x = data_x[:-test_size]
train_y = data_y[:-test_size]
test_x = data_x[-test_size:]
test_y = data_y[-test_size:]
cnn_train_x = np.expand_dims(train_x, axis=1)
cnn_train_x = np.expand_dims(cnn_train_x, axis=3)
cnn_train_y = to_categorical(train_y)
cnn_test_x = np.expand_dims(test_x ,axis=1)
cnn_test_x = np.expand_dims(cnn_test_x, axis=3)
cnn_test_y = to_categorical(test_y)

model = my_CNN()
history = LossHistory()
model.fit(cnn_train_x,cnn_train_y,epochs=50,batch_size = 16, validation_data=(cnn_test_x, cnn_test_y),callbacks=[history])
cnn_pre = model.predict(cnn_test_x,batch_size=16)[:,1]
cnn_pred = (cnn_pre >= 0.5)*1
print ('CNN AUC: ',str(metrics.roc_auc_score(test_y,cnn_pre)))
print ('CNN ACC: ',str(metrics.accuracy_score(test_y,cnn_pred)))
print ('CNN Recall: ',str(metrics.recall_score(test_y,cnn_pred)))
print ('CNN F1-score: ',str(metrics.f1_score(test_y,cnn_pred)))
print ('CNN Precesion: ',str(metrics.precision_score(test_y,cnn_pred)))
metrics.confusion_matrix(test_y,cnn_pred)
fpr, tpr, thresholds = metrics.roc_curve(test_y, cnn_pre)  
roc_auc = metrics.auc(fpr,tpr)
plt.figure(3)
plt.plot(fpr, tpr, lw=1)
plt.title('ROC Curve(CNN)')
history.loss_plot('epoch')


gnb = GaussianNB()
gnb.fit(train_x, train_y.reshape((-1)))
Bayes_pre = gnb.predict_proba(test_x)[:,1]
Bayes_pred = (Bayes_pre >= 0.5)*1
print ('Bayes AUC: ',str(metrics.roc_auc_score(test_y,Bayes_pre)))
print ('Bayes ACC: ',str(metrics.accuracy_score(test_y,Bayes_pred)))
print ('Bayes Recall: ',str(metrics.recall_score(test_y,Bayes_pred)))
print ('Bayes F1-score: ',str(metrics.f1_score(test_y,Bayes_pred)))
print ('Bayes Precesion: ',str(metrics.precision_score(test_y,Bayes_pred)))
metrics.confusion_matrix(test_y,Bayes_pred)
fpr, tpr, thresholds = metrics.roc_curve(test_y, Bayes_pre)  
roc_auc = metrics.auc(fpr,tpr)
plt.figure(3)
plt.plot(fpr, tpr, lw=1)
plt.title('ROC Curve(Bayes)')
