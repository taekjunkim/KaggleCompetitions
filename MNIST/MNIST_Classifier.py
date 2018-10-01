#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 12:37:45 2018
MNIST_Classifier

@author: taekjunkim
"""
import pandas as pd;
import json;
import gzip;

import numpy as np;
import matplotlib.pyplot as plt;

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout
from keras.layers import Conv2D, MaxPooling2D
from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.model_selection import cross_val_score

def make_gz():
    data = pd.read_csv('./csvFiles/train.csv');
    train = dict();
    train['label'] = data['label'].values.tolist();
    train['img'] = data.iloc[:,1:].values.tolist();

    fout = gzip.GzipFile('./csvFiles/train.json.gz','wb');
    fout.write(json.dumps(train).encode('utf-8'));
    fout.close();

    data = pd.read_csv('./csvFiles/test.csv');
    test = dict();
    test['img'] = data.values.tolist();
    
    fout = gzip.GzipFile('./csvFiles/test.json.gz','wb');
    fout.write(json.dumps(test).encode('utf-8'));
    fout.close();

def load():
    fin = gzip.GzipFile('./csvFiles/train.json.gz','rb');
    train = json.loads(fin.read().decode('utf-8'));
    fin.close();
    
    fin = gzip.GzipFile('./csvFiles/test.json.gz','rb');
    test = json.loads(fin.read().decode('utf-8'));
    fin.close();    
    return train,test

def createCNN():
    model = Sequential();
    model.add(Conv2D(32, (5,5), activation='relu',
                     data_format='channels_last',input_shape=(28,28,1)))
    model.add(MaxPooling2D(pool_size=(2,2)))    
    model.add(Conv2D(32, (10,10), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))        
    model.add(Flatten())
    model.add(Dense(200, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model;

def main():
    [train,test] = load();
    XTrain = np.array(train['img']);
    XTrain = XTrain.reshape([-1,28,28,1]);
    XTrain = XTrain/255;
    YTrain0 = np.array(train['label']);
    
    YTrain = np.zeros([42000,10]);
    for i in range(42000):
        YTrain[i,YTrain0[i]] = 1;
    
    XTest = np.array(test['img']);
    XTest = XTest.reshape([-1,28,28,1]);
    XTest = XTest/255;
    
    model = createCNN();
    model.fit(XTrain,YTrain,epochs=100,batch_size=32);
    
    Submitted = pd.DataFrame();
    Submitted['ImageId'] = range(1,28001);
    Submitted['Label'] = model.predict_classes(XTest).astype(int);
    Submitted.to_csv("MNIS_Submission.csv", index=False)
    
