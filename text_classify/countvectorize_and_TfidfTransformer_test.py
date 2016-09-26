# -*- coding: utf-8 -*-

import numpy
from sklearn import metrics
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn import linear_model
from sklearn.datasets import load_iris
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn import cross_validation
from sklearn import preprocessing
#import iris_data

def load_data():
    iris = load_iris()
    x, y = iris.data, iris.target
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)
    return x_train,y_train,x_test,y_test

def train_clf3(train_data, train_tags):
    clf = LinearSVC(C=1100.0)#default with 'rbf'  
    clf.fit(train_data,train_tags)
    return clf

def train_clf(train_data, train_tags):
    clf = MultinomialNB(alpha=0.01)
    print numpy.asarray(train_tags)
    clf.fit(train_data, numpy.asarray(train_tags))
    return clf

def evaluate(actual, pred):
    m_precision = metrics.precision_score(actual, pred)
    m_recall = metrics.recall_score(actual, pred)
    print 'precision:{0:.3f}'.format(m_precision)
    print 'recall:{0:0.3f}'.format(m_recall)
    print 'f1-score:{0:.8f}'.format(metrics.f1_score(actual,pred));

x_train,y_train,x_test,y_test = load_data()

clf = train_clf(x_train, y_train)

pred = clf.predict(x_test)

evaluate(numpy.asarray(y_test), pred)
