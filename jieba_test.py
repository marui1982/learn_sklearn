# coding:utf-8
import jieba
import jieba.posseg as pseg
import os
import sys
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

def fenci(line):
    words=pseg.cut(line)
    return ' '.join([x.word for x in words])

def get_fenci_result(corpus):
    return [fenci(line) for line in corpus_new]

def print_countvector(word_countvector,feature_names):
    res_matrix=word_countvector.toarray()
    for i in range(len(res_matrix)):
        print corpus[i]
        for j in  range(len(feature_names)):
            print feature_names[j],res_matrix[i][j]

def print_csr_matrix(word_countvector):
    #para: 
    #   word_countvector:csr_matrix

    print type(word_countvector),type(word_countvector.data) # toarray()
    #print 'nnz:',word_countvector.nnz
    #print 'has_sorted_indices:',word_countvector.has_sorted_indices
    #print 'indices:',word_countvector.indices
    #print 'indptr:',word_countvector.indptr
    pass 

if __name__ == "__main__":
    corpus_new=[
        "我来到北京清华大学",        #第一类文本切词后的结果，词之间以空格隔开
        "他来到了网易杭研大厦",    #第二类文本的切词结果
        "小明硕士毕业与中国科学院",#第三类文本的切词结果
        "我爱北京天安门"             #第四类文本的切词结果
    ]

    corpus=get_fenci_result(corpus_new)

    vectorizer=CountVectorizer()#该类会将文本中的词语转换为词频矩阵，矩阵元素a[i][j] 表示j词在i类文本下的词频
    #自动去除停用词：你、我、他、了、的等

    word_countvector=vectorizer.fit_transform(corpus)

    #print_csr_matrix(word_countvector)
    #print_countvector(word_countvector,vectorizer.get_feature_names())

    tfidftransformer=TfidfTransformer()#该类会统计每个词语的tf-idf权值

    tfidf=tfidftransformer.fit_transform(vectorizer.fit_transform(corpus))#第一个fit_transform是计算tf-idf，第二个fit_transform是将文本转为词频矩阵
    word=vectorizer.get_feature_names()#获取词袋模型中的所有词语
    weight=tfidf.toarray()#将tf-idf矩阵抽取出来，元素a[i][j]表示j词在i类文本中的tf-idf权重

    for i in range(len(weight)):#打印每类文本的tf-idf词语权重，第一个for遍历所有文本，第二个for便利某一类文本下的词语权重
        print u"-------这里输出第",i,u"类文本的词语tf-idf权重------"
        for j in range(len(word)):
            print word[j],weight[i][j]
