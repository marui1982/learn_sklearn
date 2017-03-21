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

for line in sys.stdin:
    #l=line.strip().split('\t')
    info=line.strip()
    #if len(l)!=3:continue
    #res=l[1]
    #info=l[2]
    r=fenci(info)
    print r.encode('utf8')
    #print fenci(i.strip().decode('utf8'))
    #print type(i.strip().decode('utf8')) #i.strip()
    #print [i.decode('utf8')]
    
