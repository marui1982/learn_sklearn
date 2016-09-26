# -*- coding: utf-8 -*-
"""
@author: jiangfuqiang
"""

import os
import jieba
import jieba.posseg as pseg
import sys
import re
import time
import string
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
reload(sys)

sys.setdefaultencoding('utf-8')

def getFileList(path):
  filelist = []
  files = os.listdir(path)
  for f in files:
    if f[0] == '.':
      pass
    else:
      filelist.append(f)
  return filelist,path

def fenci(filename,path,segPath):
  f = open(path +"/" + filename,'r+')
  file_list = f.read()
  f.close()

   #保存粉刺结果的目录

  if not os.path.exists(segPath):
    os.mkdir(segPath)

  #对文档进行分词处理
  seg_list = jieba.cut(file_list,cut_all=True)
  #对空格，换行符进行处理
  result = []
  for seg in seg_list:
    seg = ''.join(seg.split())
    reg = 'w+'
    r = re.search(reg,seg)
    if seg != '' and seg != '=' and 
            seg != '[' and seg != ']' and seg != '(' and seg != ')' and not r:
      result.append(seg)

  #将分词后的结果用空格隔开，保存至本地
  f = open(segPath+"/"+filename+"-seg.txt","w+")
  f.write(' '.join(result))
  f.close()

#读取已经分词好的文档，进行TF-IDF计算
def Tfidf(filelist,sFilePath,path):
  corpus = []
  for ff in filelist:
    fname = path + ff
    f = open(fname+"-seg.txt",'r+')
    content = f.read()
    f.close()
    corpus.append(content)

  vectorizer = CountVectorizer()
  transformer = TfidfTransformer()
  tfidf = transformer.fit_transform(vectorizer.fit_transform(corpus))
  word = vectorizer.get_feature_names()  #所有文本的关键字
  weight = tfidf.toarray()


  if not os.path.exists(sFilePath):
    os.mkdir(sFilePath)

  for i in range(len(weight)):
    print u'----------writing all the tf-idf in the ',i,u'file into ', sFilePath+'/' +string.zfill(i,5)+".txt"
    f = open(sFilePath+"/"+string.zfill(i,5)+".txt",'w+')
    for j in range(len(word)):
      f.write(word[j] + "  " + str(weight[i][j]) + "
")
    f.close()


if __name__ == "__main__":
  #保存tf-idf的计算结果目录
  sFilePath = "data/tfidffile"+str(time.time())

  #保存分词的目录
  segPath = 'data/segfile'

  (allfile,path) = getFileList('data/tt_task')
  for ff in allfile:
    print "Using jieba on " + ff
    fenci(ff,path,segPath)

  Tfidf(allfile,sFilePath,segPath)
  #对整个文档进行排序
  #os.system("sort -nrk 2 " + sFilePath+"/*.txt >" + sFilePath + "/sorted.txt")
