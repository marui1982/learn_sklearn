# -*- coding: utf-8 -*-
from sklearn.feature_extraction import DictVectorizer
import numpy
import json

def test_DictVectorizer():
    #example is also from http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.DictVectorizer.html
    measurements = [
        {'city': 'Dubai', 'temperature': 33.},
        {'city': 'London', 'temperature': 12.},
        {'city': 'San Fransisco', 'temperature': 18.},
        {'city': 'San Fransisco', 'temperature': 17.},
        {'city': 'Dubai=', 'temperature': 33.},
    ]
    measurements2 = [
        {'city': 'xxx', 'temperature': 33.}
    ]
    vec = DictVectorizer(dtype=numpy.float32,separator='=')
    print vec.fit(measurements)             #fit用于训练,如果需要存储，则可以使用python串行化
    print vec.transform(measurements).toarray() 
    #print vec.fit_transform(measurements).toarray() #fit_transform=fit+transform
    print vec.transform(measurements2).toarray()    #transform用于测试
    print vec.get_params()

    print vec.get_feature_names()
    vec.restrict([1,0,1,0,1])       #特征筛选
    print vec.get_feature_names()
    print vec.transform(measurements2).toarray()
    #说明：
    #   1) dict中的value值如果是字符串，会做onehot转换

def test_FeatureHasher():
    from sklearn.feature_extraction import FeatureHasher
    #h = FeatureHasher(n_features=10)
    h = FeatureHasher(6)
    D = [{'dog': 1, 'cat':2, 'elephant':4},{'dog': 2, 'run': 5}]
    f = h.transform(D)
    print f.toarray()

def test_text_feature_extraction():
    def Common_Vectorizer_usage():
        from sklearn.feature_extraction.text import CountVectorizer
        vectorizer = CountVectorizer(min_df=1)
        corpus = [
            'This is the first document.',
            'This is the second second document.',
            'And the third one.',
            'Is this the first document?',
        ]

        analyze = vectorizer.build_analyzer()
        print analyze("This is a text document to analyze.")
        print analyze("This is a text document to analyze.") == ['this', 'is', 'text', 'document', 'to', 'analyze']
        
        X=vectorizer.fit_transform(corpus)
        print vectorizer.get_feature_names()
        print vectorizer.vocabulary_    #.get('document')
        print vectorizer.transform(['Something completely new.']).toarray()
        print list(X) 
        
        #bigram========================================================
        bigram_vectorizer = CountVectorizer(ngram_range=(1, 2),token_pattern=r'\b\w+\b', min_df=1)
        analyze = bigram_vectorizer.build_analyzer()
        print analyze('Bi-grams are cool!')
        X_2 = bigram_vectorizer.fit_transform(corpus).toarray()
        print X_2

        feature_index = bigram_vectorizer.vocabulary_.get('is this')
        print X_2[:, feature_index] 
        
        #marui test
        print '\n\nmarui test====================='
        def t_preprocessor(s):
            return ','.join([x.lower() for x in s.split(' ')])

        stop_words1=['is','a','this']           #is ok: frozenset(['a', 'this', 'is'])
        stop_words2={'is':0,'a':1,'this':2}     #is ok: convert to frozenset(['a', 'this', 'is'])    
            
        cv = CountVectorizer(preprocessor=t_preprocessor,stop_words=stop_words2)
        params=cv.get_params()
        print 'get_params()',type(params),'---------------'
        for k in params:
            print k,'\t',params[k]
        print 'get_params end--------------'
        print '\nget_stop_words=',cv.get_stop_words()
        
        cv.fit(corpus)
        print cv.get_feature_names()
        print cv.transform(corpus).toarray()
        print '\n测试preprocesser, result:\t',cv.build_preprocessor()('this is a document')
        print '\n测试tokenizer,result',cv.build_tokenizer()('this is a document')
        print '\n测试tokenizer2,result',cv.build_tokenizer()('th-is is a document')
        print '\n测试tokenizer2,result',cv.build_tokenizer()('th_is is a document')
        print '\n测试tokenizer2,result',cv.build_tokenizer()('th&is is a document')

        """
        sklearn.feature_extraction.text.CountVectorizer
        function:
            get_params([deep])      Get parameters for this estimator.
            set_params(**params)    Set the parameters of this estimator.
            get_stop_words()        Build or fetch the effective stop words list
            

            decode(doc)             Decode the input into a string of unicode symbols   (utf8->unicode)
            build_analyzer()        Return a callable that handles preprocessing and tokenization
            build_preprocessor()    Return a function to preprocess the text before tokenization
            build_tokenizer()       Return a function that splits a string into a sequence of tokens

            fit(raw_documents[, y])     Learn a vocabulary dictionary of all tokens in the raw documents.
            fit_transform(raw_documents[, y])   Learn the vocabulary dictionary and return term-document matrix.
            transform(raw_documents)    Transform documents to document-term matrix.
            inverse_transform(X)        Return terms per document with nonzero entries in X.

            get_feature_names()         Array mapping from feature integer indices to feature name
        Attributes:
            vocabulary_ : dict
            stop_words_ : set
        """

    Common_Vectorizer_usage()
        

if __name__ == "__main__":
    #all tests are from http://scikit-learn.org/stable/modules/feature_extraction.html
    #text is  like:
    #4.2. Feature extraction
    #4.2.1. Loading features from dicts     #test_DictVectorizer()
    #4.2.2. Feature hashing                 #test_FeatureHasher()
    #4.2.3. Text feature extraction         #test_text_feature_extraction()
    #   4.2.3.1. The Bag of Words representation
    #   4.2.3.2. Sparsity
    #   4.2.3.3. Common Vectorizer usage    sklearn.feature_extraction.text.CountVectorizer
    #   4.2.3.4. Tf–idf term weighting
    #   4.2.3.5. Decoding text files
    #   4.2.3.6. Applications and examples
    #   4.2.3.7. Limitations of the Bag of Words representation
    #   4.2.3.8. Vectorizing a large text corpus with the hashing trick
    #   4.2.3.9. Performing out-of-core scaling with HashingVectorizer
    #   4.2.3.10. Customizing the vectorizer classes
    #4.2.4. Image feature extraction

    #相关的类：
    #   sklearn.feature_extraction
    #       DictVectorizer
    #       FeatureHasher
    #   sklearn.feature_extraction.text
    #       CountVectorizer
    #       HashingVectorizer
    #       TfidfTransformer
    #       TfidfVectorizer(=CountVectorizer+TfidfTransformer)
    #   sklearn.feature_extraction.image
    #       略

    #test_DictVectorizer()
    #test_FeatureHasher()
    test_text_feature_extraction()
