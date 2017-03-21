import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

def test_roc_auc_score(y_true,y_scores):
    print roc_auc_score(y_true, y_scores)

def test_average_precision_score(y_true,y_scores):
    print average_precision_score(y_true, y_scores)  

if __name__ == '__main__':
    y_true = np.array([0, 0, 1, 1])
    y_scores = np.array([0.1, 0.4, 0.35, 0.8])
    
    test_roc_auc_score(y_true,y_scores)
    test_average_precision_score(y_true,y_scores)
