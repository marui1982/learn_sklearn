import numpy as np
from sklearn.cross_validation import train_test_split
X, y = np.arange(10).reshape((5, 2)), range(5)
print X 
print y

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42)
print 'train============='
print X_train
print y_train
print 'test============='
print X_test
print y_test

