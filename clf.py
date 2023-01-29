import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import precision_score, recall_score
import time

var_dim = 90
# load data
def load_data(moead_object):
    # load data&label
    # moead_object.data = 
    # moead_object.target = 
    pass 

# logistic regression
def logistic_func(moead_object):
    clf = LogisticRegression()
    
    return clf

# object function
def obj_func(moead_object, solution):
    lgc = logistic_func(moead_object)
    
    lgc.fit(moead_object.X_train[:, solution], 
            moead_object.y_train)
    y_pred = lgc.predict(moead_object.X_test[:, solution])
    y_true = moead_object.y_test
    # precision score
    p_score = 1-precision_score(y_true, y_pred, zero_division=0)
    # recall score
    r_score = 1-recall_score(y_true, y_pred, labels=[0, 1], zero_division=0)
    return [p_score, r_score]
