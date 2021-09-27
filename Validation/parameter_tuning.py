import shap
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from tqdm.notebook import tqdm

from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_curve, auc
from sklearn.svm import NuSVC, SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.inspection import permutation_importance
import sklearn.datasets as datasets
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import csv
import sys

from tqdm.notebook import tqdm



def tune_rf(X_train, y_train, X_test, y_test, success_metric='roc'):
    
    print('Tuning random forest classifier:')
    
    n_estimators_range = tqdm(range(1, 200))
    max_depth_range = range(1, 5)
    best_n_estimators = None
    best_max_depth = None
    best_roc = 0
    best_accuracy = 0
    
    
    for max_depth in max_depth_range:
        for n_estimators in n_estimators_range:
            model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=98).fit(X_train, y_train)
            if success_metric == 'roc':
                roc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
                if roc > best_roc:
                    best_roc = roc
                    best_n_estimators = n_estimators
                    best_max_depth = max_depth
            if success_metric == 'accuracy':
                accuracy = accuracy_score(y_test, model.predict(X_test))
                if roc > best_roc:
                    best_roc = roc
                    best_n_estimators = n_estimators
                    best_max_depth = max_depth
            
    print('====================RESULTS=====================')
    print('Best results with tuning: ')
    print('roc: ' + str(best_roc))
    print('n_estimators: ' + str(best_n_estimators))
    print('max_depth: ' + str(best_max_depth))

def tune_xgb(X_train, y_train, X_test, y_test, success_metric='roc'):
    
    n_estimators_range = range(0, 200)
    max_depth_range = range(0, 5)
    best_n_estimators = None
    best_max_depth = None
    best_roc = 0
    best_accuracy = 0
    
    for n_estimators in n_estimators_range:
        for max_depth in max_depth_range:
            model = XGBClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=98).train(X_train, y_train)
            if success_metric == 'roc':
                roc = roc_auc_score(y_test, model.predict_proba(X_test[meta_features])[:, 1])
                if roc > best_roc:
                    best_roc = roc
                    best_n_estimators = n_estimators
                    best_max_depth = max_depth
            if success_metric == 'accuracy':
                accuracy = accuracy_score(y_test, model.predict(X_test))
                if roc > best_roc:
                    best_roc = roc
                    best_n_estimators = n_estimators
                    best_max_depth = max_depth
                
            
    print('====================RESULTS=====================')
    print('Best results with tuning: ')
    print('roc: ' + str(best_roc))
    print('n_estimators: ' + str(best_n_estimators))
    print('max_depth: ' + str(best_max_depth))