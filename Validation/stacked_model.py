import shap
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from tqdm.notebook import tqdm

from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_curve, auc, roc_curve
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
import random
import math

random.seed(a=98)

def choose_explainer(model, X, model_type):
    tree_models = ['rf', 'xgb']
    kernel_models = ['SVC', 'MLP', 'NuSVC', 'KNN']
    if model_type in tree_models:
        return shap.TreeExplainer(model)
    else:
        return shap.KernelExplainer(model.predict_proba, X)

def add_predict_proba_to_df(model, model_type, df, features):
    model_predict = model.predict(df[features])
    model_proba = [proba[1] for proba in model.predict_proba(df[features])]
    df[model_type + '_predict'] = model_predict
    df[model_type + '_proba'] = model_proba
    return df

def add_shap_to_df(model, model_type, df, index_shap=True, features=[]):
#     print('Original # of features: ', len(df.columns))
    if index_shap:
        explainer = choose_explainer(model=model, X=df[features], model_type=model_type)
        shap_values = explainer.shap_values(df[features])
#         display('Generated SHAP values for model type: ' + model_type)
        for feat in features:
            df[feat + '_' + model_type +'_shap'] = [shap_values[1][n][df[features].columns.get_loc(feat)] for n in range(0, len(df[features]))]
    else: 
        explainer = choose_explainer(model=model, X=df[features], model_type=model_type)
        shap_values = explainer.shap_values(df[features])
#         display('Generated SHAP values for model type: ' + model_type)
        for feat in features:
            df[feat + '_' + model_type +'_shap'] = [shap_values[n][df[features].columns.get_loc(feat)] for n in range(0, len(df[features]))]
        
    return df
#     print('New # of features: ', len(df.columns))


def add_shap_pred_proba_to_df(model, model_type, df, index_shap=True, features=[]):
    df = add_shap_to_df(model=model, model_type=model_type, df=df, index_shap=index_shap, features=features)
    df = add_predict_proba_to_df(model=model, model_type=model_type, df=df, features=features)
    return df
    
def prepare_csv_data(filepath, y_label, hot_encode_labels=[], features_to_drop=[],  test_size=0.33):
    df = pd.read_csv(filepath)
    
    if features_to_drop:
#         print('Dropped: ', str(features_to_drop))
        df = df.drop(columns=features_to_drop)
    
    for label in hot_encode_labels:
        hot_enc = OneHotEncoder(handle_unknown='ignore')
        enc_df = pd.DataFrame(hot_enc.fit_transform(df[[label]]).toarray())
        for col in enc_df.columns:
            enc_df = enc_df.rename(columns={col: '{label}_{col}'.format(label=label, col=col)})
        df = df.join(enc_df)
        df = df.drop(columns=[label])
    
    df = df.dropna()
    
    y = df[y_label].copy()
    df = df.drop(columns=[y_label])
    X = df.copy()
    return train_test_split(X, y, test_size=test_size, random_state=98)

def prepare_csv_data_k_folds(filepath, y_label, hot_encode_labels=[], features_to_drop=[],  k_folds=5):
    df = pd.read_csv(filepath)
    
    if features_to_drop:
#         print('Dropped: ', str(features_to_drop))
        df = df.drop(columns=features_to_drop)
    
    for label in hot_encode_labels:
        hot_enc = OneHotEncoder(handle_unknown='ignore')
        enc_df = pd.DataFrame(hot_enc.fit_transform(df[[label]]).toarray())
        for col in enc_df.columns:
            enc_df = enc_df.rename(columns={col: '{label}_{col}'.format(label=label, col=col)})
        df = df.join(enc_df)
        df = df.drop(columns=[label])
    
    df = df.dropna()
    
    y = df[y_label].copy()
    df = df.drop(columns=[y_label])
    X = df.copy()
    
    k_fold_array = []
    for k in range(0, k_folds):
        k_fold_array += [k]*(math.ceil(len(X)/k_folds))
    
    del k_fold_array[-(len(k_fold_array) - len(X)):]
    
    random.shuffle(k_fold_array)
    
    X = X.assign(k_fold=k_fold_array)
    
    return X, y

def train_level_1(model_parameters, X_train, y_train):
    
    stacked_model = {}
    
    feature_selection_technique = model_parameters['feature_selection_level_1']['technique']
    
    level_1_features = feature_selection_fun_full(X_train, y_train, selection_technique=feature_selection_technique, correlation=True)
    
    for model in model_parameters.keys() - ['feature_selection_level_1', 'feature_selection_meta']:
        if model == 'xgb':
            stacked_model['xgb'] = XGBClassifier(use_label_encoder=False, eval_metric='logloss', n_estimators=model_parameters['xgb']['n_estimators'], max_depth=model_parameters['xgb']['max_depth']).fit(X_train[level_1_features], y_train)
        if model == 'KNN':
            stacked_model['KNN'] = KNeighborsClassifier(n_neighbors=model_parameters['KNN']['n_neighbors']).fit(X_train[level_1_features], y_train)
        if model == 'MLP':
            stacked_model['MLP'] = MLPClassifier(activation = "relu", alpha = model_parameters['MLP']['alpha'], hidden_layer_sizes = (10,10,10), learning_rate = "constant", max_iter = model_parameters['MLP']['max_iter'], random_state = 1000).fit(X_train[level_1_features], y_train)
    
#     stacked_model['rf'] = RandomForestClassifier(n_estimators=10, max_depth=2).fit(X_train[level_1_features], y_train)
#     stacked_model['SVC'] = SVC(kernel='linear', probability=True).fit(X_train[level_1_features], y_train)
    
#     stacked_model['NuSVC'] = NuSVC(degree = 1, kernel = "rbf", nu = 0.25, probability = True).fit(X_train[level_1_features], y_train)
    return stacked_model, level_1_features
    

def add_shap_pred_proba_to_level_1(model_parameters, stacked_model, X, y, features):
    for model in model_parameters.keys() - ['feature_selection_level_1', 'feature_selection_meta']:
        if model == 'xgb':
            X = add_shap_pred_proba_to_df(model=stacked_model['xgb'], model_type='xgb', df=X, index_shap=False, features=features)
        if model == 'KNN':
            X = add_shap_pred_proba_to_df(model=stacked_model['KNN'], model_type='KNN', df=X, index_shap=True, features=features)
#     X = add_shap_pred_proba_to_df(model=stacked_model['rf'], model_type='rf', df=X, index_shap=True, features=features)
#     X = add_shap_pred_proba_to_df(model=stacked_model['SVC'], model_type='SVC', df=X, index_shap=True, features=features)
        if model == 'MLP':
            X = add_shap_pred_proba_to_df(model=stacked_model['MLP'], model_type='MLP', df=X, index_shap=True, features=features)
#     X = add_shap_pred_proba_to_df(model=stacked_model['NuSVC'], model_type='NuSVC', df=X, index_shap=True, features=features)
    return X
    

def train_meta_model(model_parameters, stacked_model, X_train, y_train, level_1_features):
    
    feature_selection_technique = model_parameters['feature_selection_meta']['technique']
    
    shap_columns = []
    
    for feat in level_1_features:
        shap_columns += [feat + '_' + model_type + '_shap' for model_type in stacked_model.keys()]
    
    meta_features = feature_selection_fun_full(X_train.drop(columns=shap_columns), y_train, selection_technique=feature_selection_technique, correlation=True)

    meta_shap_features = feature_selection_fun_full(X_train, y_train, selection_technique=feature_selection_technique, correlation=True)   
    
    stacked_model['meta'] = SVC(kernel='linear', probability=True).fit(X_train[meta_features], y_train)
    
    stacked_model['meta_shap'] = SVC(kernel='linear', probability=True).fit(X_train[meta_shap_features], y_train)
    
#     display('Meta features: ', meta_features)
    
#     display('Meta_shap features: ', meta_shap_features)
    
    return  stacked_model, shap_columns, meta_features, meta_shap_features

def test_stacked_model(stacked_model, X_test, y_test, level_1_features, shap_columns, meta_features, meta_shap_features):
    X_test = add_shap_pred_proba_to_level_1(stacked_model=stacked_model, X=X_test, y=y_test, features=level_1_features)
    
#     print('# of features W/O SHAP: ', len(meta_features))
#     print('# of features W/ SHAP: ', len(meta_shap_features))
    
#     display('Accuracy W/O SHAP: ', accuracy_score(y_test, stacked_model['meta'].predict(X_test[meta_features])))
    
#     display('Accuracy W/ SHAP: ', accuracy_score(y_test, stacked_model['meta_shap'].predict(X_test[meta_shap_features])))
    
#     display(stacked_model['meta_shap'].coef_)
    
    print('ROC W/O SHAP: ', roc_auc_score(y_test, stacked_model['meta'].predict_proba(X_test[meta_features])[:, 1]))
    print('ROC W/ SHAP: ', roc_auc_score(y_test, stacked_model['meta_shap'].predict_proba(X_test[meta_shap_features])[:, 1]))
    
#     r = permutation_importance(stacked_model['meta_shap'], X_test, y_test, n_repeats=30, random_state=0)
    
#     for i in r.importances_mean.argsort()[::-1]:
#         if r.importances_mean[i] - 2 * r.importances_std[i] > 0:
#             print(f"{X_test.columns[i]:<8}"f"{r.importances_mean[i]:.3f}"f" +/- {r.importances_std[i]:.3f}")
            
    
def run_analysis_on_csv_data(filepath, hot_encode_labels, features_to_drop, y_label):
    
    stacked_model = {}
    
    X_train, X_test, y_train, y_test = prepare_csv_data(filepath, hot_encode_labels, features_to_drop, y_label) #Do test_train_split on data from csv file
    
    original_features = X_train.copy().columns
    
    stacked_model = train_level_1(stacked_model, X_train, y_train)

# ============================ FEATURE SELECTION FUNCTIONS ===============================
def feature_selection_fun_full(X, y, selection_technique='roc', correlation=True):
    if selection_technique=='roc':
        if correlation:
            roc_feature_ini = feature_selection_fun_correlation_with_roc_sorting(X, y, correlation_treshold=0.9)
            selected_features = feature_selection_fun_roc_based(X[roc_feature_ini], y, roc_treshold=0.6, pr_treshold=0.15)
        else:
            selected_features = feature_selection_fun_roc_based(X, y, roc_treshold=0.6, pr_treshold=0.15)
    if selection_technique=='tree':
        selected_features = feature_selection_fun_tree_based(X, y)
    
    return selected_features
            

def feature_selection_fun_tree_based(X, y, threshold=90):
    clf = RandomForestClassifier(n_estimators=50, max_depth=3, max_features=2, random_state=0).fit(X, y)
    f = dict(zip(X.columns,clf.feature_importances_))
    selected_features = [k for k,v in f.items() if v > np.percentile(clf.feature_importances_, threshold)]
    return selected_features
    
    
def feature_selection_fun_correlation_with_roc_sorting(X, y, correlation_treshold=0.9):

    feature_list = X.columns.values
    roc_auc = []
    pr_auc = []
    for col in feature_list:
        roc_auc.append(roc_auc_score(y, X[col].values))
        precision, recall, thresholds = precision_recall_curve(y.values, X[col].values)
        pr_auc.append(auc(recall, precision))
    final_res = pd.DataFrame()
    final_res['prauc'] = np.array(pr_auc)
    final_res['rocauc'] = np.array(roc_auc)
    final_res['feature_id'] = feature_list

    X = X.reindex(final_res.sort_values(by=['prauc', 'rocauc'],ascending=False).feature_id.values, axis=1)

    corr = X.corr()
    #select columns with less than 0.9 correlation
    columns = np.full((corr.shape[0],), True, dtype=bool)
    for i in range(corr.shape[0]):
        for j in range(i+1, corr.shape[0]):
            if corr.iloc[i,j] >= correlation_treshold:
                if columns[j]:
                    columns[j] = False
    selected_columns = X.columns[columns]
    return list(selected_columns)

def feature_selection_fun_roc_based(X, y, roc_treshold=0.8, pr_treshold=0.3):
    feature_list = X.columns.values
    roc_auc = []
    pr_auc = []
    for col in feature_list:
        roc_auc.append(roc_auc_score(y, X[col].values))
        precision, recall, thresholds = precision_recall_curve(y.values, X[col].values)
        pr_auc.append(auc(recall, precision))
    final_res = pd.DataFrame()
    final_res['prauc'] = np.array(pr_auc)
    final_res['rocauc'] = np.array(roc_auc)
    final_res['feature_id'] = feature_list
    important_features = list(final_res[(final_res.rocauc>roc_treshold)&(final_res.prauc>pr_treshold)].feature_id.values)
    return important_features

def train_stacked_model_full(model_parameters, X_train, y_train):

    stacked_model, level_1_features = train_level_1(model_parameters, X_train, y_train)

    X_train = add_shap_pred_proba_to_level_1(model_parameters, stacked_model, X_train, y_train, level_1_features)

    stacked_model, shap_columns, meta_features, meta_shap_features = train_meta_model(model_parameters, stacked_model, X_train, y_train, level_1_features)
    
    stacked_model['meta_features'] = meta_features
    
    stacked_model['meta_shap_features'] = meta_shap_features

    stacked_model['level_1_features'] = level_1_features
    
    return stacked_model

def test_stacked_model_full(model_parameters, stacked_model, X_test, y_test):
    X_test = add_shap_pred_proba_to_level_1(model_parameters=model_parameters, stacked_model=stacked_model, X=X_test, y=y_test, features=stacked_model['level_1_features'])
    
    roc_no_shap = roc_auc_score(y_test, stacked_model['meta'].predict_proba(X_test[stacked_model['meta_features']])[:, 1])
    roc_shap = roc_auc_score(y_test, stacked_model['meta_shap'].predict_proba(X_test[stacked_model['meta_shap_features']])[:, 1])
    
    acc_no_shap = accuracy_score(y_test, stacked_model['meta'].predict(X_test[stacked_model['meta_features']]))
    acc_shap = accuracy_score(y_test, stacked_model['meta_shap'].predict(X_test[stacked_model['meta_shap_features']]))
    
    return roc_no_shap, roc_shap, acc_no_shap, acc_shap
    
#     print('ROC W/O SHAP: ', roc_no_shap)
#     print('ROC W/ SHAP: ', roc_shap)
#     print('Change: ', str(roc_shap - roc_no_shap))
    
#     proba = stacked_model['meta'].predict_proba(X_test[stacked_model['meta_features']])[:, 1]
#     fpr, tpr, threshold = roc_curve(y_test, proba)
#     roc_auc = auc(fpr, tpr)
    
#     plt.title('Receiver Operating Characteristic')
#     plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    
#     proba = stacked_model['meta_shap'].predict_proba(X_test[stacked_model['meta_shap_features']])[:, 1]
#     fpr, tpr, threshold = roc_curve(y_test, proba)
#     roc_auc = auc(fpr, tpr)
    
#     plt.plot(fpr, tpr, 'r', label = 'AUC = %0.2f' % roc_auc)
    
#     plt.legend(loc = 'lower right')
#     plt.plot([0, 1], [0, 1],'r--')
#     plt.xlim([0, 1])
#     plt.ylim([0, 1])
#     plt.ylabel('True Positive Rate')
#     plt.xlabel('False Positive Rate')
#     plt.show()
    
