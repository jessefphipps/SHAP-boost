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
from datetime import datetime

from imp import reload

from parameter_tuning import tune_rf, tune_xgb

from stacked_model import prepare_csv_data, train_stacked_model_full, test_stacked_model_full

print('Starting program...')

experiment_description = 'Parameter experiment without top 3 features '

filepath = 'Datasets/full_cohort_data.csv'
y_label = 'day_28_flg'
hot_encode_labels = ['day_icu_intime', 'service_unit']
features_to_drop = ['mort_day_censored', 'censor_flg', 'hosp_exp_flg', 'icu_exp_flg', 'stroke_flg']

model_parameters = {}
model_parameters['xgb'] = {}
model_parameters['KNN'] = {}
model_parameters['MLP'] = {}
model_parameters['MLP']['max_iter'] = 1000

model_parameters['feature_selection_level_1'] = {}
model_parameters['feature_selection_level_1']['technique'] = 'roc'

model_parameters['feature_selection_meta'] = {}
model_parameters['feature_selection_meta']['technique'] = 'roc'

xgb_n_estimators_range = range(1, 202, 50)
xgb_max_depth_range = range(2, 11, 2)

KNN_n_neighbors_range = range(2, 11, 2)

MLP_alpha_range = np.logspace(-4, 0, num=5)

num_models = len(xgb_n_estimators_range)*len(xgb_max_depth_range)*len(KNN_n_neighbors_range)*len(MLP_alpha_range)

print('Number of models to be trained: ' + str(len(xgb_n_estimators_range)*len(xgb_max_depth_range)*len(KNN_n_neighbors_range)*len(MLP_alpha_range)))

df_columns = ['xgb_n_estimators', 'xgb_max_range', 'KNN_n_neighbors', 'MLP_alpha', 'level_1_feats', 'meta_feats', 'meta_feats_shap', 'roc_no_shap', 'roc_shap', 'acc_no_shap', 'acc_shap', 'precision_no_shap', 'precision_shap', 'recall_no_shap', 'recall_shap', 'specificity_no_shap', 'specificity_shap']

now = datetime.now()

output_filename = 'parameter_experiment_results/' + now.strftime("%y%m%d_%H%M%S") + '.csv'

print('Output filename: ' + output_filename)

output_df = pd.DataFrame(columns=df_columns)

with open('parameter_experiment_results/parameter_experiment_log.txt', 'a') as log_file:
    log_file.write('\n{:=^40}\n'.format(now.strftime("%y%m%d_%H%M%S")))
    log_file.write('Description: {}\n'.format(experiment_description))
    log_file.write('Output file: {}\n'.format(output_filename))
    log_file.write('Level 1 feature selection: {}\n'.format(model_parameters['feature_selection_level_1']['technique']))
    log_file.write('Meta classifier feature selection: {}\n'.format(model_parameters['feature_selection_meta']['technique']))
    log_file.write('Dropped features: {}\n'.format(features_to_drop))
    log_file.write('Number of models trained: {}\n'.format(num_models))
    log_file.write('xgb_n_estimators_range: {}\n'.format(xgb_n_estimators_range))
    log_file.write('xgb_max_depth_range: {}\n'.format(xgb_max_depth_range))
    log_file.write('KNN_n_neighbors_range: {}\n'.format(KNN_n_neighbors_range))
    log_file.write('MLP_alpha_range: {}\n'.format(MLP_alpha_range))
    log_file.write('\n\n')

with open(output_filename, "w") as csv_file:
    csv_file.write('xgb_n_estimators,xgb_max_range,KNN_n_neighbors,MLP_alpha,level_1_feats,meta_feats,meta_feats_shap,roc_no_shap,roc_shap,acc_no_shap,acc_shap,precision_no_shap,precision_shap,recall_no_shap,recall_shap,specificity_no_shap,specificity_shap\n')
    for xgb_n_estimators in xgb_n_estimators_range:
        model_parameters['xgb']['n_estimators'] = xgb_n_estimators
        for xgb_max_depth in xgb_max_depth_range:
            model_parameters['xgb']['max_depth'] = xgb_max_depth
            for KNN_n_neighbors in KNN_n_neighbors_range:
                model_parameters['KNN']['n_neighbors'] = KNN_n_neighbors
                for MLP_alpha in MLP_alpha_range:
                    X_train, X_test, y_train, y_test = prepare_csv_data(filepath=filepath, y_label=y_label, hot_encode_labels=hot_encode_labels, features_to_drop=features_to_drop)
                    model_parameters['MLP']['alpha'] = MLP_alpha
                    stacked_model = train_stacked_model_full(model_parameters, X_train, y_train)
                    roc_no_shap, roc_shap, acc_no_shap, acc_shap, precision_no_shap, precision_shap, recall_no_shap, recall_shap, specificity_no_shap, specificity_shap = test_stacked_model_full(model_parameters, stacked_model, X_test, y_test)
                    new_row = {
                        'xgb_n_estimators': model_parameters['xgb']['n_estimators'],
                        'xgb_max_range': model_parameters['xgb']['max_depth'],
                        'KNN_n_neighbors': model_parameters['KNN']['n_neighbors'],
                        'MLP_alpha': model_parameters['MLP']['alpha'],
                        'level_1_feats': [stacked_model['level_1_features']],
                        'meta_feats': [stacked_model['meta_features']],
                        'meta_feats_shap': [stacked_model['meta_shap_features']],
                        'roc_no_shap': roc_no_shap,
                        'roc_shap': roc_shap,
                        'acc_no_shap': acc_no_shap,
                        'acc_shap': acc_shap,
                        'precision_no_shap': precision_no_shap,
                        'precision_shap': precision_shap,
                        'recall_no_shap': recall_no_shap,
                        'recall_shap': recall_shap,
                        'specificity_no_shap': specificity_no_shap,
                        'specificity_shap': specificity_shap
                    }
                    df_row = pd.DataFrame.from_dict(new_row)
                    csv_row = df_row.to_csv(index=False, header=False)
                    csv_file.write(csv_row)
                    output_df = output_df.append(new_row, ignore_index=True)
