'''
I will develop three unique pipelines for predicting backorder. I will use the data I wrote into a file from the preprocessing section to fit and evaluate these pipelines
'''

%matplotlib inline
import matplotlib.pyplot as plt
import os, sys
import itertools
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
from scipy.stats import uniform
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, chi2, f_classif, mutual_info_classif, RFE
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVR
from sklearn.naive_bayes import GaussianNB


# Reloading sample file from preprocessing section
sampled_X, sampled_y, model = joblib.load('sampled-data1.pkl')

# Normlaizing the data
column_names = ['national_inv', 'lead_time', 'in_transit_qty', 'forecast_3_month', 'forecast_6_month', 'forcast_9_month',
                'sales_1_month', 'sales_3_month', 'sales_6_month', 'sales_9_month', 'min_bank', 'potential_issue',
                'pieces_past_due', 'perf_6_month_avg', 'perf_12_month_avg', 'local_bo_qty', 'deck_risk', 'oe_constraint', 
                'ppap_risk', 'stop_auto_buy', 'rev_stop']

scale_boi = MinMaxScaler() # Using MinMaxScaler()
scale_boi.fit(sampled_X)

sampled_X = scale_boi.transform(sampled_X)
sampled_X = pd.DataFrame(sampled_X, columns=column_names)

# Split the data into Train/Test
X_train, X_test, y_train, y_test = train_test_split(sampled_X, sampled_y, test_size = 0.30)





## PIPELINE 1



# Anamoly Detection using LocalOutlierFactor()
# construct LocalOutlierFactor()
LOF = LocalOutlierFactor().fit(X_train, y_train)

# get labels from classifier and cull outliers
LOF_outliers = LOF.fit_predict(X_train)==-1
print("Num of outliers = {}".format({np.sum(LOF_outliers)}))
X_LOF = X_train[~LOF_outliers]
y_LOF = y_train[~LOF_outliers]
X_train_reduced1, X_test_reduced1, y_train_reduced1, y_test_reduced1 = train_test_split(X_LOF, y_LOF, test_size=0.30)

# Dimensionality reduction using SelectKBest and Naive Bayes
# parameter configuration
param_grid1 = {'CHI__k': [1, 30],
              'NB__var_smoothing': [1e-9, 1e-5]}

# the 1st pipeline
pipe1 = Pipeline([
    ('CHI', SelectKBest(chi2)), 
    ('NB', GaussianNB())                 
])

grid_model1 = GridSearchCV(pipe1, param_grid = param_grid1, n_jobs=5, cv=5)
grid_model1.fit(X_train_reduced1, y_train_reduced1)
predicted_y1 = grid_model1.predict(X_test_reduced1)

# Evaluation of Pipeline
print(classification_report(y_test_reduced1, predicted_y1))
print("Pipeline Precision:", np.round(precision_score(y_test_reduced1, predicted_y1, average='weighted'), 3))
print("Pipeline Recall:", np.round(precision_score(y_test_reduced1, predicted_y1, average='weighted'), 3))
print("Pipeline F1-Score:", np.round(f1_score(y_test_reduced1, predicted_y1, average='weighted'), 3))
print("Pipeline Accuracy:", np.round(accuracy_score(y_test_reduced1, predicted_y1), 3))

# confusion matrix
pd.DataFrame(confusion_matrix(y_test_reduced1, predicted_y1))

# best parameter and estimator
print(grid_model1.best_estimator_)
print(grid_model1.best_params_)





## PIPELINE 2


# Anamoly Detection using EllipticEnvelope
EEmodel = EllipticEnvelope().fit(X_train, y_train)

EE_outliers = EEmodel.predict(X_train)==-1
X_EE = X_train[~EE_outliers]
y_EE = y_train[~EE_outliers]
print("Number of outliers = {}".format(np.sum(EE_outliers)))
X_train_reduced2, X_test_reduced2, y_train_reduced2, y_test_reduced2 = train_test_split(X_EE, y_EE, test_size=0.30)

# Dimensionality reduction using recursive feature elimination and logistic regression
estimator =SVR(kernel = "rbf")

param_grid2 = {'RFE__n_features_to_select': [1, 30],
              'RFE__step': [0, 10],        
              'LR__C': [0.1, 5]}

# the 2nd pipeline
pipe2 = Pipeline([
    ('RFE', RFE(estimator)), 
    ('LR', LogisticRegression())                 
])

grid_model2 = GridSearchCV(pipe2, param_grid = param_grid2, n_jobs=5, cv=5)
grid_model2.fit(X_train_reduced2, y_train_reduced2)
predicted_y2 = grid_model2.predict(X_test_reduced2)

# Evaluation of Pipeline
print(classification_report(y_test_reduced2, predicted_y2))
print("Pipeline Precision:", np.round(precision_score(y_test_reduced2, predicted_y2, average='weighted'), 3))
print("Pipeline Recall:", np.round(precision_score(y_test_reduced2, predicted_y2, average='weighted'), 3))
print("Pipeline F1-Score:", np.round(f1_score(y_test_reduced2, predicted_y2, average='weighted'), 3))
print("Pipeline Accuracy:", np.round(accuracy_score(y_test_reduced2, predicted_y2), 3))

# confusion matrix
pd.DataFrame(confusion_matrix(y_test_reduced2, predicted_y2))

# best parameter and estimator
print(grid_model2.best_estimator_)
print(grid_model2.best_params_)





## PIPELINE 3


# Anamoly Detection using IsolationForest
reduction_model = IsolationForest(contamination=0.9)
forest = reduction_model.fit(X_train, y_train)

forest_outliers = forest.predict(X_train)==-1
X_forest = X_train[~forest_outliers]
y_forest = y_train[~forest_outliers]
print("Number of outliers = {}".format(np.sum(forest_outliers)))
X_train_reduced3, X_test_reduced3, y_train_reduced3, y_test_reduced3 = train_test_split(X_forest, y_forest, test_size=0.30)

# Dimensionality reduction using principal component analysis and SVC
param_grid3 = {'PCA__n_components': [5, 30],
              'SVC__C': [1e3, 5e3],        
              'SVC__kernel': ['rbf']}

# the 3rd pipeline
pipe3 = Pipeline([
    ('PCA', PCA()), 
    ('SVC', SVC())                 
])

grid_model3 = GridSearchCV(pipe3, param_grid = param_grid3, n_jobs=5, cv=5)
grid_model3.fit(X_train_reduced3, y_train_reduced3)
predicted_y3 = grid_model3.predict(X_test_reduced3)

# Evaluation of Pipeline
print(classification_report(y_test_reduced3, predicted_y3))
print("Pipeline Precision:", np.round(precision_score(y_test_reduced3, predicted_y3, average='weighted'), 3))
print("Pipeline Recall:", np.round(precision_score(y_test_reduced3, predicted_y3, average='weighted'), 3))
print("Pipeline F1-Score:", np.round(f1_score(y_test_reduced3, predicted_y3, average='weighted'), 3))
print("Pipeline Accuracy:", np.round(accuracy_score(y_test_reduced3, predicted_y3), 3))

# confusion matrix
pd.DataFrame(confusion_matrix(y_test_reduced3, predicted_y3))

# best parameter and estimator
print(grid_model3.best_estimator_)
print(grid_model3.best_params_)


# pickle the best pipeline/model
joblib.dump([param_grid3, pipe3, reduction_model], 'pipeline-model.pkl')
