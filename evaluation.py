'''
Using a new test set I will retrain a pipeline using the optimal parameters that the pipeline learned.
'''

import joblib
%matplotlib inline
import matplotlib.pyplot as plt
import os, sys
import itertools
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix

# Load sample and the best pipeline
sampled_X, sampled_y, model = joblib.load('sampled-data1.pkl')
param_grid3, pipe3, reduction_model = joblib.load('pipeline-model.pkl')

# Retrain a pipeline using the full sampled training data set
column_names = ['national_inv', 'lead_time', 'in_transit_qty', 'forecast_3_month', 'forecast_6_month', 'forcast_9_month',
                'sales_1_month', 'sales_3_month', 'sales_6_month', 'sales_9_month', 'min_bank', 'potential_issue',
                'pieces_past_due', 'perf_6_month_avg', 'perf_12_month_avg', 'local_bo_qty', 'deck_risk', 'oe_constraint', 
                'ppap_risk', 'stop_auto_buy', 'rev_stop']

scale_boi = MinMaxScaler() # Using MinMaxScaler()
scale_boi.fit(sampled_X)

sampled_X = scale_boi.transform(sampled_X)
sampled_X = pd.DataFrame(sampled_X, columns=column_names)

grid_model = GridSearchCV(pipe3, param_grid = param_grid3, n_jobs=5, cv=5)

grid_model.fit(sampled_X, sampled_y)

# save the trained model with the pickle library
joblib.dump([grid_model], 'grid_models.pkl')

# loading testing data and evaluate the model
# preprocessing needs to be done here
TEST_DATASET = '/dsa/data/all_datasets/back_order/Kaggle_Test_Dataset_v2.csv'
assert os.path.exists(TEST_DATASET)

# Load and shuffle
df = pd.read_csv(TEST_DATASET).sample(frac = 1).reset_index(drop=True)

df = df.drop('sku', axis=1)

yes_no_columns = list(filter(lambda i: df[i].dtype!=np.float64, df.columns))

for column_name in yes_no_columns:
    mode = df[column_name].apply(str).mode()[0]
    print('Filling missing values of {} with {}'.format(column_name, mode))
    df[column_name].fillna(mode, inplace=True)
    
df.lead_time = df.lead_time.fillna(df.lead_time.median())

df = df.dropna()


encoding_list = ['potential_issue', 'deck_risk', 'oe_constraint', 'ppap_risk', 'stop_auto_buy', 'rev_stop',
                  'went_on_backorder']

for col in encoding_list:                   
    df[col] = df[col].map({'No':0, 'Yes':1})
    
X = df.drop('went_on_backorder', axis=1)
y = df['went_on_backorder']


column_names = ['national_inv', 'lead_time', 'in_transit_qty', 'forecast_3_month', 'forecast_6_month', 'forcast_9_month',
                'sales_1_month', 'sales_3_month', 'sales_6_month', 'sales_9_month', 'min_bank', 'potential_issue',
                'pieces_past_due', 'perf_6_month_avg', 'perf_12_month_avg', 'local_bo_qty', 'deck_risk', 'oe_constraint', 
                'ppap_risk', 'stop_auto_buy', 'rev_stop']

scale_boi = MinMaxScaler() # Using MinMaxScaler()
scale_boi.fit(X)

X = scale_boi.transform(X)
X = pd.DataFrame(X, columns=column_names)

# now I can preidct and evaluate with the preprocessed test set.
## without outliers

forest = reduction_model.fit(X, y)

forest_outliers = forest.predict(X)==-1
X_forest = X[~forest_outliers]
y_forest = y[~forest_outliers]
print("Number of outliers = {}".format(np.sum(forest_outliers)))

## wihtout outliers
predicted_y_reduced = grid_model.predict(X_forest)

## with outliers
predicted_y_norm = grid_model.predict(X)

# without outliers evaluation
print(classification_report(y_forest, predicted_y_reduced))
print("Pipeline Precision:", np.round(precision_score(y_forest, predicted_y_reduced, average='weighted'), 3))
print("Pipeline Recall:", np.round(precision_score(y_forest, predicted_y_reduced, average='weighted'), 3))
print("Pipeline F1-Score:", np.round(f1_score(y_forest, predicted_y_reduced, average='weighted'), 3))
print("Pipeline Accuracy:", np.round(accuracy_score(y_forest, predicted_y_reduced), 3))

pd.DataFrame(confusion_matrix(y_forest, predicted_y_reduced))

# with outliers evaluation
print(classification_report(y, predicted_y_norm))
print("Pipeline Precision:", np.round(precision_score(y, predicted_y_norm, average='weighted'), 3))
print("Pipeline Recall:", np.round(precision_score(y, predicted_y_norm, average='weighted'), 3))
print("Pipeline F1-Score:", np.round(f1_score(y, predicted_y_norm, average='weighted'), 3))
print("Pipeline Accuracy:", np.round(accuracy_score(y, predicted_y_norm), 3))

pd.DataFrame(confusion_matrix(y, predicted_y_norm))
