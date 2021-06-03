'''
The dataset is originally from Kaggle. The key task it to predict whether a product will go on backorder. The preprocessing will icnlude looking at the training dataset
and perform data preparation, exploratory daya analysis, dimensionality reduction, then train and validate
'''

## PART 1: Data Preprocessiong

%matplotlib inline
import matplotlib.pyplot as plt
import os, sys
import itertools
import numpy as np
import pandas as pd
import sys
from imblearn.under_sampling import RandomUnderSampler 
import joblib

# These are the usual ipython objects, including this one you are creating
ipython_vars = ['In', 'Out', 'exit', 'quit', 'get_ipython', 'ipython_vars']

# Get a sorted list of the objects and their sizes
temp = sorted([(x, sys.getsizeof(globals().get(x))/1024)
        for x in dir() 
        if not x.startswith('_') and x not in sys.modules and x not in ipython_vars], key=lambda x: x[1], reverse=True)
df_temp = pd.DataFrame(temp, columns = ['variable', 'size_in_kilobytes'])


# Dataset location
DATASET = '/dsa/data/all_datasets/back_order/Kaggle_Training_Dataset_v2.csv'
assert os.path.exists(DATASET)

# Load and shuffle
df = pd.read_csv(DATASET).sample(frac = 1).reset_index(drop=True)

df.describe().transpose()


## Processing

df.info() # getting info on dataset

# Taking samples and examining the dataset
df.iloc[:3,:6]
df.iloc[:3,6:12]
df.iloc[:3,12:18]
df.iloc[:3,18:24]

# dropping columns that are irrelevant and not processable
df = df.drop('sku', axis=1)

# find unique values of string columns
# All the column names of these yes/no columns
yes_no_columns = list(filter(lambda i: df[i].dtype!=np.float64, df.columns))
print(yes_no_columns)

# making a separate dataframe for the yes no columns to check for unqiue values. Sku is being dropped and will not be used
yes_no_col = df[['potential_issue', 'deck_risk', 'oe_constraint', 'ppap_risk', 'stop_auto_buy', 'rev_stop',
                  'went_on_backorder']]


for col in list(yes_no_col):          # for col in previously made separate dataframe converted to list structure
    print(col)                        # print the column name
    print(yes_no_col[col].unique())   # print the unique values in each column
    
# the previous bit of code showed some nan values. Those will be replaced with the mode
for column_name in yes_no_columns:
    mode = df[column_name].apply(str).mode()[0]
    print('Filling missing values of {} with {}'.format(column_name, mode))
    df[column_name].fillna(mode, inplace=True)
    
# there are other null values in this dataset that needs to be taken care of
total_nan = df.isnull().sum().sort_values(ascending=False) # This will show all the null values for each column when called

df.lead_time = df.lead_time.fillna(df.lead_time.median()) # using median for lead_time

# There is only 1 na value in each column besides lead_time so I will just drop them
df = df.dropna()

# now I will convert yes/no columns into binary
# probably couldve used yes_no_columns or yes_no_col, but made this list anyways for encoding
encoding_list = ['potential_issue', 'deck_risk', 'oe_constraint', 'ppap_risk', 'stop_auto_buy', 'rev_stop',
                  'went_on_backorder']

for col in encoding_list:                       # for column in the list made above
    df[col] = df[col].map({'No':0, 'Yes':1})    # map N0 = 0 and yes = 0 for columns in df that are in the encoding_list
    
# Will sample the data into a more manageable size for cross-fold validation in Grid Search
num_backorder = np.sum(df['went_on_backorder']==1)
print('backorder ratio:', num_backorder, '/', len(df), '=', num_backorder / len(df))
df.went_on_backorder.value_counts()

# because the dataset is so large I will try an undersampling technique

X = df.drop('went_on_backorder', axis=1)
y = df['went_on_backorder']

# RandomUnderSampler

# method
undersampler = RandomUnderSampler()
sampled_X, sampled_y = undersampler.fit_resample(X, y)

# writing the data to a file for reloading later
joblib.dump([sampled_X, sampled_y, undersampler], 'sampled-data1.pkl')
