### Setup

import os
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
import sklearn.metrics as skmetric
from sklearn.preprocessing import StandardScaler


### Functions

def read_from_files(DIR_INPUT, BEGIN_DATE, END_DATE):
    
    files = [os.path.join(DIR_INPUT, f) for f in os.listdir(DIR_INPUT) if f>=BEGIN_DATE+'.pkl' and f<=END_DATE+'.pkl']

    frames = []
    for f in files:
        df = pd.read_pickle(f)
        frames.append(df)
        del df
    df_final = pd.concat(frames)
    
    df_final=df_final.sort_values('TRANSACTION_ID')
    df_final.reset_index(drop=True,inplace=True)
    #  Note: -1 are missing values for real world data 
    df_final=df_final.replace([-1],0)
    
    return df_final

def output_df(x_train, x_test, y_train, y_test):
    return x_train, x_test, y_train, y_test



### Data import

DIR_INPUT = './dataset/' 

BEGIN_DATE = "2021-01-01"
END_DATE = "2021-12-31"

df = read_from_files(DIR_INPUT, BEGIN_DATE, END_DATE)

# print("Load  files")
# print("{0} transactions loaded, containing {1} fraudulent transactions".format(len(df), df.TX_FRAUD.sum()))

output = 'TX_FRAUD'

input = ['TX_AMOUNT','TX_DURING_WEEKEND', 'TX_DURING_NIGHT',
         'CUSTOMER_ID_NB_TX_1DAY_WINDOW', 'CUSTOMER_ID_AVG_AMOUNT_1DAY_WINDOW',
         'CUSTOMER_ID_NB_TX_7DAY_WINDOW', 'CUSTOMER_ID_AVG_AMOUNT_7DAY_WINDOW',
         'CUSTOMER_ID_NB_TX_30DAY_WINDOW', 'CUSTOMER_ID_AVG_AMOUNT_30DAY_WINDOW',
         'TERMINAL_ID_NB_TX_1DAY_WINDOW', 'TERMINAL_ID_RISK_1DAY_WINDOW',
         'TERMINAL_ID_NB_TX_7DAY_WINDOW', 'TERMINAL_ID_RISK_7DAY_WINDOW',
         'TERMINAL_ID_NB_TX_30DAY_WINDOW', 'TERMINAL_ID_RISK_30DAY_WINDOW']

x = StandardScaler().fit_transform(df[input].values)
y = np.array(df[output].values)

# print(x.shape)
# print(y.shape)

# counter = Counter(y)
# print('Fraud Count: ', counter[1])
# print('Fraud Ratio: ', round(counter[1]/(counter[1]+counter[0]) ,3)*100 , ' %')



### Oversampling with SMOTE

oversample = SMOTE()

x_os, y_os = oversample.fit_resample(x, y)

# counter = Counter(y_os)
# print('Data after SMOTE')
# print('Fraud Count: ', counter[1])
# print('Fraud Ratio: ', round(counter[1]/(counter[1]+counter[0]) ,3)*100 , ' %')


### Split train and test

x_train, x_test, y_train, y_test = train_test_split(x_os, y_os, test_size = 0.3)

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)


### Output for model 

output_df(x_train, x_test, y_train, y_test)