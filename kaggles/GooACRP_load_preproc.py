import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import json
from pandas.io.json import json_normalize

import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from catboost import CatBoostRegressor

#json preprocessing function
def load_df(csv_path, nrows=None):
    USE_COLUMNS = ["channelGrouping", "date", "device", "fullVisitorId", "geoNetwork",
        "socialEngagementType", "totals", "trafficSource", "visitId",
        "visitNumber", "visitStartTime"]
    JSON_COLUMNS = ["totals", "trafficSource", "geoNetwork", "device"]
   
    df = pd.read_csv(csv_path, 
                    converters = {column: json.loads for column in JSON_COLUMNS},
                    dtype = {"fullVisitorId": 'str'},
                    nrows = nrows, usecols = USE_COLUMNS)
    
    for column in JSON_COLUMNS:
        column_as_df = json_normalize(df[column])
        column_as_df.columns = [f"{column}.{subcolumn}" for subcolumn in column_as_df.columns]
        df = df.drop(column, axis=1).merge(column_as_df, left_index=True, right_index=True)
    return df

train = load_df('../input/train_v2.csv')
test = load_df('../input/test_v2.csv')

test_id = test['fullVisitorId']
test_len=len(test)
data = train.append(test, sort=False)
del train

data = data[500000:]
data = data.drop(labels=data.columns[data.nunique()<=2], axis=1)
data['totals.totalTransactionRevenue'] = np.log1p(data['totals.totalTransactionRevenue'].astype(float))
data['totals.transactionRevenue'] = np.log1p(test['totals.transactionRevenue'].astype(float))

#fillnas for numerical columns
def fillnas(data):   #fill in nans in numerical columns
    totals_cols = [col for col in data.columns if col.startswith('totals') \
                and col not in ['totals.totalTransactionRevenue', \
                                'totals.transactionRevenue', 'totals.transactions']]
    for col in totals_cols:
        data[col].fillna(1, inplace=True)
        data[col] = data[col].astype(int)
    
    trans_cols = ['totals.totalTransactionRevenue', 'totals.transactionRevenue', 'totals.transactions']
    for col in trans_cols:
        data[col].fillna(0, inplace=True)
        data[col] = data[col].astype(float)
    
    return data
fillnas(data)

#fillna categorical columns and return their names
cat_cols = []
def cat_col_process(data):
    for col in data.columns:
        if data[col].dtype == 'object':
            cat_cols.append(col)
    for col in cat_cols:
        data[col].fillna('unknown', inplace=True)
    cat_cols.remove('fullVisitorId')
    return data
cat_col_process(data)


#datetime processing function 
def datetime_proc(data):
    data['date'] = data['date'].apply(lambda x: str(x)[:4] + '-' + str(x)[4:6] + '-' + str(x)[6:])
    data['date'] = pd.to_datetime(data['date'])
    
    data['day'] = data['date'].dt.day
    data['weekday'] = data['date'].dt.weekday
    data['month'] = data['date'].dt.month
    data['weekofyear'] = data['date'].dt.weekofyear
    data['year'] = data['date'].dt.year
    
    data['day_unique_users'] = data.groupby('day')['fullVisitorId'].transform('nunique')
    data['weekday_unique_users'] = data.groupby('weekday')['fullVisitorId'].transform('nunique')
    data['month_unique_users'] = data.groupby('month')['fullVisitorId'].transform('nunique')
    data['weekofyear_unique_users'] = data.groupby('weekofyear')['fullVisitorId'].transform('nunique')
    data['year_unique_users'] = data.groupby('year')['fullVisitorId'].transform('nunique')
    return data 
datetime_proc(data)


#detect and count nulls in data
nulls = data.isnull().sum()
print(nulls[nulls>0])
print('-'*30)
print(data.shape)
print('-'*30)
