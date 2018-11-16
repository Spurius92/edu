import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import json
from pandas.io.json import json_normalize
import gc
import sys
import math
from datetime import datetime

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import TimeSeriesSplit, KFold
from sklearn.metrics import mean_squared_error

from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

import warnings
warnings.filterwarnings("ignore")
#import os


#this is a preprocessing of some JSON columns in dataset. 
gc.enable()
features = ['channelGrouping', 'date', 'fullVisitorId', 'visitId',\
       'visitNumber', 'visitStartTime', 'device.browser',\
       'device.deviceCategory', 'device.isMobile', 'device.operatingSystem',\
       'geoNetwork.city', 'geoNetwork.continent', 'geoNetwork.country',\
       'geoNetwork.metro', 'geoNetwork.networkDomain', 'geoNetwork.region',\
       'geoNetwork.subContinent', 'totals.bounces', 'totals.hits',\
       'totals.newVisits', 'totals.pageviews', 'totals.transactionRevenue',\
       'trafficSource.adContent', 'trafficSource.campaign',\
       'trafficSource.isTrueDirect', 'trafficSource.keyword',\
       'trafficSource.medium', 'trafficSource.referralPath',\
       'trafficSource.source']

def load_df(csv_path):
    JSON_COLUMNS = ["device", "geoNetwork", "trafficSource", "totals"]
    ans = pd.DataFrame()
    dfs = pd.read_csv(csv_path, sep=',',
                    converters = {column: json.loads for column in JSON_COLUMNS},
                    dtype={'fullVisitorId': 'str'}, # otherwise error occurs
                    chunksize=100000)    #chunksize needed to avoid memory overload
    for df in dfs:
        df.reset_index(drop=True, inplace=True)
        for column in JSON_COLUMNS:
            column_as_df = json_normalize(df[column])
            column_as_df.columns = [f"{column}.{subcolumn}" for subcolumn in column_as_df.columns]
            df = df.drop(column, axis=1).merge(column_as_df, right_index=True, left_index=True)
    #print(f"Loaded {os.path.basename(csv_path)}. Shape: {df.shape}")
        use_df = df[features]
        del df
        gc.collect()
        ans = pd.concat([ans, use_df], axis=0).reset_index(drop=True)
    return ans

train = load_df('../input/train_v2.csv')
test = load_df('../input/test_v2.csv')
data = train.append(test, sort=False)

train['totals.transactionRevenue'].fillna(0, inplace=True)
test['totals.transactionRevenue'].fillna(0, inplace=True)
train['totals.transactionRevenue'] = np.log1p(train['totals.transactionRevenue'].astype(float))
#train['totals.transactionRevenue'].describe()
'''
data = train.append(test, sort=False)
nulls = data.isnull().sum()
print(nulls[nulls>0])
print('-'*30)
print(train.shape, test.shape)         This i will need 
print('-'*30)
print(data.shape)
'''
#datetime processing function 
def datetime_proc(train):
    train['date'] = pd.to_datetime(train['date'])
    
    train['day'] = train['date'].dt.day
    train['weekday'] = train['date'].dt.weekday
    train['month'] = train['date'].dt.month
    train['weekofyear'] = train['date'].dt.weekofyear
    train['year'] = train['date'].dt.year
    
    train['day_unique_users'] = train.groupby('day')['fullVisitorId'].transform('nunique')
    train['weekday_unique_users'] = train.groupby('weekday')['fullVisitorId'].transform('nunique')
    train['month_unique_users'] = train.groupby('month')['fullVisitorId'].transform('nunique')
    train['weekofyear_unique_users'] = train.groupby('weekofyear')['fullVisitorId'].transform('nunique')
    train['year_unique_users'] = train.groupby('year')['fullVisitorId'].transform('nunique')
datetime_proc(train)
datetime_proc(test)

for col in ['trafficSource.adContent', 'trafficSource.keyword', 'trafficSource.referralPath']:
    data[col].fillna('unknown', inplace=True)
#filling missing values as integers 
data['totals.pageviews'].fillna(1, inplace=True) 
data['totals.newVisits'].fillna(0, inplace=True)  
data['totals.bounces'].fillna(0, inplace=True)  
data['totals.pageviews'] = data['totals.pageviews'].astype(int)
data['totals.newVisits'] = data['totals.newVisits'].astype(int)
data['totals.bounces'] = data['totals.bounces'].astype(int)

data['trafficSource.isTrueDirect'].fillna(False, inplace=True)

#rows with revenue >0
train_rev = train[train['totals.transactionRevenue']>0].copy()
print(len(train_rev))
train_rev.head()


#preparing for modeling
train = data[data['totals.transactionRevenue'].notnull()]
test = data[data['totals.transactionRevenue'].isnull()].drop(['totals.transactionRevenue'], axis=1)

train_id = train['fullVisitorId']
test_id = test['fullVisitorId']
X_train = train.drop(['fullVisitorId'], axis=1)
X_test = test.drop(['fullVisitorId'],axis=1)
Y_train_reg = train.pop('totals.transactionRevenue')
print(X_train.shape, X_test.shape)

#set parameters for GroupedFold
params = {'learning_rate': 0.01,
         'objective': 'regression',
         'metric': 'rmse',
         'num_leaves': 31,
         'verbose': 1,
         'random_state': 42,
         'bagging_fraction': 0.6,
         'feature_fraction': 0.6
         }
fold = GroupKFold(n_splits=5)
oof_preds = np.zeros(X_train.shape[0])
sub_preds = np.zeros(X_test.shape[0])
for fold_, (train_, valid_) in enumerate(fold.split(X_train, Y_train_reg, groups=train_id)):
    train_x, train_y = X_train.iloc[train_], Y_train_reg.iloc[train_]
    valid_x, valid_y = X_train.iloc[valid_], Y_train_reg.iloc[valid_]
    
    reg = lgb.LGBMRegressor(**params, n_estimators=3000)
    reg.fit(train_x, train_y, eval_set=[(valid_x, valid_y)], early_stopping_rounds=50, verbose=500)
    
    oof_preds[valid_] = reg.predict(valid_x, num_iteration=reg.best_iteration_)
    sub_preds += reg.predict(X_test, num_iteration=reg.best_iteration) / fold.n_splits
preds = sub_preds


submission = pd.DataFrame({'fullVisitorId': test_id, 'PredictedLogRevenue': preds})
submission['PredictedLogRevenue'] = np.expm1(submission['PredictedLogRevenue'])
submission['PredictedLogRevenue'] = submission['PredictedLogRevenue'].apply(lambda x: 0.0 if x<0 else x)
#submission['PredictedLogRevenue'] = submission['PredictedLogRevenue'].fillna(0.0)
submission1 = submission[['fullVisitorId', 'PredictedLogRevenue']].groupby('fullVisitorId').sum().reset_index()
submission1['PredictedLogRevenue'] = np.log1p(submission1['PredictedLogRevenue'])
submission1.to_csv('submission1.csv', index=False)
