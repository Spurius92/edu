from kaggle.competitions import twosigmanews
env = twosigmanews.make_env()
(market_train, news_train) = env.get_training_data()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

import datetime
import lightgbm as lgb
from scipy import stats
from scipy.sparse import hstack, csr_matrix
from xgboost import XGBClassifier
from wordcloud import WordCloud
from collections import Counter
from nltk.corpus import stopwords
from nltk.util import ngrams
stop = set(stopwords.words('english'))

import plotly.offline as py
import plotly.graph_objs as go
import plotly.tools as tls
py.init_notebook_mode(connected=True)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import model_selection
from sklearn.metrics import accuracy_score

#checking data for outliers
market_train['price_diff'] = market_train['close'] - market_train['open']  
market_train.sort_values('price_diff')
market_train['close_to_open'] = np.abs(market_train['close']) / (market_train['open'])

#Substitute values of open and close prices, if they are too big or too small. Placing mean values instead.
market_train['assetName_open_mean'] = market_train.groupby('assetName')['open'].transform('mean')
market_train['assetName_close_mean'] = market_train.groupby('assetName')['close'].transform('mean')
for i, row in market_train.loc[market_train['close_to_open'] >= 2].iterrows():
    if np.abs(row['assetName_open_mean'] - row['open']) > np.abs(row['assetName_close_mean'] - row['close']):
        market_train.iloc[i,5] = row['assetName_open_mean']
    else:
        market_train.iloc[i,4] = row['assetName_close_mean']
for i, row in market_train.loc[market_train['close_to_open'] <= 0.5].iterrows():
    if np.abs(row['assetName_open_mean'] - row['open']) > np.abs(row['assetName_close_mean'] - row['close']):
        market_train.iloc[i,5] = row['assetName_open_mean']
    else:
        market_train.iloc[i,4] = row['assetName_close_mean']

#start modelling. data preparation function. work with 2 new dataframes and then use them
#to create simple binary model, predicting whether stock prices go up or down after the news
def data_prep(market, news):
    market['time'] = market.time.dt.date
    market['returnsOpenPrevRaw1_to_volume'] = market['returnsOpenPrevRaw1'] / market['volume']
    market['close_to_open'] = market['close'] / market['open']
    market['volume_to_mean'] = market['volume'] / market['volume'].mean()
    
    news['time'] = news.time.dt.hour
    news['sourceTimestamp']= news.sourceTimestamp.dt.hour
    news['firstCreated'] = news.firstCreated.dt.date
    news['assetCodesLen'] = news['assetCodes'].map(lambda x: len(eval(x)))
    news['assetCodes'] = news['assetCodes'].map(lambda x: list(eval(x))[0])
    news['headline'] = news['headline'].apply(lambda x: len(x))
    news['assetCodesLen'] = news['assetCodes'].apply(lambda x: len(x))
    news['asset_sentiment_count'] = news.groupby(['assetName', 'sentimentClass'])['time'].transform('count')
    news['asset_sentence_mean'] = news.groupby(['assetName', 'sentenceCount'])['time'].transform('mean')
    lbl = {k: v for v, k in enumerate(news['headlineTag'].unique())}
    news['headlineTagT'] = news['headlineTag'].map(lbl)
    kcol = ['firstCreated', 'assetCodes']
    news = news.groupby(kcol, as_index=False).mean()
    
    market = pd.merge(market, news, how='left', 
                      left_on=['time', 'assetCode'], right_on=['firstCreated', 'assetCodes'])
    lbl = {k: v for v, k in enumerate(market_df['assetCode'].unique())}
    news['assetCodeT'] = news['assetCode'].map(lbl)
    
    market = market.dropna(axis=0)
    return market

#preparing values fop the model
up = market_train.returnsOpenNextMktres10 >= 0

fcol = [c for c in market_train.columns if c not in ['assetCode', 'assetCodes', 'assetCodesLen', 'assetName', 'assetCodeT', 'volume_to_mean', 'sentence_word_count',
                                             'firstCreated', 'headline', 'headlineTag', 'marketCommentary', 'provider', 'returnsOpenPrevRaw1_to_volume',
                                             'returnsOpenNextMktres10', 'sourceId', 'subjects', 'time', 'time_x', 'universe','sourceTimestamp']]

X = market_train[fcol].values
up = up.values
r = market_train.returnsOpenNextMktres10.values

# Scaling of X values
mins = np.min(X, axis=0)
maxs = np.max(X, axis=0)
rng = maxs - mins
X = 1 - ((maxs - X) / rng)


#splitting the data and model using XGB. I was surprised by how important parameters are
X_train, X_test, up_train, up_test, r_train, r_test = model_selection.train_test_split(X, up, r, test_size = 0.1, random_state = 99)
xgb_up = XGBClassifier(n_jobs = 4,
                      n_estimators = 300,
                      max_depth = 2,
                      eta = 0.05)

%%time
xgb_up.fit(X_train, up_train)
print('Accuracy score: ', accuracy_score(xgb_up.predict(X_test), up_test))

#finally. getting next days and submitting
import time
days = env.get_prediction_days()
n_days = 0
prep_time = 0
packaging_time = 0
prediction_time = 0
for (market_obs_df, news_obs_df, prediction_template_df) in days:
    n_days +=1
    if n_days % 50 == 0:
        print(n_days, end=' ')
    
    t = time.time()
    market_obs_df = data_prep(market_obs_df, news_obs_df)
    market_obs_df = market_obs_df[market_obs_df.assetCode.isin(prediction_template_df.assetCode)]
    X_live = market_obs_df[fcol].values
    X_live = 1 - ((maxs - X_live)/rnge)
    prep_time += time.time() - t
    
    t = time.time()
    lp = xgb_up.predict_proba(X_live)
    prediction_time += time.time() - t
    
    t = time.time()
    confidence = 2* lp[:,1] -1
    predictions = pd.DataFrame({'assetCode': market_obs_df['assetCode'], 'confidence': confidence})
    prediction_template_df = prediction_template_df.merge(predictions, 
                                                            how='left').drop('confidenceValue',
                                                            axis=1).fillna(0).rename(columns={'confidence':'confidenceValue'})
    env.predict(prediction_template_df)
    packaging_time += time.time() -t
    
env.write_submission_file()
