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


data = []   #vis using plotly   prep the data for plot
for asset in np.random.choice(market_train['assetName'].unique(),10):
    asset_df = market_train[(market_train['assetName'] == asset)]
    data.append(go.Scatter(
    x = asset_df['time'].dt.strftime(date_format = '%Y-%m-%d').values,
    y = asset_df['close'].values,
    name = asset
))
layout = go.Layout(dict(title = 'Closing values of 10 random stocks',   #part dedicated to the plot
                      xaxis = dict(title = 'Timeframe'),
                      yaxis = dict(title = 'prices(USD)'),
                      ), legend = dict(orientation='h'))
py.iplot(dict(data = data, layout= layout))

#checking data for outliers
market_train['price_diff'] = market_train['close'] - market_train['open']  
market_train.sort_values('price_diff')
market_train['close_to_open'] = np.abs(market_train['close']) / (market_train['open'])
print((market_train['close_to_open'] >= 2).sum())
print((market_train['close_to_open'] <=0.5).sum())

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

market_train.head()


#wordcloud. finally, i know where those pictures come from
text = ' '.join(news_train['headline'].str.lower().values[-100000:])
wordcloud = WordCloud(max_font_size = None, stopwords = stop, background_color = 'white',
                     width = 1200, height = 1000).generate(text)
plt.figure(figsize=(12, 8))
plt.imshow(wordcloud)
plt.title('Top words in headline')
plt.axis('off')
plt.show()


#another plotly example. time and changes in prices
market_train['price_diff'] = market_train['close'] - market_train['open']
grouped = market_train.groupby(['time']).agg({'price_diff':['std', 'min']}).reset_index()
g = grouped.sort_values(('price_diff', 'std'), ascending=False)[:7]
g['min_text']= 'Maximum price drop: ' + (-1 * np.round(g['price_diff']['min'], 2)).astype(str)
trace = go.Scatter(
    x = g['time'].dt.strftime(date_format='%Y-%m-%d').values,
    y = g['price_diff']['std'].values,
    mode='markers',
    marker = dict(
        size = g['price_diff']['std'].values *5,
        color = g['price_diff']['std'].values,
        colorscale = 'Portland',
        showscale = True   
    ),
    text = g['min_text'].values
)
data = [trace]
layout= go.Layout(
    autosize= True,
    title= 'Top 10 months by standard deviation of price change within a day',
    hovermode= 'closest',
    yaxis=dict(
        title= 'price_diff',
        ticklen= 5,
        gridwidth= 2,
    ),
    showlegend= False
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig,filename='scatter2010')


#sentiments in news data
for i, j in zip([-1, 0 ,1], ['positive', 'neutral', 'negative']):
    sentiment = news_train.loc[news_train['sentimentClass'] == i, 'assetName']
    print(f'Top mentioned companies for {j} sentiment are:')
    print(sentiment.value_counts().head(20))
    print(' ')




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
