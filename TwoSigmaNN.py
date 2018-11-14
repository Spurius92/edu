from kaggle.competitions import twosigmanews
env = twosigmanews.make_env()
(market, news) = env.get_training_data()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split
#market = market.dropna()
cat_cols = ['assetCode']
num_cols = ['volume', 'close', 'open', 'returnsClosePrevRaw1', 'returnsOpenPrevRaw1', 'returnsClosePrevMktres1',
                    'returnsOpenPrevMktres1', 'returnsClosePrevRaw10', 'returnsOpenPrevRaw10', 'returnsClosePrevMktres10',
                    'returnsOpenPrevMktres10']
market_train, market_val = train_test_split(market.index.values, test_size= 0.25, random_state=20)

#some work with categorical values
#encoder is useful. standard encoder doesn't work because there are new assets in the news data
def encode(encoder, x):
    len_encoder = len(encoder)
    try:
        id = encoder[x]
    except KeyError:
        id = len_encoder
    return id

encoders = [{}for cat in cat_cols]

for i, cat in enumerate(cat_cols):
    print('encoding %s ...' % cat, end=' ')
    encoders[i] = {l: id for id, l in enumerate(market.loc[market_train, cat].astype(str).unique())}
    market[cat] = market[cat].astype(str).apply(lambda x: encode(encoders[i], x))
    print('Done!')
embed_sizes = [len(encoder) +1 for encoder in encoders]  # +1 for possible unknown assets

#encoder for numerical values
from sklearn.preprocessing import StandardScaler
market[num_cols] = market[num_cols].fillna(0)
scaler = StandardScaler()
market[num_cols] = scaler.fit_transform(market[num_cols])

#define NN architecture
from keras.models import Model
from keras.layers import Embedding, Input, Dense, Concatenate, Flatten, BatchNormalization
from keras.losses import binary_crossentropy, mse

categorical_inputs = []
for cat in cat_cols:
    categorical_inputs.append(Input(shape=[1], name=cat))
    
categorical_embeddings = []
for i, cat in enumerate(cat_cols):
    categorical_embeddings.append(Embedding(embed_sizes[i], 10)(categorical_inputs[i]))

categorical_logits = Flatten()(categorical_embeddings[0])
categorical_logits = Dense(32, activation='relu')(categorical_logits)

numerical_inputs = Input(shape=(11,), name = 'num')
numerical_logits = numerical_inputs
numerical_logits = BatchNormalization()(numerical_logits)

numerical_logits = Dense(128, activation='relu')(numerical_logits)
numerical_logits = Dense(64, activation='relu')(numerical_logits)

logits = Concatenate()([numerical_logits, categorical_logits])
logits = Dense(64, activation='relu')(logits)
out = Dense(1, activation='sigmoid')(logits)

model = Model(inputs = categorical_inputs + [numerical_inputs], outputs=out)
model.compile(optimizer='adam', loss= binary_crossentropy)
model.summary()

#
def get_input(market_train, indices):
    X_num = market.loc[indices, num_cols].values
    X = {'num': X_num}
    for cat in cat_cols:
        X[cat] = market.loc[indices, cat_cols].values
        y = (market.loc[indices, 'returnsOpenNextMktres10'] >= 0).values
        r = market.loc[indices, 'returnsOpenNextMktres10'].values
        u = market.loc[indices, 'universe']
        d = market.loc[indices, 'time'].dt.date
        return X, y, r, u, d
X_train,y_train,r_train,u_train,d_train = get_input(market, market_train)
X_valid,y_valid,r_valid,u_valid,d_valid = get_input(market, market_val)


#changed epochs from 2 to 3. score improved 0.65 to 0.67
from keras.callbacks import EarlyStopping, ModelCheckpoint
check_point = ModelCheckpoint('model.hdf5', verbose=True, save_best_only=True)
early_stop = EarlyStopping(patience=5, verbose=True)
model.fit(X_train, y_train.astype('int'),
          validation_data = (X_valid, y_valid.astype('int')),
          epochs = 3,
          callbacks= [early_stop, check_point])


model.load_weights('model.hdf5')
confidence_valid = model.predict(X_valid)[:,0]*2 -1
print(accuracy_score(confidence_valid>0, y_valid))
plt.hist(confidence_valid, bins='auto')
plt.title('predicted confidence')


r_valid = r_valid.clip(-1, 1)
x_t_i = confidence_valid * r_valid * u_valid
data = {'day': d_valid, 'x_t_i': x_t_i}
df = pd.DataFrame(data)
x_t = df.groupby('day').sum().values.flatten()
mean = np.mean(x_t)
std = np.std(x_t)
score_valid = mean / std
print(score_valid)


days = env.get_prediction_days()
n_days = 0
prep_time = 0
packaging_time = 0
prediction_time = 0
predicted_confidences = np.array([])
for (market_obs_df, news_obs_df, prediction_template_df) in days:
    n_days +=1
    print(n_days, end=' ')
    
    t = time.time()
    market_obs_df['assetCodeEncoded'] = market_obs_df[cat].astype(str).apply(lambda x: encode(encoders[i],x))
    
    market_obs_df[num_cols] = market_obs_df[num_cols].fillna(0)
    market_obs_df[num_cols] = scaler.transform(market_obs_df[num_cols])
    X_live = market_obs_df[num_cols].values
    X_test = {'num': X_live}
    X_test['assetCode'] = market_obs_df['assetCodeEncoded'].values 
    prep_time += time.time() - t
    
    t = time.time()
    market_prediction = model.predict(X_test)[:,0]*2 -1
    predicted_confidences = np.concatenate((predicted_confidences, market_prediction))
    prediction_time += time.time() - t
    
    t = time.time()
    preds = pd.DataFrame({'assetCode': market_obs_df['assetCode'], 'confidence': market_prediction})
    prediction_template_df = prediction_template_df.merge(preds, 
                                                            how='left').drop('confidenceValue',
                                                            axis=1).fillna(0).rename(columns={'confidence':'confidenceValue'})
    env.predict(prediction_template_df)
    packaging_time += time.time() -t
    
env.write_submission_file()
