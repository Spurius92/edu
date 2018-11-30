# Encoder:
lbl = LabelEncoder()
for col in cat_cols:
    data[col] = lbl.fit_transform(data[col].astype(str))


# Prepare Categorical Variables for catboost
def column_index(data, query_cols):
    cols = data.columns.values
    sidx = np.argsort(cols)
    return sidx[np.searchsorted(cols,query_cols,sorter=sidx)]
categorical_pos = column_index(data, cat_cols)

#prepared data for splitting
data = data.drop(labels=['date', 'visitId', 'fullVisitorId', 'visitStartTime'], axis=1)
y = data['totals.transactionRevenue']

X_train, X_valid, y_train, y_valid = train_test_split(
    data, y, test_size=0.20, random_state=15)


model = CatBoostRegressor(iterations=200,
                             learning_rate=0.01,
                             eval_metric='RMSE',
                             random_seed = 15,
                             bagging_temperature = 0.1)
                             #od_type='Iter',
                             #metric_period = 75,
                             #od_wait=100)


model.fit(X_train, y_train,
             eval_set=(X_valid,y_valid),
             cat_features=categorical_pos,
             use_best_model=True)

test = data.tail(test_len)
catpred = model.predict(test)
catsub = pd.DataFrame({'fullVisitorId': test_id, 'PredictedLogRevenue': catpred})
catsub.to_csv("catsub.csv",index=True,header=True)
