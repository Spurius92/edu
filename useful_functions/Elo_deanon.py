

def deanonymize_transactions(df):
    df['purchase_amount_new'] = np.round(df['purchase_amount'] / 0.00150265118 + 497.07, 2)
   # df['']
    return df

def deanonymize_merchants(df):
    df['numerical_1'] = np.round(df['numerical_1'] / 0.009914905 + 5.79639, 0)
    df['numerical_1'] = np.round(df['numerical_1'] / 0.009914905 + 5.79639, 0)

    return df

def target(df):
    df['target_new'] = 10**(df['target']*np.log10(2))
    return df
