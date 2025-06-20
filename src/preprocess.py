from sklearn.preprocessing import StandardScaler

def preprocess_features(df):
    X = df.drop(['Class', 'Time'], axis=1)
    y = df['Class']

    scaler = StandardScaler()
    X['Amount'] = scaler.fit_transform(X[['Amount']])

    return X, y, scaler
