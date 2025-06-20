from sklearn.ensemble import RandomForestClassifier

def get_model(n_estimators=100, random_state=42):
    clf = RandomForestClassifier(n_estimators=n_estimators,
                                 random_state=random_state,
                                 n_jobs=-1)
    return clf
