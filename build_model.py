import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import StratifiedKFold, cross_val_score
from sklearn.ensemble.forest import RandomForestClassifier

def load_data():
    train = pd.read_csv("data/train_feat.csv")
    test = pd.read_csv("data/test_feat.csv")
    return train, test


if __name__ == "__main__":
    train, test = load_data()
    
    features = ["phone_brand", "device_model",  "event_count", "action_radius_max", "medianTime", "minTime", "maxTime", "weekday", "appcounts1"]
    encoder = LabelEncoder()
    train["group"] = encoder.fit_transform(train["group"].values)
    
    params = 
    rf = RandomForestClassifier(n_estimators=200, max_depth=15, max_features=6, bootstrap=True, n_jobs=6, random_state=2016, class_weight=None)
    
    skf = StratifiedKFold(train["group"].values, n_folds=5, shuffle=True, random_state=2016)
    scores = cross_val_score(rf, train[features].values, train["group"].values, scoring="log_loss", cv=skf, n_jobs=1)
    print(scores)
    print("RF Score: %0.5f" %(-scores.mean())) # RF Score: 2.39867