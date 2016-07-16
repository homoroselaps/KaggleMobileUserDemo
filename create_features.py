# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from sklearn.ensemble.forest import RandomForestClassifier
from sklearn.cross_validation import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder

brands = {
        "三星": "samsung",
        "天语 ": "Ktouch",
        "海信": "hisense",
        "联想": "lenovo",
        "欧比": "obi",
        "爱派尔": "ipair",
        "努比亚": "nubia",
        "优米": "youmi",
        "朵唯": "dowe",
        "黑米": "heymi",
        "锤子": "hammer",
        "酷比魔方": "koobee",
        "美图": "meitu",
        "尼比鲁": "nibilu",
        "一加": "oneplus",
        "优购": "yougo",
        "诺基亚": "nokia",
        "糖葫芦": "candy",
        "中国移动": "ccmc",
        "语信": "yuxin",
        "基伍": "kiwu",
        "青橙": "greeno",
        "华硕": "asus",
        "夏新": "panosonic",
        "维图": "weitu",
        "艾优尼": "aiyouni",
        "摩托罗拉": "moto",
        "乡米": "xiangmi",
        "米奇": "micky",
        "大可乐": "bigcola",
        "沃普丰": "wpf",
        "神舟": "hasse",
        "摩乐": "mole",
        "飞秒 ": "fs",
        "米歌": "mige",
        "富可视": "fks",
        "德赛": "desci",
        "梦米": "mengmi",
        "乐视": "lshi",
        "小杨树": "smallt",
        "纽曼": "newman",
        "邦华": "banghua",
        "E派": "epai",
        "易派": "epai",
        "普耐尔": "pner",
        "欧新": "ouxin",
        "西米": "ximi",
        "海尔": "haier",
        "波导": "bodao",
        "糯米": "nuomi",
        "唯米": "weimi",
        "酷珀": "kupo",
        "谷歌": "google",
        "昂达": "ada",
        "聆韵": "lingyun",
        "小米": "xiaomi",
        "酷派": "coolpad",
        "华为": "huawei"
        
}

def load_data():
    gender_age_train = pd.read_csv("data/gender_age_train.csv")
    gender_age_test = pd.read_csv("data/gender_age_test.csv")
    
    phone_brand_device_model = pd.read_csv("data/phone_brand_device_model.csv")
    phone_brand_device_model = phone_brand_device_model.replace({"phone_brand": brands})
    
    app_labels = pd.read_csv("data/app_labels.csv")
    label_categories = pd.read_csv("data/label_categories.csv")
    apps = app_labels.merge(label_categories, on="label_id", how="left")
    
    app_events = pd.read_csv("data/app_events.csv")
    
    events = pd.read_csv("data/events.csv")
    
    return gender_age_train, gender_age_test, phone_brand_device_model, apps, app_events, events

def build_event_count(df, events):
    '''
    number of events per device
    '''
    #event_counts = events[["device_id","event_id"]][events["device_id"].isin(df["device_id"])].groupby("device_id").agg("count")
    event_counts = pd.DataFrame({'count' : events[["device_id","event_id"]][events["device_id"].isin(df["device_id"])].groupby("device_id").size()}).reset_index()
    tmp = df.merge(event_counts, on="device_id", how="left").fillna(0.0)
    return np.array(tmp["count"].values)    

def build_features(train, test, phone_brand_device_model, apps, app_events, events):
    train_out = train.drop(["gender", "age"], axis=1)
    test_out = test
    
    # add brand features
    train_out = train_out.merge(phone_brand_device_model[["device_id","phone_brand"]], on="device_id", how="left")
    test_out = test_out.merge(phone_brand_device_model[["device_id","phone_brand"]], on="device_id", how="left")
    
    # add event count
    train_out["event_count"] = build_event_count(train_out, events)
    test_out["event_count"] = build_event_count(test_out, events)
    
    # add longitude and latitude
    #train_out = train_out.merge(events[["device_id","longitude","latitude"]], on="device_id", how="left").fillna(-999)
    #test_out = test_out.merge(events[["device_id","longitude","latitude"]], on="device_id", how="left").fillna(-999)
    
    return train_out, test_out

def feature_importance(clf,feature_names=[]):
    '''
    print importance of the features
    '''
    print()
    print("Feature importance of the fitted model")
    print()
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]
    for f in range(len(feature_names)):
        if feature_names != []:
            print("%s : (%f)" % (feature_names[indices[f]], importances[indices[f]]))
        else:
            print("(%f)" % (importances[indices[f]]))
    print()
  
def try_model(train):
    print(train.shape)
    features = ["phone_brand",  "event_count"]
    encoder = LabelEncoder()
    train["phone_brand"] = encoder.fit_transform(train["phone_brand"].values)
    train["group"] = encoder.fit_transform(train["group"].values)
    
    rf = RandomForestClassifier(n_estimators=50, max_depth=7, max_features=2, bootstrap=True, n_jobs=4, random_state=2016, class_weight=None)
    
    rf.fit(train[features].values, train["group"].values)
    feature_importance(rf, features)
    
    skf = StratifiedKFold(train["group"].values, n_folds=5, shuffle=True, random_state=2016)
    scores = cross_val_score(rf, train[features].values, train["group"].values, scoring="log_loss", cv=skf, n_jobs=1)
    print(scores)
    print("RF Score: %0.5f" %(-scores.mean()))
    
if __name__ == "__main__":
    gender_age_train, gender_age_test, phone_brand_device_model, apps, app_events, events = load_data()
    
    train_out, test_out = build_features(gender_age_train, gender_age_test, phone_brand_device_model, apps, app_events, events)
    
    print(train_out.head(10))
    
    # try_model(train_out)