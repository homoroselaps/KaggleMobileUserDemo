# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from datetime import datetime
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
        "华为": "huawei",
        "世纪天元": "tianyuan",
        "世纪星": "unknown",
        "中兴": "zte",
        "丰米": "wahot",
        "索尼" : "Sony",
        "欧博信" : "Opssom",
        "奇酷" : "Qiku",
        "酷比" : "CUBE",
        "康佳" : "Konka",
        "亿通" : "Yitong",
        "金星数码" : "JXD",
        "至尊宝" : "Monkey King",
        "百立丰" : "Hundred Li Feng",
        "贝尔丰" : "Bifer",
        "百加" : "Bacardi",
        "诺亚信" : "Noain",
        "广信" : "Kingsun",
        "世纪天元" : "Ctyon",
        "青葱" : "Cong",
        "果米" : "Taobao",
        "斐讯" : "Phicomm",
        "长虹" : "Changhong",
        "欧奇" : "Oukimobile",
        "先锋" : "XFPLAY",
        "台电" : "Teclast",
        "大Q" : "Daq",
        "蓝魔" : "Ramos",
        "奥克斯" : "AUX" 
}

epoch = datetime.utcfromtimestamp(0)

def unix_time_seconds(dt):
    return (dt - epoch).total_seconds()

def convertTimestamp(value):
    return unix_time_seconds(datetime.strptime(value, '%Y-%m-%d %H:%M:%S'))

def load_data():
    print("Load Data")
    gender_age_train = pd.read_csv("data/gender_age_train.csv")
    gender_age_test = pd.read_csv("data/gender_age_test.csv")
    
    phone_brand_device_model = pd.read_csv("data/phone_brand_device_model.csv")
    phone_brand_device_model = phone_brand_device_model.replace({"phone_brand": brands})
    
    app_labels = pd.read_csv("data/app_labels.csv")
    label_categories = pd.read_csv("data/label_categories.csv")
    apps = app_labels.merge(label_categories, on="label_id", how="left")
    
    app_events = pd.read_csv("data/app_events_small.csv")
    
    events = pd.read_csv("data/events_small.csv").head()
    print("Load Data done")
    print("Convert Data")
    events['timestamp'] = events['timestamp'].apply(convertTimestamp)
    print("Convert Done")

    print("done")
    return gender_age_train, gender_age_test, phone_brand_device_model, apps, app_events, events

def build_active_time(train, events):
    def extractDayOfWeek(timestamp):
        date = datetime.utcfromtimestamp(timestamp)
        return date.weekday()
    def extractTime(timestamp):
        time = datetime.utcfromtimestamp(timestamp).time()
        return time.hour*60*60 + time.minute*60 + time.second
    def mode(x):
        return x.mode() if len(x) > 2 else x.values[0]
    events['weekday'] = events['timestamp'].apply(extractDayOfWeek)
    events['time'] = events['timestamp'].apply(extractTime)
    did = events[['device_id', 'weekday', 'time']].groupby(['device_id'])
    weekday = did['weekday'].agg(mode)
    result = did['time'].agg(['median', 'max', 'min'])
    result['weekday'] = weekday.values
    return result

def build_event_count(df, events):
    '''
    number of events per device
    '''
    event_counts = pd.DataFrame({'count' : events[["device_id","event_id"]][events["device_id"].isin(df["device_id"])].groupby("device_id").size()}).reset_index()
    tmp = df.merge(event_counts, on="device_id", how="left").fillna(0.0)
    return np.array(tmp["count"].values)    

def action_distance(df, events):
    '''
    maximual euclidian distance between the events
    '''
    from scipy.spatial.distance import pdist, squareform
    max_dist_list = []
    grouped = events.groupby("device_id")[["longitude","latitude"]]
    for dev_id in df["device_id"].values:
        try:
            A = grouped.get_group(dev_id).values.transpose()
            D = squareform(pdist(A))
            max_dist_list.append(np.max(D))
        except:
            max_dist_list.append(0.0)
    return np.array(max_dist_list)

def map_column(table, f):
    labels = sorted(table[f].unique())
    mappings = dict()
    for i in range(len(labels)):
        mappings[labels[i]] = i
    table = table.replace({f: mappings})
    return table

def build_features(train, test, phone_brand_device_model, apps, app_events, events):
    print("BUILD FEATURES...")
    train_out = train.drop(["gender", "age"], axis=1)
    test_out = test

    # add brand features
    phone_brand_device_model.drop_duplicates('device_id', keep='first', inplace=True)
    phone_brand_device_model = map_column(phone_brand_device_model, 'phone_brand')
    phone_brand_device_model = map_column(phone_brand_device_model, 'device_model')
    train_out = pd.merge(train_out, phone_brand_device_model, how='left', on='device_id', left_index=True)
    test_out = pd.merge(test_out, phone_brand_device_model, how='left', on='device_id', left_index=True)
    
    # add event count
    train_out["event_count"] = build_event_count(train_out, events)
    test_out["event_count"] = build_event_count(test_out, events)

    # add max action distance
    train_out["action_radius_max"] = action_distance(train_out, events)
    test_out["action_radius_max"] = action_distance(test_out, events)
    
    # add app count
    app = pd.read_csv("../input/app_events.csv", dtype={'device_id': np.str})
    app['appcounts'] = app.groupby(['event_id'])['app_id'].transform('count')
    app_small = app[['event_id', 'appcounts']].drop_duplicates('event_id', keep='first')
    e1=pd.merge(events, app_small, how='left', on='event_id', left_index=True)
    e1.loc[e1.isnull()['appcounts'] ==True, 'appcounts']=0
    e1['appcounts1'] = e1.groupby(['device_id'])['appcounts'].transform('sum')
    e1_small = e1[['device_id', 'appcounts1']].drop_duplicates('device_id', keep='first')
    train_out = pd.merge(train_out, e1_small, how='left', on='device_id', left_index=True)
    train_out.fillna(-1, inplace=True)
    test_out = pd.merge(test_out, e1_small, how='left', on='device_id', left_index=True)
    test_out.fillna(-1, inplace=True)
    
    print("done")
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
    features = ["phone_brand",  "event_count", "action_radius_max"]
    encoder = LabelEncoder()
    train["group"] = encoder.fit_transform(train["group"].values)
    
    rf = RandomForestClassifier(n_estimators=50, max_depth=7, max_features=2, bootstrap=True, n_jobs=4, random_state=2016, class_weight=None)
    
    rf.fit(train[features].values, train["group"].values)
    feature_importance(rf, features)
    
    skf = StratifiedKFold(train["group"].values, n_folds=5, shuffle=True, random_state=2016)
    scores = cross_val_score(rf, train[features].values, train["group"].values, scoring="log_loss", cv=skf, n_jobs=1)
    print(scores)
    print("RF Score: %0.5f" %(-scores.mean())) # RF Score: 2.39884
    
if __name__ == "__main__":
    gender_age_train, gender_age_test, phone_brand_device_model, apps, app_events, events = load_data()
    
    train_out, test_out = build_features(gender_age_train, gender_age_test, phone_brand_device_model, apps, app_events, events)
    
    print(train_out.head(10))
    
    try_model(train_out)