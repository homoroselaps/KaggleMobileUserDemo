# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from datetime import datetime

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
        "聆韵": "lingyun"
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

def build_event_count(train, test, events):
    '''
    number of events per device
    '''
    def event_count(dev_id):
        return 1
    
    event_counts = []
    for dev_id in train["device_id"].values:
        event_counts.append(event_count(dev_id))
    
def build_features(train, test, phone_brand_device_model, apps, app_events, events):
    train_out = train.drop(["gender", "age"], axis=1)
    test_out = test
    
    #train_out, test_out = build_event_count()
    build_active_time(train, events)
    
    return train_out, test_out
    
if __name__ == "__main__":
    gender_age_train, gender_age_test, phone_brand_device_model, apps, app_events, events = load_data()
    
    build_features(gender_age_train, gender_age_test, phone_brand_device_model, apps, app_events, events)