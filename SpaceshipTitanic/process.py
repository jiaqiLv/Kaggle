# -*- coding: utf-8 -*-#
# -------------------------------------------------------------------------------
# Name:         process.py
# Description:  数据预处理
# Author:       Lv
# Date:         2023/8/3
# -------------------------------------------------------------------------------

import pandas as pd
import os
import numpy as np

RAW_DATA_PATH = "./data/raw_data"
PROCESSED_DATA_PATH = "./data/processed"
# deprecate Name,Cabin,PassengerId
KEY = ['HomePlanet', 'CryoSleep', 'Destination', 'Age',
       'VIP', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
VALUE = 'Transported'  # label
file_names = ['train.csv', 'test.csv']

for file_name in file_names:
    train_df = pd.read_csv(os.path.join(RAW_DATA_PATH, file_name))
    train_df.drop(columns='Name', inplace=True)
    train_df.drop(columns='Cabin', inplace=True)
    train_df.drop(columns='PassengerId', inplace=True)
    train_df.dropna(inplace=True)
    for key in KEY:
        if isinstance(train_df[key][0], str):
            category = train_df[key].unique()
            # print(f'{key} category:', category)
            # print('len(category):', len(category))
            train_df = pd.get_dummies(train_df, columns=[key])  # 独热编码
        elif isinstance(train_df[key][0], (bool, np.bool_)):
            train_df[key] = train_df[key].astype(int)
            # print(train_df[key])
        elif isinstance(train_df[key][0], (np.float64, float)):
            train_df[key] = (train_df[key] - train_df[key].mean()) / train_df[key].std()
            # print(train_df[key])
        else:
            raise ValueError('Invalid data type!')

    if file_name == 'train.csv':
        train_df[VALUE] = train_df[VALUE].astype(int)
        col_to_move = train_df.pop(VALUE)
        train_df[VALUE] = col_to_move

    print('train_df.keys():', train_df.keys())

    """将处理后的数据写入新文件中"""
    train_df.to_csv(os.path.join(PROCESSED_DATA_PATH, file_name), index=False)
