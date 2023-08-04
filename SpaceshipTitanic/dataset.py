# -*- coding: utf-8 -*-#
# -------------------------------------------------------------------------------
# Name:         dataset.py
# Description:
# Author:       Lv
# Date:         2023/8/3
# -------------------------------------------------------------------------------
import pandas as pd
import torch
from torch.utils import data
from utils import VALUE


# VALUE = 'Transported'  # label
class STDataset(data.Dataset):
    def __init__(self, df):
        self.data = df
        self.label = torch.tensor(self.data[VALUE].values)
        features_df = self.data.drop(columns=VALUE).copy()
        # print('features_df.keys():', features_df.keys())
        self.features = torch.tensor(features_df.values, dtype=torch.float32)
        # print('self.features.shape:', self.features.shape)
        # print('self.label.shape:', self.label.shape)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.features[item], self.label[item]
