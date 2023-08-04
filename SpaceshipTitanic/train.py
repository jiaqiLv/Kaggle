# -*- coding: utf-8 -*-#
# -------------------------------------------------------------------------------
# Name:         train.py
# Description:
# Author:       Lv
# Date:         2023/8/3
# -------------------------------------------------------------------------------

import torch
from torch import nn
from torch.utils import data
import pandas as pd
import os
from utils import VALUE
from dataset import STDataset
from sklearn.model_selection import train_test_split

DATA_PATH = "./data/processed"
BATCH_SIZE = 64
INPUT_NUM = 14
OUTPUT_NUM = 2
EPOCH_NUM = 1000

# VALUE = 'Transported'  # label


if __name__ == "__main__":
    df = pd.read_csv(os.path.join(DATA_PATH, 'train.csv'))
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    test_label = torch.tensor(test_df[VALUE].values)
    test_df.drop(columns=VALUE, inplace=True)
    # print('test_df:', test_df)
    test_features = torch.tensor(test_df.values, dtype=torch.float32)
    # print('test_df.keys():', test_df.keys())
    # print('train_df.keys():', train_df.keys())
    # test_df = pd.read_csv(os.path.join(DATA_PATH, 'test.csv'))
    train_iter = data.DataLoader(STDataset(train_df), BATCH_SIZE, shuffle=True)
    # test_iter = data.DataLoader(STDataset(test_df), BATCH_SIZE, shuffle=True)
    # w = torch.normal(0, 0.01, size=(INPUT_NUM, OUTPUT_NUM), requires_grad=True)
    # b = torch.zeros(OUTPUT_NUM, requires_grad=True)
    """定义模型"""
    model = nn.Sequential(nn.Linear(INPUT_NUM, OUTPUT_NUM))
    """初始化模型参数"""
    model[0].weight.data.normal_(0, 0.01)
    model[0].bias.data.fill_(0)
    """损失函数"""
    criterion = nn.CrossEntropyLoss(reduction='none')
    """优化器"""
    optim = torch.optim.SGD(model.parameters(), lr=0.01)
    """Train"""
    for epoch in range(EPOCH_NUM):
        for X, y in train_iter:
            # print('X.shape:', X.shape)
            # print('y.shape:', y.shape)
            train_loss = criterion(model(X), y)
            optim.zero_grad()
            train_loss.sum().backward()
            optim.step()
        # print('test_features.shape:', test_features.shape)
        # print('test_label.shape:', test_label.shape)
        test_loss = criterion(model(test_features), test_label)
        print('test_loss:', test_loss.sum())
