import numpy as np
import torch
import os
from preprocess import normalized_offers, read_data
from random import shuffle
import random
import csv
from tqdm import tqdm
import pandas
import torch.nn as nn


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data, offers):
        self.data = data
        self.offers = offers

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        line = self.data.iloc[[index]]
        round_num = int(line['rounds'])
        round = torch.zeros(10)
        round[round_num - 1] = 1
        info = torch.tensor(np.array(line).reshape(-1)[3:])
        offer = torch.tensor(self.offers[round_num - 1])
        x = torch.cat((info, round, offer))
        y = int(line['stopping_position']) - 1
        return x.float(), y

class Dataset_for_LSTM_V1(torch.utils.data.Dataset):
    def __init__(self, data, offers):
        self.data = data
        self.offers = offers

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        line = self.data.iloc[[index]]
        round_num = int(line['rounds'])
        round = torch.zeros(10)
        round[round_num - 1] = 1
        info = torch.tensor(np.array(line).reshape(-1)[3:])
        offer = torch.tensor(self.offers[round_num - 1])
        y = int(line['stopping_position']) - 1
        x = [torch.cat((info, round, torch.tensor([offer[i]]))) for i in range(y + 1)]
        x = torch.stack(x)
        x = torch.cat((x,torch.full(size=(20 - (y +1),16), fill_value= 0,dtype=torch.float64)), dim=0)
        y_s = torch.zeros(20,dtype=torch.long)
        y_s[y] = 1
        y_s[y+1:] =2
        return x.float(), torch.LongTensor(y_s), y + 1


class Dataset_for_LSTM_V2(torch.utils.data.Dataset):
    def __init__(self, data, offers, Y):
        self.data = data
        self.offers = offers
        self.y = Y

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        line = self.data[index]
        info = torch.tensor(np.array(line))
        offer = torch.tensor(self.offers)
        x = torch.zeros((info.shape[0], info.shape[1] + offer.shape[1]))
        x[:, :info.shape[1]] += info[:, :]
        x[:, info.shape[1]:] += offer[:, :]
        y = torch.tensor(self.y[index], dtype=torch.long)
        return x.float(), y
