import numpy as np
import matplotlib.pyplot as plt
from preprocess import *
from dataset import *
import torch
from sklearn.model_selection import train_test_split
import pandas
from torch.utils.data import DataLoader
from model import *
from torch.optim import Adam, AdamW
import torch.nn as nn
from tqdm import tqdm
import random
import sys

def evaluate(model, loader):
    loss_func = nn.CrossEntropyLoss(reduction='sum')
    total_loss = 0
    acc = 0
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            output = model(x)
            loss = loss_func(output, y)
            loss = np.array(loss)
            total_loss += loss.item()
            for i, sample in enumerate(output):
                if torch.argmax(sample) == y[i]:
                    acc += 1
    loss = total_loss / len(loader.dataset)
    acc = acc / len(loader.dataset)
    return loss, acc


def train(model, train_loader, val_loader, epochs):
    loss_func = nn.CrossEntropyLoss(reduction='sum')
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    for epoch in range(epochs):
        print('epoch', epoch + 1)
        model.train()
        for x, y in tqdm(train_loader):
            optimizer.zero_grad()  # zero the gradient buffers
            output = model(x)
            loss = loss_func(output, y)
            loss.backward()
            optimizer.step()
        train_loss, train_acc = evaluate(model, train_loader)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        print('train loss:', train_loss, 'train acc: ', train_acc)
        val_loss, val_acc = evaluate(model, val_loader)
        print('val loss:', val_loss, 'val acc: ', val_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
    plot(train_losses, train_accs, val_losses, val_accs, 'plot.png')


def eval_lstm_v1(model, loader):
    model.eval()
    loss_func = nn.CrossEntropyLoss(reduction='sum', ignore_index=2)
    total_loss = 0
    acc = 0
    with torch.no_grad():
        for x, y, lens in loader:
            x, y = x, y
            y = y[:, :max(lens)]
            output = model(x, lens).permute(0,2,1)
            loss = loss_func(output, y)
            loss = np.array(loss)
            total_loss += loss.item()
            output = output.data.max(dim=1)[1]
            for i, sample in enumerate(output):
                stopping_pos_pred = (sample == 1).nonzero(as_tuple=True)[0]
                if stopping_pos_pred.size(0) != 0:
                    if stopping_pos_pred[0] == lens[i] - 1:
                        acc += 1
    loss = total_loss / len(loader.dataset)
    acc = acc / len(loader.dataset)
    return loss, acc


def train_lstm_v1(model, train_loader, val_loader, epochs):
    model.train()
    optimizer = Adam(model.parameters(), lr=1e-4)
    loss_func = nn.CrossEntropyLoss(reduction='sum', ignore_index=2)
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    for epoch in range(epochs):
        model.train()
        print('epoch', epoch + 1)
        for x, y, lens in tqdm(train_loader):
            x, y = x, y
            optimizer.zero_grad()  # zero the gradient buffers
            output = model(x,lens).permute(0,2,1)
            y = y[:,:max(lens)]
            loss = loss_func(output, y)
            loss.backward()
            optimizer.step()
        train_loss, train_acc = eval_lstm_v1(model, train_loader)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        print('train loss:', train_loss, 'train acc: ', train_acc)
        val_loss, val_acc = eval_lstm_v1(model, val_loader)
        print('val loss:', val_loss, 'val acc: ', val_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
    plot(train_losses, train_accs, val_losses, val_accs, 'plot.png')



def plot(train_loss, train_acc, val_loss, val_acc, name):
    fig, axs = plt.subplots(2)
    axs[0].plot(train_loss)
    axs[0].grid(color="w")
    axs[0].set_facecolor('xkcd:light gray')
    axs[0].set_title("Loss")
    axs[0].plot(train_loss, color='red')
    axs[0].plot(val_loss, color='blue')
    axs[1].set_title("Acc")
    axs[1].grid(color="w")
    axs[1].set_facecolor('xkcd:light gray')
    axs[1].plot(train_acc, color='red')
    axs[1].plot(val_acc, color='blue')
    fig.tight_layout()
    plt.savefig(name)


if __name__ == '__main__':
    flag = str(sys.argv[1])
    data, offers = read_data(str(sys.argv[2]))
    offers = normalized_offers(offers)
    
    if flag == "MLP":
        training_set, validation_set = train_test_split(data, test_size=0.2)
        train_data = Dataset(training_set, offers)
        val_data = Dataset(validation_set, offers)
        model = MLP()
        optimizer = Adam(model.parameters(), lr=1e-4)
        train_loader = DataLoader(train_data, batch_size=4, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=1, shuffle=False)
        train(model, train_loader, val_loader, 25)

    elif flag == "LSTM_V2":
        data, y = create_data_for_LSTM_V2(data)
        data = np.array(data)
        y = np.array(y)
        data_idx = [i for i in range(len(data))]
        random.seed(42)
        train_idx = random.sample(data_idx, int(len(data) * 0.8))
        valid_idx = list(set(data_idx) - set(train_idx))
        training_set = data[train_idx]
        y_train = y[train_idx]
        validation_set = data[valid_idx]
        y_test = y[valid_idx]
        train_data = Dataset_for_LSTM_V2(training_set, offers, y_train)
        val_data = Dataset_for_LSTM_V2(validation_set, offers, y_test)
        model = LSTM_V2()
        optimizer = AdamW(model.parameters(), lr=1e-4)
        train_loader = DataLoader(train_data, batch_size=1, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=1)
        train(model, train_loader, val_loader, 50)

    else:
        training_set, validation_set = train_test_split(data, test_size=0.2)
        train_data = Dataset_for_LSTM_V1(training_set, offers)
        val_data = Dataset_for_LSTM_V1(validation_set, offers)
        train_loader = DataLoader(train_data, batch_size=4, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=4, shuffle=False)
        model = LSTM_V1()
        optimizer = Adam(model.parameters(), lr=1e-4)
        train_lstm_v1(model, train_loader, val_loader, 50)
