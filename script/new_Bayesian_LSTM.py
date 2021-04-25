import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy import array
from torch.autograd import Variable
from blitz.modules import BayesianLSTM
from blitz.utils import variational_estimator
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

torch.manual_seed(100)
N_STEPS = 7 # 用多少天/周预测一天/周
N_FEATURES = 1 # 用来预测的总变量数，如果同时适用销量和定价来预测销量的话这里改成2，以此类推
PRED_DAYS = 7 # 预测多少天
NUM_EPOCH = 2000
LEARNING_RATE = 0.01


def splitSequence(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the sequence
        if end_ix > len(sequence)-1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)


def readData(tableName, columnName):
    date_read = pd.read_csv(tableName)
    dataset = date_read[columnName].values
    dataset = dataset.astype('float32')
    maxValue = np.max(dataset)  # 获得最大值
    minValue = np.min(dataset)
    scaler = maxValue - minValue
    dataset_scaled = dataset.copy()
    #for i in np.nditer(dataset_scaled, op_flags=['readwrite']):
    #    i[...] = (i - minValue)/scaler

    return dataset_scaled, dataset, maxValue, minValue, scaler


def inverseTransform(data, scaler, minValue):
    data_copy = data.copy()
    for i in np.nditer(data_copy, op_flags=['readwrite']):
        i[...] = i * scaler + minValue
    return data_copy


def getTrainAndTest(dataset):
    X_all, y_all = splitSequence(dataset, N_STEPS)
    X_all = X_all.reshape((X_all.shape[0], X_all.shape[1], N_FEATURES))

    train_size = len(y_all) - PRED_DAYS
    test_size = len(X_all) - train_size
    X_train = X_all[:train_size]
    y_train = y_all[:train_size]
    X_test = X_all[train_size:]
    y_test = y_all[train_size:]
    X_train = Variable(torch.Tensor(X_train))
    y_train = Variable(torch.Tensor(y_train))
    X_test = Variable(torch.Tensor(X_test))

    return X_train, y_train, X_test, y_test


def metrics(test, predict):
    print('MSE均方误差,越小越好')
    mse = mean_squared_error(test, predict)
    print("MSE=", mse)

    print('MAE数值越小越好，可以通过对比判断好坏')
    mae = mean_absolute_error(test, predict)
    print("MAE=", mae)

    print('R平方值，越接近1越好')
    r2 = r2_score(test, predict)
    print("R_square=", r2)

    accu_all = 0
    for i in range(len(predict)):
        if int(test[i]) == 0:
            continue
        accu = abs(test[i] - predict[i]) / test[i]
        accu_all += accu
    accu_avg = accu_all / len(predict) * 100
    print('平均误差为' + str(accu_avg) + '%')


@variational_estimator
class LSTM(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers):
        super(LSTM, self).__init__()
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.lstm = BayesianLSTM(input_size, hidden_size)
        self.linear = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x_, _ = self.lstm(x)

        # gathering only the latent end-of-sequence for the linear layer
        x_ = x_[:, -1, :]
        x_ = self.linear(x_)
        return x_


if __name__ == '__main__':
    tableName = 'Xingang_nansha_after_add.csv'
    columnName = 'NUM_BOX'
    dataset_scaled, dataset, maxValue, minValue, scaler = readData(tableName, columnName)
    X_train, y_train, X_test, y_test = getTrainAndTest(dataset_scaled)
    lstm = LSTM(1, 1, 96, 1)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(lstm.parameters(), lr=LEARNING_RATE)

    for epoch in range(NUM_EPOCH):
        outputs = lstm(X_train)
        outputs = outputs.reshape(-1)
        optimizer.zero_grad()
        # obtain the loss function
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            print("Epoch: %d, loss: %1.5f" % (epoch, loss.item()))

    lstm.eval()
    train_predict = lstm(X_test)
    data_predict = train_predict.data.numpy()
    metrics(y_test, data_predict)
    plt.plot(y_test)
    plt.plot(data_predict)
    plt.suptitle('Time-Series Prediction')
    plt.show()