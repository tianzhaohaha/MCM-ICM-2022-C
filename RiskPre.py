import pandas as pd
import numpy as np
import talib as tb
from sklearn.preprocessing import MinMaxScaler
#from __future__ import print_function
import math
#importing keras modules
from keras.models import Sequential
from keras.layers import Dense, Activation ,Dropout , Flatten , Conv1D , MaxPooling1D
from keras.layers.recurrent import LSTM
from keras import losses
from keras import optimizers
from timeit import default_timer as timer


class RiskModel:
    def __init__(self,bitcoin,gold,Win_size=14,):
        self.bitcoin = bitcoin
        self.gold = gold
        self.Win_size = Win_size


        self.model = Sequential()



    def train(self,train,train_label):
        self.model.add(Dense(128, input_shape=(6, 11)))
        self.model.add(
            Conv1D(filters=160, kernel_size=1, padding='same', activation='relu', kernel_initializer="glorot_uniform"))
        self.model.add(MaxPooling1D(pool_size=2, padding='valid'))
        self.model.add(
            Conv1D(filters=96, kernel_size=1, padding='same', activation='relu', kernel_initializer="glorot_uniform"))
        self.model.add(MaxPooling1D(pool_size=2, padding='valid'))
        self.model.add(Dropout(0.2))
        self.model.add(LSTM(64, return_sequences=True))
        self.model.add(LSTM(32, return_sequences=False))
        self.model.add(Dense(64, activation="relu", kernel_initializer="uniform"))
        self.model.add(Dense(1, activation="relu", kernel_initializer="uniform"))
        self.model.compile(loss='mse', optimizer='adam', metrics=['mae'])

        # Summary of the Model
        print(self.model.summary())

        print("train risk model")
        start = timer()
        history = self.model.fit(train,
                            train_label,
                            batch_size=128,
                            epochs=1500,
                            validation_split=0.2,
                            verbose=2)
        end = timer()
        print(end - start)

        print("train finished")
        return self.model




    def get_traindata(self,feature_G,data):
        """
        ????????????????????????7????????????
        :param data: DataFrame??????,??????get_feature?????????
        :return:
        """


        # ??????????????????
        result = []
        sequence_length = 6
        for index in range(len(feature_G) - sequence_length):
            result.append(feature_G[index: index + sequence_length])
        result = np.array(result)

        train = result
        train_label = self.get_train__point(data)

        print("train", train.shape)
        print("train", train_label.shape)


        return train,train_label
    def get_train__point(self,data):
        """
        ?????????DataFrame????????????gold or bitcoin
        :return: ??????????????????????????????????????????
        """
        point_posi = np.zeros((len(data)-6,1))
        point_num = 0
        if self.bitcoin:
            for i in range(len(data) - int(self.Win_size / 2)):
                point = i + int(self.Win_size / 2)
                if point <= (len(data) - 8):
                    if (data[point] > data[point + 1] and data[point] > data[point - 1]) or (data[point] < data[point + 1] and data[point] < data[point - 1]):
                        # print(point)
                        min_G = data[i]
                        max_G = data[i]
                        for j in range(self.Win_size):
                            if min_G > data[i + j]:
                                min_G = data[i + j]
                            if max_G < data[i + j]:
                                max_G = data[i + j]
                        if (max_G - min_G) >= 300:
                                point_posi[point] = 1
                                point_num += 1
        if self.gold:
            for i in range(len(data) - int(self.Win_size / 2)):
                point = i + int(self.Win_size / 2)
                if point <= (len(data) - 8):
                    if (data[point] > data[point + 1] and data[point] > data[point - 1]) or (data[point] < data[point + 1] and data[point] < data[point - 1]):
                        # print(point)
                        min_G = data[i]
                        max_G = data[i]
                        for j in range(self.Win_size):
                            if min_G > data[i + j]:
                                min_G = data[i + j]
                            if max_G < data[i + j]:
                                max_G = data[i + j]
                        if (max_G - min_G) >= 30:
                                point_posi[point] = 1
                                point_num += 1
        return point_posi[6:]

    def get_point(self,data):
        """
        ?????????DataFrame????????????gold or bitcoin
        :return: ??????????????????????????????????????????
        """
        point_posi = np.zeros((len(data),1))
        point_num = 0
        if self.bitcoin:
            for i in range(len(data) - int(self.Win_size / 2)):
                point = i + int(self.Win_size / 2)
                if point <= (len(data) - 8):
                    if (data[point] > data[point + 1] and data[point] > data[point - 1]) or (data[point] < data[point + 1] and data[point] < data[point - 1]):
                        # print(point)
                        min_G = data[i]
                        max_G = data[i]
                        for j in range(self.Win_size):
                            if min_G > data[i + j]:
                                min_G = data[i + j]
                            if max_G < data[i + j]:
                                max_G = data[i + j]
                        if (max_G - min_G) >= 300:
                                point_posi[point] = 1
                                point_num += 1
        if self.gold:
            for i in range(len(data) - int(self.Win_size / 2)):
                point = i + int(self.Win_size / 2)
                if point <= (len(data) - 8):
                    if (data[point] > data[point + 1] and data[point] > data[point - 1]) or (data[point] < data[point + 1] and data[point] < data[point - 1]):
                        # print(point)
                        min_G = data[i]
                        max_G = data[i]
                        for j in range(self.Win_size):
                            if min_G > data[i + j]:
                                min_G = data[i + j]
                            if max_G < data[i + j]:
                                max_G = data[i + j]
                        if (max_G - min_G) >= 30:
                                point_posi[point] = 1
                                point_num += 1
        return point_posi





    def get_feature(self, data):
        """
        ?????????DataFrame????????????gold or bitcoin
        :return: RISK?????????????????????????????????
        """
        # MACD
        # ??????Talib???MACD????????????????????????????????? ???close ????????????????????????????????????12???26???9??????????????????????????????macd???macdsignal???macdhist
        macd, macdsignal, macdhist = tb.MACD(data, fastperiod=12, slowperiod=26, signalperiod=9)
        macd = pd.DataFrame(macd, columns=['macd'])
        macdsignal = pd.DataFrame(macdsignal, columns=['macdsignal'])
        macdhist = pd.DataFrame(macdhist, columns=['macdhist'])
        # RSI
        # ??????Talib???RSI???????????????????????????????????????open???????????????????????????????????????12????????????????????????rsi
        rsi = tb.RSI(data, timeperiod=12)
        rsi = pd.DataFrame(rsi, columns=['rsi'])
        # SMA
        # ??????????????????
        sma5 = tb.SMA(data, timeperiod=5)
        sma15 = tb.SMA(data, timeperiod=10)
        sma30 = tb.SMA(data, timeperiod=30)

        sma5 = pd.DataFrame(sma5, columns=['sma5'])
        sma15 = pd.DataFrame(sma15, columns=['sma15'])
        sma30 = pd.DataFrame(sma30, columns=['sma30'])

        # ROC
        roc = tb.ROC(data, timeperiod=6)
        roc = pd.DataFrame(roc, columns=['roc'])

        # MOM
        mom = tb.MOM(data, timeperiod=5)
        mom = pd.DataFrame(mom, columns=['mom'])
        # ROCR
        rocr = tb.ROCR(data, timeperiod=6)
        rocr = pd.DataFrame(rocr, columns=['rocr'])

        # ????????????
        sma5.fillna(method="bfill", inplace=True)
        sma15.fillna(method="bfill", inplace=True)
        sma30.fillna(method="bfill", inplace=True)
        macd.fillna(method="bfill", inplace=True)
        macdsignal.fillna(method="bfill", inplace=True)
        macdhist.fillna(method="bfill", inplace=True)
        rsi.fillna(method="bfill", inplace=True)
        roc.fillna(method="bfill", inplace=True)
        mom.fillna(method="bfill", inplace=True)
        rocr.fillna(method="bfill", inplace=True)

        feature = pd.concat([data, macd, macdhist, macdsignal, rsi, sma5, sma15, sma30, roc, mom, rocr], axis=1)
        if len(feature)<20:
            print("??????????????????20????????????")
        # ?????????????????????
        scaler = MinMaxScaler(copy=True, feature_range=(-1, 1))
        feature_G = scaler.fit_transform(feature)
        feature_G = pd.DataFrame(feature_G,
                                 columns=['price', 'macd', 'macdhist', 'macdsignal', 'rsi', 'sma5', 'sma15', 'sma30',
                                          'roc', 'mom', 'rocr'])
        return feature_G

