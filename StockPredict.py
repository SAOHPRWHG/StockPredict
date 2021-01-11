# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 13:13:14 2021

@author: Augus
"""

# encoding=gbk
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,LSTM,Dropout
import talib
from jqdatasdk import *
from jqdatasdk import finance#获取外盘数据
import datetime 
import pandas as pd
import dateutil

GLOBALTIME=datetime.datetime.now().strftime('%Y-%m-%d')#计算出当前时间

auth('13633411298', 'Jk12345678')
df = get_price('000016.XSHG', start_date='2017-01-01', end_date=GLOBALTIME, 
               frequency='daily',
               #frequency='minute',
               fields=['open', 'close', 'high', 'low', 'volume', 'pre_close'])
df['pct_chg'] = ( df['close'] - df['pre_close'])/df['pre_close']
df = df.rename({'volume':'vol'}, axis='columns')#，这种方法比较灵活。
df=df.drop('pre_close',axis=1)

# 各个指标
data=pd.DataFrame(df[['open', 'high', 'close', 'low', 'vol', 'pct_chg']])

## 名称：随机指标,俗称KD
data['slowk'], data['slowd'] = \
talib.STOCH(df['high'].values, df['low'].values, df['close'].values,
            fastk_period=9, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)

## 名称： 布林线指标
data["BBANDS_upper"],data["BBANDS_middle"],data["BBANDS_lower"] =talib.BBANDS(df['close'].values, matype=talib.MA_Type.T3)

## 名称：平滑异同移动平均线
data["MACD"],df['MACDsignal'],df['MACDhist'] =talib.MACD(df['close'].values, fastperiod=12, slowperiod=26, signalperiod=9)

## 名称：真实波动幅度均值
data["ATR"] = talib.ATR(df['high'].values,df['low'].values, df['close'].values, timeperiod=14)

## AD - Chaikin A/D Line 量价指标
data["AD"] = talib.AD(df['high'].values,df['low'].values, df['close'].values, df['vol'].values)

data=data[~data["slowk"].isin(['NaN'])]
data=data[~data["slowd"].isin(['NaN'])]
data=data[~data["BBANDS_upper"].isin(['NaN'])]
data=data[~data["MACD"].isin(['NaN'])]
data=data[~data["BBANDS_middle"].isin(['NaN'])]
data=data[~data["BBANDS_lower"].isin(['NaN'])]
data=data[~data["ATR"].isin(['NaN'])]
data=data[~data["AD"].isin(['NaN'])]

data=data[['open', 'high', 'close', 'low', 'vol', 'pct_chg','slowk','BBANDS_upper','BBANDS_lower','MACD','slowd']]

data.head(10)

Alldata=data.shape[0]
print("数据总量:",Alldata)
first_length=int(Alldata*0.8)
print("训练数据:",first_length)
second_length=int(Alldata*0.2)
print("测试数据:",second_length)

# 训练时间长度
time_stamp = 1

train = data[0:first_length + time_stamp]
valid = data[first_length - time_stamp:]

judge_result=valid[time_stamp-1:-1]
judge_result=judge_result['close']
judge_result=np.array(judge_result)

# 归一化
scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
scaled_data_train = scaler.fit_transform(train)
scaled_data_valid = scaler.fit_transform(valid)

# 训练数据获取
x_train, y_train = [], []
for i in range(time_stamp, len(train)):
    x_train.append(scaled_data_train[i - time_stamp:i])
    y_train.append(scaled_data_train[i, 2])

x_train, y_train = np.array(x_train), np.array(y_train)

# 验证数据获取
x_valid, y_valid = [], []
for i in range(time_stamp, len(valid)):
    x_valid.append(scaled_data_valid[i - time_stamp:i])
    y_valid.append(scaled_data_valid[i, 2])

x_valid, y_valid = np.array(x_valid), np.array(y_valid)

# lstm训练
epochs = 10
batch_size = 16

model = Sequential()
#model.add(LSTM(units=80, return_sequences=True, input_dim=x_train.shape[-1], input_length=x_train.shape[1]))
#model.add(Dropout(0.2))
model.add(LSTM(units=80, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[-1])))
model.add(LSTM(units=20))#20
model.add(Dense(units=1))
model.compile(loss='mean_squared_error', optimizer='adam')
history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)

# 模型预测
closing_price = model.predict(x_valid)
scaler.fit_transform(pd.DataFrame(valid['close'].values))

closing_price = scaler.inverse_transform(closing_price)
y_valid = scaler.inverse_transform([y_valid])

total_count=y_valid.size
predict=closing_price.flatten()#np.resize(predict,(-1,))
real=y_valid.flatten()#np.resize(real,(-1,))

#total_count=646
result_sum=0
for i in range(total_count):
    if ( predict[i] - judge_result[i] ) * ( real[i] - judge_result[i] ) > 0:
        result_sum = result_sum + 1
        
#print( result_sum/total_count )

#total_count=646
result=[]
for i in range(total_count):
    if ( predict[i]> judge_result[i] ) :
        result.append(1)
    else:
        result.append(0)
        
print(result[-8:0])

print("当前时间：", GLOBALTIME)
print("收盘价预测结果：",predict[-1] )
print("前日收盘价：",judge_result[-1] )
print("二分预测结果：")
if result[-1]:
    print("涨")
else:
    print("跌")

# 画图
plt.figure(figsize=(16, 8))
predict_data = {
    'Predictions': closing_price.reshape(1,-1)[0],
    'Close': y_valid[0],
    'pre_close' : judge_result
}
predict_data = pd.DataFrame(predict_data)

plt.plot(predict_data[['Predictions']],label='Prediction')
plt.plot(predict_data[['Close']],label='Real')
plt.plot(predict_data[['pre_close']],label='Close')
plt.xlabel('time')
plt.ylabel('price')
plt.legend()
plt.savefig('predict.png')
plt.show()

