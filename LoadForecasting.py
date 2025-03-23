import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


from datetime import datetime
from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import LSTM,Dense
import numpy as np
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.seasonal import seasonal_decompose
# 定义随机种子，以便重现结果
np.random.seed(2)
#转成有监督数据
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    #数据序列(也将就是input) input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
        #预测数据（input对应的输出值） forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    #拼接 put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # 删除值为NAN的行 drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg

dataset =pd.read_excel('e:/programs/python/timeseriesanalysis/2023_load_data_clean.xlsx', header=0, index_col=0)
dataset=dataset.loc[(dataset['datetime']>='2023-02-01') & (dataset['datetime'] < '2023-03-02')]
#dataset=dataset.loc[(dataset['datetime']>='2023-03-01') & (dataset['datetime'] < '2023-03-02')]
#pyplot.plot(dataset.load,label='load')
#pyplot.legend()
#pyplot.show()
#时间长度288*n天数据用于训练+测试
days_trainAndVal=int(dataset.shape[0]/24/12)-1
days_predict=1
values_initial = dataset.load.values
#标签编码 integer encode direction
#encoder = LabelEncoder()
#values_initial[:, 4] = encoder.fit_transform(values_initial[:, 4]) #第四列是风力string

#保证为float ensure all data is float
values = values_initial.astype('float32')
''' 
    ACF for searching Periodicity.
    The periodicity of load data in one year is 288,
    which is time nodes per 5 minutes in one day.
'''
#lag_acf = acf(dataset['pollution'], nlags = 365)
#np.argsort(lag_acf)    #returns the indices that would sort the array in ascending order
#pyplot.subplot(121) 
#pyplot.plot(lag_acf)
#pyplot.axhline(y=0,linestyle='--')
#pyplot.axhline(y=-1.96/np.sqrt(len(dataset['pollution'])), linestyle='--')
#pyplot.axhline(y=1.96/np.sqrt(len(dataset['pollution'])), linestyle='--')
#pyplot.show()

values=DataFrame(values).diff(1, axis=0)
values=values.dropna(axis=0)
values=values.to_numpy()

#归一化 normalize features
scaler = MinMaxScaler(feature_range=(-1, 1))
scaled = scaler.fit_transform(values.reshape(-1,1))
#转成有监督数据 frame as supervised learning
reframed = series_to_supervised(scaled, 1, 1)
#删除不预测的列 drop columns we don't want to predict
#reframed.drop(reframed.columns[[9, 10, 11, 12, 13, 14, 15]], axis=1, inplace=True)

print(reframed.head())

#数据准备
#把数据分为训练数据和测试数据 split into train and test sets
values = reframed.values
n_train_hours = int(24*12*days_trainAndVal)
split_num=int(24*12*days_trainAndVal*0.8)
end_num=n_train_hours+24*12*days_predict
#划分训练数据和测试数据
train = values[:split_num, :] 
val= values[split_num:n_train_hours, :]
test = values[n_train_hours:end_num, :]
#拆分输入输出 split into input and outputs
train_X, train_y = train[:, :-1], train[:, -1]
val_X,val_y=val[:,:-1],val[:,-1]
test_X, test_y = test[:, :-1], test[:, -1]
#reshape输入为LSTM的输入格式 reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
val_X=val_X.reshape((val_X.shape[0],1,val_X.shape[1]))
print ('train_x.shape, train_y.shape,val_x.shape, val_y.shape, test_x.shape, test_y.shape')
print(train_X.shape, train_y.shape,val_X.shape, val_y.shape, test_X.shape, test_y.shape)

##模型定义 design network
model = Sequential()
model.add(LSTM(100, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(5, 'relu'))
model.add(Dense(1, 'linear'))

#from keras.optimizers import legacy
#sgd= legacy.SGD(learning_rate=0.01, momentum=0.9, decay=1e-3, nesterov=True)
#model.compile(loss='mean_squared_error', optimizer=sgd)
model.compile(loss='mean_squared_error', optimizer='adam')
 
#模型训练 fit network
history = model.fit(train_X, train_y, epochs=20, batch_size=36*2, validation_data=(val_X, val_y), verbose=2,
                   shuffle=False)

#输出 plot history
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.title('The change of Loss.')
pyplot.savefig('E:/programs/python/TimeSeriesAnalysis/LossDecreasing.png')
pyplot.show()

#进行预测 make a prediction
yhat = model.predict(test_X)
#预测数据逆缩放 invert scaling for forecast
test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
inv_yhat = concatenate((yhat, test_X), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
rescaled_yhat=[x+y for x, y in zip(inv_yhat[:,0], values_initial[n_train_hours+2:end_num+2])]

#真实数据逆缩放 invert scaling for actual
test_y = test_y.reshape((len(test_y), 1))
inv_y = concatenate((test_y, test_X), axis=1)
inv_y = scaler.inverse_transform(inv_y)
rescaled_y=[x+y for x, y in zip(inv_y[:,0], values_initial[n_train_hours+2:end_num+2])]

pyplot.plot(inv_yhat[:100,0],label='prediction')
pyplot.plot(inv_y[:100,0],label='true')
pyplot.legend()
pyplot.show()

#画出真实数据和预测数据
pyplot.plot(rescaled_yhat,label='Prediction')
pyplot.plot(rescaled_y,label='RealLoad')
pyplot.legend()
pyplot.xlabel('Time /5min')
pyplot.ylabel('Load /MW')
pyplot.title('Prediction of One Day.')
pyplot.savefig('E:/programs/python/TimeSeriesAnalysis/PredictionLoad.png')
pyplot.show()

# calculate RMSE
rmse = sqrt(mean_squared_error(rescaled_y, rescaled_yhat))
print('Test RMSE: %.3f' % rmse)

