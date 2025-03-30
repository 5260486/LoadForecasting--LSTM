import pandas as pd
import numpy as np

class TimeDataConstruct():
    """转成有监督数据"""

    def __init__(self, data):
        self.dataset=data

    #axis：指定移动的方向，可以为 0（默认，沿行移动）或 1（沿列移动）
    def series_to_supervised(self, direction, n_in=1, n_out=1):
        data=self.dataset
        n_vars = 1 if type(data) is list else data.shape[1]
        df = pd.DataFrame(data)
        cols, names = list(), list()
        #数据序列(也就是input) input sequence (t-n, ... t-1)
        for i in range(n_in, 0, -1):
            cols.append(df.shift(i,axis=direction))
            names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
            #预测数据（input对应的输出值） forecast sequence (t, t+1, ... t+n)
        for i in range(0, n_out):
            cols.append(df.shift(-i,axis=direction))
            if i == 0:
                names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
            else:
                names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
        #拼接 put it all together
        agg = pd.concat(cols, axis=1)
        agg.columns = names
        # 删除值为NAN的行或列 drop rows with NaN values
        # 一般不删除列，因为列是特征，特征不完整会影响模型的训练
        agg.dropna(axis=direction,how='any',inplace=True)
        return agg

    def prepare_data(self, nlags):
    #prepares data for LSTM model, x=last nlags values, y=(nlags+1)'th value
        data=self.dataset
        data_x, data_y = [], []
        for i in range(data.shape[0]):
            for j in range(0, data.shape[1]-nlags):
                data_x.append(data[i, j:j+nlags])
                data_y.append(data[i, j+nlags])
        data_x = np.array(data_x)
        data_y = np.array(data_y).reshape(-1, 1)
        return data_x, data_y


    #把数据分为训练数据和测试数据 split into train and test sets
    def data_split(self,n_train_hours,split_num,end_num):
        values = self.dataset
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
        return train_X, train_y, val_X, val_y, test_X, test_y