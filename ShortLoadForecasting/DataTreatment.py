import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
class DataTreatment():
    """
        data pretreatment
        1. If there are data missing, insert data based on the forward data.
        2. Extract month data used for predicting.
    """

    def __init__(self, dataset):
        self.dataset=dataset

    def FillData(self):
        dataset=self.dataset
        dataset = dataset.replace(0, np.nan)
        dataset = dataset.ffill()
        values=dataset.values
        return values

    # axis=0,按行差分；axis=1，按列差分
    def DiffData(self,direction):
        values = self.dataset
        values=pd.DataFrame(values).diff(1, axis=direction)
        values=values.dropna(axis=direction)
        values=values.to_numpy()
        return values

    def ScaleData(self):
        values = self.dataset
        #归一化 normalize features
        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaled = scaler.fit_transform(values.reshape(values.shape[0],values.shape[1]))
        return scaler,scaled

    def InverseScale(self,scaler,y,values_initial,test_X,n_train_hours,end_num):
        #invert scaling for predict
        inv_yhat = np.concatenate((y, test_X), axis=1)
        inv_yhat = scaler.inverse_transform(inv_yhat)
        rescaled_yhat=[x+y for x, y in zip(inv_yhat[:,0], values_initial[n_train_hours+2:end_num+2])]


        return rescaled_yhat