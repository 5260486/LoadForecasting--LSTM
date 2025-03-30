import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from matplotlib import pyplot
from math import sqrt
import ReadData
import DataTreatment
import TimeDataConstruct
import Model

filepath='e:/programs/python/ShortLoadForecasting/data.xlsx'
time_lower=20090101
time_upper=20090202
sheetname="Area1_Load"

data=ReadData.ReadData(filepath,sheetname)
dataset=data.FormatData(2,time_lower,time_upper)
days_trainAndVal=int(dataset.shape[0]/24/12)-1
days_predict=1

data_pretreat=DataTreatment.DataTreatment(dataset)
values_initial=data_pretreat.FillData()
values_initial=values_initial[:,1]
data_pretreat.dataset=values_initial
dataset=data_pretreat.DiffData(direction=0)
data_pretreat.dataset=dataset
scaler,data_scaled=data_pretreat.ScaleData()
data_series=TimeDataConstruct.TimeDataConstruct(data_scaled)
reframed=data_series.series_to_supervised(direction=0,n_in=1,n_out=1)

# nlags=20
# split_idx = int(31*0.8)
# end_idx=data_scaled.shape[0]-1
# train, val = data_scaled[:split_idx, :], data_scaled[split_idx:end_idx, :]
# test= data_scaled[end_idx:, :]

# train_series=TimeDataConstruct.TimeDataConstruct(train)
# val_series=TimeDataConstruct.TimeDataConstruct(val)
# test_series=TimeDataConstruct.TimeDataConstruct(test)
# train_x, train_y = train_series.prepare_data(nlags)
# val_x, val_y = val_series.prepare_data(nlags)
# test_x, test_y = test_series.prepare_data(nlags)

n_train_hours = int(24*12*days_trainAndVal)
split_num=int(24*12*days_trainAndVal*0.8)
end_num=n_train_hours+24*12*days_predict
dataforsplit=TimeDataConstruct.TimeDataConstruct(reframed.values)
train_X, train_y, val_X, val_y, test_X, test_y=dataforsplit.data_split(n_train_hours,split_num,end_num)
print(train_X.shape, train_y.shape,val_X.shape, val_y.shape, test_X.shape, test_y.shape)

model=Model.Model(train_X,train_y,val_X,val_y)
inv_yhat,loss,val_loss= model.FitLSTM(test_X)

pyplot.plot(loss, label='trainloss')
pyplot.plot(val_loss,label='testloss')
pyplot.legend()
pyplot.title('The change of Loss.')
#pyplot.savefig('E:/programs/python/TimeSeriesAnalysis/LossDecreasing.png')
pyplot.show()

test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
test_y = test_y.reshape((len(test_y), 1))
rescaled_yhat=data_pretreat.InverseScale(scaler,inv_yhat,values_initial,test_X,n_train_hours,end_num)
rescaled_y=data_pretreat.InverseScale(scaler,test_y,values_initial,test_X,n_train_hours,end_num)

pyplot.plot(rescaled_yhat,label='Prediction')
pyplot.plot(rescaled_y,label='RealLoad')
pyplot.legend()
pyplot.xlabel('Time /5min')
pyplot.ylabel('Load /MW')
pyplot.title('Prediction of One Day.')
#pyplot.savefig('E:/programs/python/TimeSeriesAnalysis/PredictionLoad.png')
pyplot.show()

# calculate RMSE
rmse = sqrt(mean_squared_error(rescaled_y, rescaled_yhat))
print('Test RMSE: %.3f' % rmse)