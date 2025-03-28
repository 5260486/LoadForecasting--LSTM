**The entire process of time series prediction**
**以下为对时序数据预测的普遍流程，LSTM为例**

# 使用框架版本
python 3.9

tensorflow 2.13.0

在2.13.0下如何引用keras:

from keras.models import Sequential

from keras.layers import Dense,LSTM

其他的包默认安装版本即可

# 数据获取
```python
def get_load_data(date):
    url = 'http://www.delhisldc.org/Loaddata.aspx?mode='
    print('Scraping ' + date, end=' ')
    resp = requests.get(url + date) # send a get request to the url, get response
    soup = BeautifulSoup(resp.text, 'lxml') # Yummy HTML soup
    table = soup.find('table', {'id':'ContentPlaceHolder3_DGGridAv'}) # get the table from html
    trs = table.findAll('tr') # extract all rows of the table
    if len(trs[1:])==288: # no need to create csv file, if there's no data
        with open('monthdata.csv', 'a') as f:  #'a' makes sure the values are appended at the end of the already existing file
            writer = csv.writer(f)
            for tr in trs[1:]:
                time, delhi = tr.findChildren('font')[:2]
                writer.writerow([date + ' ' + time.text, delhi.text])
    if len(trs[1:]) != 288:
        print('Some of the load values are missing..')
    else:
        print('Done')
```

Error：第四行请求程序报错，无法连接或者无法响应。挂梯子可以网页直接打开，但是请求的话不行，所以根本原因是国外的网站。搜索获取国外网站的数据基本都通过代理，需要￥，所以还是老老实实直接用开源的数据或者从网页手动复制粘贴。

# 数据预处理
+ **缺失值处理**：采用插值法（如线性插值）或基于LSTM的序列生成填补缺失段。
+ **去趋势**：对有趋势、周期性、季节性、噪音的数据进行处理，差分法简单常用，其他有滑动窗口、Hodrick-Prescott (HP)滤波
+ **归一化**：将负荷数据缩放到 [0,1] 或 [-1,1] 区间，加速模型收敛，常用Min-Max 或 Z-Score。

```python
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose

data = pd.read_csv('……/lstm_data.csv', header=None, names=['datetime', 'load'])
# or read by excel
# data = pd.read_excel('……/_load_data.xlsx',header=0)
# data=data.loc[(dataframe['datetime']>='2023-01-01') & (dataframe['datetime'] < '2023-01-03')]

# delete the null data
data.dropna(inplace=True)
# or add missing slots to a time series dataframe
from ts2ml.core import add_missing_slots
if (data_lenth%288 != 0):
    df = add_missing_slots(data, datetime_col='datetime', entity_col='day', value_col='load', freq='5min')

# deseasonalized + detrended data
dt_data=data.diff(1).dropna()
ds_dt_data = dt_data.diff(288).dropna()  
decompfreq = 288 #daily freq
result = seasonal_decompose(ds_dt_data['load'], period=decompfreq, model='aditive')
result.plot()
plt.show() 

#归一化 normalize features
scaler = MinMaxScaler(feature_range=(-1, 1))
scaled = scaler.fit_transform(ds_dt_data)
```

# 特征工程
+ **时序特征（序列构造）**：利用滑动窗口将时间序列转为监督学习问题。例如，用前 24 小时的数据预测未来 1 小时的负荷。
+ **划分数据集**：按时间顺序分为训练集、验证集和测试集（避免随机分割破坏时序性），再对三种数据集进一步拆分为输入X输出Y。划分后应重新排列数据集的形式，转换为机器学习中的张量形式，也就是编程语言中的多维数组、多维矩阵。

```python
def series_to_supervised(data, n_in, n_out, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    #数据序列(也将就是input) input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))	#向下i行
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
        #预测数据（input对应的输出值） forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))	#向上i行
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
```

# 模型构建
+ **输入层**：定义时间步长（如 24）和特征数（单变量或多变量）。
+ **LSTM 层**：堆叠多层 LSTM 单元，每层神经元数量需调优（如 50-200）。
+ **全连接层**：将 LSTM 输出映射到预测值（如单神经元输出未来负荷）。
+ **损失函数**：常用均方误差（MSE）或平均绝对误差（MAE）。
+ **优化器**：Adam 或 sgd。

```python
#design network
model = Sequential()
model.add(LSTM(100, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='sgd')
```

# 训练
+ **超参数调优**：调整时间步长、LSTM层数、学习率等。（欠拟合、过拟合）
+ **模型保存**：将训练好的模型参数保存，供后续直接调用

```python
#fit network
#网络训练50个周期，步长36
history = model.fit(train_X, train_y, epochs=50, batch_size=36, validation_data=(val_X, val_y), verbose=2,
                    shuffle=False)

#每次网络训练1个周期，循环执行100次，每次循环重置网络状态
#for i in range(100):
#    history=model.fit(train_X, train_y, epochs=1, batch_size=1, verbose=1, validation_data=(val_X, val_y), shuffle=False)
#    model.reset_states()
```

# 预测与评估
+ **反归一化**：将预测结果与对应的真实值都还原为原始量纲。**注意：预测数据反归一化后，与原始数据相加时，索引应匹配，其中涉及窗口移动行/列数+差分行/列数。**
+ **评估指标**：MAE、RMSE、MAPE（平均绝对百分比误差）。

```python
testPredict = model.predict(testX, batch_size=batch_size)

#invert scaling for forecast
test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
inv_yhat = concatenate((yhat, test_X[:, 1:]), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
rescaled_yhat=[x+y for x, y in zip(inv_yhat[:,0], values_initial[split_num+1:n_train_hours*2+1, 0])]

# calculate RMSE
rmse = sqrt(mean_squared_error(rescaled_y, rescaled_yhat))
print('Test RMSE: %.3f' % rmse)
```

# 预测结果出现滞后现象的解决方法
认为模型并未学习到数据的变化趋势，单纯使用前一个时间步的真实数据作为当前时间步的预测值。

产生滞后问题的根本原因是：数据序列中产生了变化趋势（或者说是非线性非平稳序列）

## 差分+归一化去趋势，使数据平稳，避免模型提前学习到数据的变化规律。
必须项❗

## 多步预测，循环预测
之前多步预测之后一步或多步

```python
# 将数据截取成n个一组的监督学习格式
def create_dataset(dataset, look_back):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return numpy.array(dataX), numpy.array(dataY)
```

## 引入多特征数据
并非必须，对精度应该有影响。

## 超参数调整
调整时间步长、LSTM层数、学习率

```python
from keras.optimizers import legacy
import keras.backend as K
from keras.callbacks import LearningRateScheduler

# 定义一个学习率更新函数
def step_decay(epoch):
    initial_lrate = 0.01
    drop = 0.5
    epochs_drop = 10.0
    lrate = initial_lrate * math.pow(drop,math.floor((1+epoch)/epochs_drop))
    return lrate

# 定义一个学习率的回调函数
myReduce_lr = LearningRateScheduler(step_decay)
sgd= legacy.SGD(learning_rate=0.01, momentum=0.9, decay=0.0, nesterov=False)
model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
# 在模型中调用这个回调函数，模型在训练的过程中就会按照定义的算法myReduce_lr自动更新学习率。
# 这里注意参数类型要为数组类型
history=model.fit(train_X, train_y, batch_size=32*2, epochs=100,validation_data=(val_X, val_y), verbose=2,
                   shuffle=False, callbacks=[myReduce_lr])
```

# 预测日负荷
差分、循环预测、多层LSTM，未引入多特征数据

2.1-2.28数据预测3.1

![](https://cdn.nlark.com/yuque/0/2025/png/12943134/1742740897046-324fef31-69f6-4801-83fa-06f8f556404a.png)![](https://cdn.nlark.com/yuque/0/2025/png/12943134/1742740897235-39181a3a-ace9-40e6-8f73-b29a96132008.png)



Test RMSE: 34.587

转折处略有滞后

# 多特征气候数据预测
![](https://cdn.nlark.com/yuque/0/2025/png/12943134/1742739503119-001ae460-ae1b-499a-b639-4aab6907f5d8.png)

Test RMSE: 1.563

去掉差分，有明显的滞后现象

![](https://cdn.nlark.com/yuque/0/2025/png/12943134/1742739754272-61461c99-86c0-4245-a823-7c572165948c.png)

