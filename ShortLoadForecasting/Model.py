from keras.models import Sequential
from keras.layers import LSTM,Dense

class Model():

    def __init__(self,train_X,train_y,val_X,val_y):
        self.train_X=train_X
        self.train_y=train_y
        self.val_X=val_X
        self.val_y=val_y

    def FitLSTM(self,test_x):
        loss=[]
        val_loss=[]
        model=Sequential()
        model.add(LSTM(100, input_shape=(self.train_X.shape[1], self.train_X.shape[2])))
        model.add(Dense(5, 'relu'))
        model.add(Dense(1, 'linear'))
        model.compile(loss='mean_squared_error', optimizer='adam')
        for i in range(5):
            history=model.fit(self.train_X,self.train_y, epochs=1, batch_size=1, verbose=1, validation_data=(self.val_X, self.val_y), shuffle=False)
            model.reset_states()
            loss.append(history.history['loss'])
            val_loss.append(history.history['val_loss'])
        inv_yhat = model.predict(test_x, batch_size=1)
        return inv_yhat,loss,val_loss
