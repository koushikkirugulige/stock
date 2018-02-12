import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import datetime
import math, time
import itertools
from sklearn import preprocessing
import datetime
from operator import itemgetter
from sklearn.metrics import mean_squared_error
from math import sqrt
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation,Reshape,Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import Conv3D
from keras.layers.recurrent import LSTM
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.advanced_activations import LeakyReLU, PReLU,ELU,ThresholdedReLU
from keras.layers.normalization import BatchNormalization
file_name='LT.NS.csv'
#col_name=['High']
stocks=pd.read_csv(file_name,header=0,names=['High'])
df=pd.DataFrame(stocks)
df['High'] = df['High']/1000
#print(df)#1973
#def create_dataset(dataset, look_back=1):

#df=np.array(df)
file_name_2='x_axis.csv'
col_name=['High','Low']
x_axis=pd.read_csv(file_name_2,header=0,names=col_name)
x_axis=pd.DataFrame(x_axis)
x_axis['High']=x_axis['High']/10000
x_axis['Low']=x_axis['Low']/10000
print(x_axis.shape)
#x_axis=x_axis.reshape((1,)+x_axis.shape)
#x_axis=x_axis.transpose(1,0)
#print(x_axis)

def load_data(stock, seq_len):
	data=stock
	amount_of_features = (stock.shape[1])
	data = stock.as_matrix() #pd.DataFrame(stock)
	sequence_length = seq_len + 1
	result = []
	for index in range(len(data) - sequence_length):
		result.append(data[index: index + sequence_length])

	result = np.array(result)
	row = int(0.95 * result.shape[0])
	train = result[:int(row), :]
	x_train = train[:, :-1]
	x_test = result[int(row):, :-1]
	x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], amount_of_features))
	x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], amount_of_features))
	# for y_parts
	#return to calling function
	return [x_train, x_test]

def load_for_y(stock,seq_len):
	amount_of_features = len(stock.columns)
	data = stock.as_matrix() #pd.DataFrame(stock)
	sequence_length = seq_len + 1
	result = []
	for index in range(len(data) - sequence_length):
		result.append(data[index: index + sequence_length])
	result=np.array(result)
	#print("result"+ str(result.shape))
	row= int(0.95*result.shape[0])
	result = np.array(result)
	train = result[:int(row), :]
	y_train = train[:, -1][:,-1]
	y_test = result[int(row):, -1][:,-1]
	return [y_train,y_test]
x_train,x_test = load_data(x_axis[::-1],5)
y_train,y_test = load_for_y(df[::-1],5)
print('y_test'+str(y_test.shape))
print('y_train'+str(y_train.shape))
print('x_test'+str(x_test.shape))
print('x_train'+str(x_train.shape))
#print(y_test)
#print(x_test)


def build():
	d=0.2
	model=Sequential()
	#model.add(LSTM(512,input_shape=(5,2),return_sequences=True))
	#model.add(Dropout(0.22))
	#model.add(LSTM(256,input_shape=(5,2),return_sequences=True))
	#model.add(Dropout(0.50))
	model.add(LSTM(128,input_shape=(5,2),return_sequences=True))
	model.add(Dropout(d))
	model.add(LSTM(64,input_shape=(5,2),return_sequences=False))
	model.add(Dropout(d))
	model.add(Dense(16,kernel_initializer='uniform',activation='relu'))
	model.add(Dense(1,kernel_initializer='uniform',activation='relu'))
	model.add(LeakyReLU(alpha=0.3))
	model.add(ELU(alpha=1.0))
	#model.add(ThresholdedReLU(theta=1.0))
	model.compile(loss='mse',optimizer='adam',metrics=['accuracy'])
	return model

model=build()
model.fit(x_train,y_train,batch_size=256,epochs=250,validation_split=0.1,verbose=2)
p=model.predict(x_test)
import matplotlib.pyplot as plt2
plt2.plot(p,color='red', label='prediction')
plt2.plot(y_test,color='blue', label='y_test')
plt2.legend(loc='upper left')
plt2.show()
