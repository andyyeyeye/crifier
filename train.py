global WINDOW
global LEN_TRAIN
global DATASET_PATH
global WHOLE_LENGTH

#Set parameters

DATASET_PATH = "ethusd.csv"
WINDOW = 400
LEN_TRAIN = 9000
WHOLE_LENGTH = 10000

#import
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

HAVE_GPU = True

if HAVE_GPU:
    session = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True)) #run on gpu

df = pd.read_csv(DATASET_PATH) #read and cut
df = df[-WHOLE_LENGTH:]

time = df.loc[LEN_TRAIN:, 'time'] #for later use (might not use at all)

training_set = df.iloc[:LEN_TRAIN, 1:2].values #separate train & test set
test_set = df.iloc[LEN_TRAIN-WINDOW:, 1:2].values

len_df = len(df) # for later use

df = None #save memory

sc = MinMaxScaler(feature_range = (0, 1)) #save minmax transforms for later use

def reshaper(dataset,length,sc): #reshape & scale between 0 and 1
    dataset_scaled = sc.fit_transform(dataset)
    X_scaled = []
    y_scaled = []
    for i in range(WINDOW, length):
        X_scaled.append(dataset_scaled[i-WINDOW:i, 0])
        y_scaled.append(dataset_scaled[i, 0])
    X_scaled, y_scaled = np.array(X_scaled), np.array(y_scaled)
    X_scaled = np.reshape(X_scaled, (X_scaled.shape[0], X_scaled.shape[1], 1))
    return X_scaled, y_scaled

X_train, y_train = reshaper(training_set,LEN_TRAIN,sc) #reshape & scale
X_test, y_test = reshaper(test_set,len_df-LEN_TRAIN+WINDOW,sc)

#making a lstm model with dropouts
model = Sequential()
model.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units = 50, return_sequences = True))
model.add(Dropout(0.2))
model.add(LSTM(units = 50, return_sequences = True))
model.add(Dropout(0.2))
model.add(LSTM(units = 50))
model.add(Dropout(0.2))
model.add(Dense(units = 1))

model.compile(optimizer = 'adam', loss = 'mean_squared_error') #compile

model.fit(X_train, y_train, epochs = 100, batch_size = 32) #fit

predicted_stock_price = model.predict(X_test) #making a predictions
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

# plot
plt.plot(np.arange(len_df-LEN_TRAIN),test_set[WINDOW:], color = 'red', label = 'Real') #use time instead (optional)
plt.plot(np.arange(len_df-LEN_TRAIN),predicted_stock_price, color = 'blue', label = 'Predicted')
plt.xticks(np.arange(0,len_df-LEN_TRAIN,50))
plt.title('Crypto Price Prediction')
plt.xlabel('Time')
plt.ylabel('Crpyto Price')
plt.legend()
plt.show()

# save model
model_json = model.to_json()
with open("model.json", "w") as json_file : 
    json_file.write(model_json)

model.save_weights("model.h5")
print("Saved model to disk")