global WINDOW
global LEN_TRAIN
global DATASET_PATH
global WHOLE_LENGTH

#Set parameters

DATASET_PATH = "ethusd.csv"
WINDOW = 400
LEN_TRAIN = 9000
WHOLE_LENGTH = 10000

from tensorflow.keras.models import model_from_json 
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np

json_file = open("model.json", "r") #import the model
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights("model.h5")
model.compile(loss="mean_squared_error", optimizer="adam")

df = pd.read_csv(DATASET_PATH) #read dataset
df = df[-WHOLE_LENGTH:]

def far_prediction(step_count,model,data):

    added_data = data[0:]
    prediction = []

    for i in range(step_count):

        sc = MinMaxScaler(feature_range = (0, 1))
        data_scaled = sc.fit_transform(added_data.reshape(-1, 1))
        predicted_stock_price = model.predict(np.array([data_scaled]))
        predicted_stock_price = sc.inverse_transform(predicted_stock_price)

        added_data = added_data[1:]

        added_data = np.append(added_data,predicted_stock_price[0])
        prediction.append(predicted_stock_price[0])
    
    return prediction

#test with graph
import matplotlib.pyplot as plt

LEN_PRED = 50
start = -470

prediction_set = df.iloc[start:start+400, 1:2].values 
real = [x[0] for x in df.iloc[start+400:start+400+LEN_PRED, 1:2].values]
pred = [x[0] for x in far_prediction(LEN_PRED,model,prediction_set)]

plt.plot(np.arange(LEN_PRED),real, color = 'red', label = 'Real') #use time instead (optional)
plt.plot(np.arange(LEN_PRED),pred, color = 'blue', label = 'Predicted')
plt.title('Crypto Price Prediction')
plt.xlabel('Time')
plt.ylabel('Crpyto Price')
plt.legend()
plt.show()
