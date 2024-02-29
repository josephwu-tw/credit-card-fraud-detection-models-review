### Setup

import dataprocess as df
import matplotlib.pyplot as plt
import sklearn.metrics as skmetric
import seaborn as sns

from tensorflow import keras
from keras import layers
from keras.optimizers import Adam

x_train, y_train = df.x_train, df.y_train, 



### Functions

def show_train_history(train_history, train, validation, title = 'Train History'):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title(title)
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.show()

def plot_confusion_matrix(cm, title):
    fig, (ax1) = plt.subplots(ncols=1, figsize=(5,5))
    sns.heatmap(cm/1000, 
                xticklabels=['Not Fraud', 'Fraud'],
                yticklabels=['Not Fraud', 'Fraud'],
                annot = True, fmt = 'g',
                linewidths=.2,linecolor="Darkblue", cmap="Blues")
    plt.title(title, fontsize=14)
    plt.show()



### Data reshape
    
x_train_lstm = x_train.reshape((x_train.shape[0], 1, x_train.shape[1]))

# print(x_train_lstm.shape)
# print(y_train.shape)
# print(x_test_lstm.shape)
# print(y_test.shape)


### Model Build

keras.backend.clear_session()

model_lstm = keras.Sequential(
    [
        keras.Input(shape=(1,15)),
        layers.LSTM(16, activation = 'tanh', return_sequences=True),
        layers.Dropout(0.2),
        layers.LSTM(8, activation = 'tanh'),
        layers.Dropout(0.2),
        layers.Dense(1, activation="sigmoid" ,trainable=True),
    ]
)

# model_lstm.summary()

model_lstm.compile(loss = 'binary_crossentropy',
                   optimizer= Adam(learning_rate = 0.001),
                   metrics=['accuracy'])


history_lstm = model_lstm.fit(x_train_lstm, y_train,  
                              epochs = 100,
                              batch_size = 200000,
                              validation_split = 0.2)

# show_train_history(history_lstm, 'loss', 'val_loss', title = 'Train History - LSTM')

model_lstm.save('./models/LSTM.h5')