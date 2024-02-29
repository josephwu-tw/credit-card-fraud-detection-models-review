### Setup

import dataprocess as df
import matplotlib.pyplot as plt
import sklearn.metrics as skmetric
import seaborn as sns

from tensorflow import keras
from keras import layers
from keras.optimizers import Adam

x_train, y_train = df.x_train, df.y_train



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


### Model Build

keras.backend.clear_session()

model_dnn = keras.Sequential(
    [
        keras.Input(shape=(15,)),
        layers.Dense(16, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(8, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(1, activation="sigmoid"),
    ]
)

# model_dnn.summary()

model_dnn.compile(loss = 'binary_crossentropy',
                  optimizer= Adam(learning_rate = 0.001),
                  metrics=['accuracy'])

history_dnn = model_dnn.fit(x_train, y_train, 
                            epochs = 100,
                            batch_size = 200000,
                            validation_split = 0.2)

# show_train_history(history_dnn, 'loss', 'val_loss', title = 'Train History - DNN')

model_dnn.save('./models/DNN.keras')