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

model_cnn = keras.Sequential(
    [
        keras.Input(shape=(15,1)),
        layers.Conv1D(16, kernel_size=3, activation="relu", padding="same"),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(0.2),
        layers.Conv1D(8, kernel_size=3, activation="relu", padding="same"),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(0.2),
        layers.Flatten(),
        layers.Dense(1, activation="sigmoid"),
    ]
)

# model_cnn.summary()

model_cnn.compile(loss = 'binary_crossentropy',
                  optimizer= Adam(learning_rate = 0.001),
                  metrics=['accuracy'])

history_cnn = model_cnn.fit(x_train, y_train, 
                            epochs = 100,
                            batch_size = 200000,
                            validation_split = 0.2)

# show_train_history(history_cnn, 'loss', 'val_loss', title = 'Train History - CNN')

model_cnn.save('./models/CNN.h5')