import os, sys, time, random, itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Lambda, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ReduceLROnPlateau

# debugging logs
# 0 = all messages are logged (default behavior)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages are not printed
# 3 = INFO, WARNING, and ERROR messages are not printed
# this should be placed before importing tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# enable GPU device
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# reproducibility
seed = 1234567
os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)
np.random.seed(seed)
session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1,inter_op_parallelism_threads=1)
sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(),config=session_conf)
tf.compat.v1.keras.backend.set_session(sess)


def read_data_sets():
    # load MSD dataset
    data = pd.read_csv('./YearPredictionMSD2.csv')

    # plot histogram
    if 0:
        data = data.rename(index=str, columns={'label':'year'})
        nsongs = {}
        for y in range(1922, 2012):
            nsongs[y] = len(data[data.year == y])
        yrs = range(1922, 2011)
        values = [nsongs[y] for y in yrs]
        plt.bar(yrs, values, align='center')
        plt.xlabel('Year')
        plt.ylabel('Number of songs')
        plt.show()

    # scale the data sets to the interval [0,1]
    X = data.iloc[:,1:].to_numpy()
    Y = data.iloc[:,0].to_numpy()
    a, b = X.min(), X.max()
    X = (X - a) / (b - a)
    Y = Y - Y.min()  # 1922-2011 are mapped to 0-89

    # shuffle the dataset
    nb_samples = X.shape[0]
    ind = np.random.permutation(nb_samples)
    X, Y = X[ind,:], Y[ind]

    # split data sets
    x_train, y_train = X[:463715,:], Y[:463715]
    x_test, y_test = X[463715:,:], Y[463715:]
    return x_train, y_train, x_test, y_test

def main():
    start = time.time()

    # set parameters
    nb_classes = 90
    epochs = 10
    batch_size = 64

    # 1) load MSD dataset
    x_train, y_train_t, x_test, y_test_t = read_data_sets()

    # 2a) standardize features by removing the mean and scaling to unit variance
    scaler = StandardScaler()
    scaler.fit(x_train)
    X_train_std = scaler.transform(x_train)
    X_test_std = scaler.transform(x_test)

    # 2b) run principal component analysis (PCA)
    pca = PCA(n_components=0.9)
    pca.fit(X_train_std)
    X_train_proc = pca.transform(X_train_std)
    X_test_proc = pca.transform(X_test_std)

    # 2c) encode labels to one-hot coding
    y_train = to_categorical(y_train_t, nb_classes)
    y_test = to_categorical(y_test_t, nb_classes)

    # 3) build a model
    if 1:
        model = Sequential()
        model.add(Dense(55, input_shape=(55,)))
        model.add(Dense(110))
        model.add(Dense(nb_classes, activation='softmax'))
    else:
        model = Sequential()
        model.add(Dense(180, input_shape=(55,)))
        model.add(Dropout(0.2))
        model.add(Dense(360, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(360, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(nb_classes, activation='softmax'))

    # 4) compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # 5) train the model
    lr_schedule = ReduceLROnPlateau(monitor='val_loss', patience=2, verbose=1, factor=0.5, min_lr=0.0001)
    hist = model.fit(X_train_proc, y_train, validation_data=(X_test_proc, y_test),
                     epochs=epochs, batch_size=batch_size, callbacks=[lr_schedule])

    # 6) evaluate model's performance
    pred = model.predict(X_test_proc)  # (51630, 55)
    pred_class = np.argmax(pred, axis=-1)  # (51630,)
    print('Mean Absolute Error: %f' % np.mean(np.absolute(y_test_t - pred_class)))
    print('Mean Square Error: %f' % np.sqrt(np.mean(np.absolute(y_test_t - pred_class)^2)))

    # model_1: Mean Absolute Error: 8.411118, Mean Square Error: 2.928557
    # model_2: Mean Absolute Error: 7.303564, Mean Square Error: 2.738096

    elapsed = time.time() - start
    print('Elapsed {:.2f} minutes'.format(elapsed/60.0))
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (ValueError,IOError) as e:
        sys.exit(e)
