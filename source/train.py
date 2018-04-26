#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 16:48:36 2018

@author: Kelvin
"""
import pandas as pd
import os
from helpers import IM_SHAPE, batch_generator
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Lambda, Conv2D, MaxPooling2D, Dropout, Dense, Flatten
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam, Nadam



#Load data from csv file
def load():
    df = pd.read_csv(os.path.join(os.getcwd(), 'data', 'driving_log_new.csv'))
    X = df[['center_camera','left_camera','right_camera']].values
    y = df['steer_angle'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)
    return X_train, X_test, y_train, y_test

def make_cnn_model():
    model = Sequential()
    #image normalization layer,
    model.add(Lambda(lambda x: x/127.5-1.0, input_shape=IM_SHAPE))
    #takes care of vanishing gradient problem, each of these will filter out images
    model.add(Conv2D(24, 5, 5, activation = 'elu', subsample=(2,2)))
    model.add(Conv2D(36, 5, 5, activation='elu', subsample=(2, 2)))
    model.add(Conv2D(48, 5, 5, activation='elu', subsample=(2, 2)))
    model.add(Conv2D(64, 3, 3, activation='elu'))
    model.add(Conv2D(64, 3, 3, activation='elu'))
    model.add(Dropout(0.5))
    model.add(Flatten())
    #fully connected layers,
    model.add(Dense(100, activation='elu'))
    model.add(Dense(50, activation='elu'))
    model.add(Dense(10, activation='elu'))
    model.add(Dense(1))
    model.summary()
    return model

def train_model(model, X_train, X_test, y_train, y_test):
    cp = ModelCheckpoint('model-{epoch:03d}.h5', monitor = 'val_loss', verbose = 0, save_best_only=True, mode='auto')
    model.compile(loss='mean_squared_error', optimizer=Nadam(lr=1.0e-4))
    model.fit_generator(batch_generator('data', X_train, y_train, 40, True),
                        20000,
                        30,
                        max_q_size=1,
                        validation_data=batch_generator('data', X_test, y_test, 40, False),
                        nb_val_samples=len(X_test),
                        callbacks=[cp],
                        verbose=1)
    
if __name__ == '__main__':
    data=load()
    model = make_cnn_model()
    train_model(model, *data)

    



    
    







