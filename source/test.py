#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  5 00:27:16 2018

@author: Kelvin
"""
from PIL import Image
from flask import Flask
import socketio
from keras.models import load_model
#Library for input output
from io import BytesIO
#import command line library
import argparse
#web server
import eventlet.wsgi
#Images library
import base64
#numpy library
import numpy as np
#helper functions
import helpers


#Creating instance of websocket, and Flask server
sio = socketio.Server()
app = Flask(__name__)
#Initiating Model and prev_image_array
model = None
#Set Speed limits for autonomous car
max_speed = 30

@sio.on('telemetry')
def telemetry(sid, data):
    if data:
        #Checking current speed, throttle, steering angle, and image
        steering_angle = float(data["steering_angle"])
        throttle = float(data["throttle"])
        speed = float(data["speed"])
        image = Image.open(BytesIO(base64.b64decode(data["image"])))
        try:
            #Changing the center image
            #convert PIL image to np array
            image = np.asarray(image)
            #apply the preprocessing
            image = helpers.preprocess(image)
            #our training model expects 4D array
            image = np.array([image])
            
            #use trained model to predict angle for a given image
            steering_angle = float(model.predict(image, batch_size=1))
            #Set throttle of the car
            throttle = float(1) - (steering_angle**2) - (speed/30)**2
            print('steering angle:{}, speed:{}, throttle:{}'.format(steering_angle, speed, throttle))
            #send back the steering angle, and throttle
            send_control(steering_angle,throttle)
        except Exception as e:
            print(e)
    else:
        sio.emit('manual', data={}, skip_sid=True)
        
@sio.on('connect')
def connect(sid, environ):
    print("Connected from car simulator!", sid)
    send_control(0,0)

def send_control(steer_angle, throttle):
    sio.emit("steer", data={'steering_angle': steer_angle.__str__(),
                            'throttle': throttle.__str__()
                            }, skip_sid=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str)
    args = parser.parse_args()
    model = load_model(args.model)
    app = socketio.Middleware(sio, app)
    eventlet.wsgi.server(eventlet.listen(('',4567)), app)
    
    
            



