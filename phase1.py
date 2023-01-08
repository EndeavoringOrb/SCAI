import tensorflow as tf
import mss
import numpy as np
import keyboard
from time import sleep
import cv2
import mouse
import winsound
import os
import threading


healthmodel = tf.keras.models.load_model('health_models/health_model105.h5')
dmgmodel = tf.keras.models.load_model('dmg_models/dmg_model35.h5')

files = os.listdir('dmg_pics')
num_files = len(files)
stop_flag = False
capture_flag = False
xy = 35
bounding = (640-xy,360-xy,640+xy,360+xy)

# HEALTH FUNCTIONS
def take_health_screenshot(bounding_box=(0,0,1280,720)):
    with mss.mss() as sct:
        # Capture the screenshot
        sct_img = sct.grab(bounding_box)

        # Convert the image to grayscale
        gray_image = cv2.cvtColor(np.array(sct_img), cv2.COLOR_BGR2GRAY)
        gray_image = np.reshape(gray_image,(19,36,1))
        im1, im2, im3 = np.split(gray_image, 3, axis=1)

        return im1,im2,im3

def convert_to_int(array):
    array = list(array[0])
    largest_val = max(array)
    number = array.index(largest_val)
    if number == 10:
        return "N"
    return number

def event():
    image1,image2,image3 = take_health_screenshot((111,598,147,617))
    predictions1 = healthmodel.predict(np.array([image1]))
    predictions2 = healthmodel.predict(np.array([image2]))
    predictions3 = healthmodel.predict(np.array([image3]))
    print(f"{convert_to_int(predictions1)}{convert_to_int(predictions2)}{convert_to_int(predictions3)}")

# DAMAGE FUNCTIONS
def on_mouseevent(event):
    global capture_flag
    try:
        if event.button == "left" and event.event_type == "down":
            capture_flag = True
    except AttributeError:
        pass

def on_keypress(event):
    global stop_flag
    stop_flag = True
    print("stop flag")
    winsound.PlaySound("start_stop_beep.wav", winsound.SND_FILENAME)

def format_damage_pred(prediction):
    prediction_arr = prediction[0]
    idx = np.argmax(prediction_arr)
    if idx == 0:
        ret_str = "Miss"
    elif idx == 1:
        ret_str = "Body Shot"
    elif idx == 2:
        ret_str = "Headshot"
    elif idx == 3:
        ret_str = "KILL"
    return ret_str

print("Press [ to start.")
while not keyboard.is_pressed("["):
    pass
timer = threading.Timer(3, event)  # create a timer that runs the "event" function every 0.1 second
timer.start()  # start the timer
keyboard.hook_key("]",on_keypress)
mouse.hook(on_mouseevent)
winsound.PlaySound("start_stop_beep.wav", winsound.SND_FILENAME)

while True:
    if stop_flag == True:
        timer.cancel()
        break
    if capture_flag == True:
        with mss.mss() as sct:
            # Capture the screenshot
            sleep(0.1)
            sct_img = sct.grab(bounding)
            image = np.array(sct_img)
        capture_flag = False
        predictions = dmgmodel.predict(np.array([image]))
        print(format_damage_pred(predictions))
    sleep(0.1)