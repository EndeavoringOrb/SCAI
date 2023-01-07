import tensorflow as tf
import mss
import numpy as np
import keyboard
from time import sleep
import cv2
import matplotlib as plt

def take_screenshot(bounding_box=(0,0,1280,720)):
    with mss.mss() as sct:
        # Capture the screenshot
        sct_img = sct.grab(bounding_box)

        # Convert the image to grayscale
        gray_image = cv2.cvtColor(np.array(sct_img), cv2.COLOR_BGR2GRAY)
        gray_image = np.reshape(gray_image,(19,12,1))
        return gray_image

def take_screenshot1(bounding_box=(0,0,1280,720)):
    with mss.mss() as sct:
        # Capture the screenshot
        sct_img = sct.grab(bounding_box)

        # Convert the image to grayscale
        gray_image = cv2.cvtColor(np.array(sct_img), cv2.COLOR_BGR2GRAY)
        gray_image = np.reshape(gray_image,(19,36,1))
        return gray_image

def convert_to_int(array):
    array = list(array[0])
    largest_val = max(array)
    number = array.index(largest_val)
    if number == 10:
        return "N"
    return number

model = tf.keras.models.load_model('health_model105.h5')

for i in range(100):
    print("WAITING")
    while not keyboard.is_pressed("["):
        pass
    image = take_screenshot1((111,598,147,617))
    predictions1 = model.predict(np.array([take_screenshot((111,598,123,617))]))
    predictions2 = model.predict(np.array([take_screenshot((123,598,135,617))]))
    predictions3 = model.predict(np.array([take_screenshot((135,598,147,617))]))
    print(predictions1)
    print(predictions2)
    print(predictions3)
    print(f"{convert_to_int(predictions1)}{convert_to_int(predictions2)}{convert_to_int(predictions3)}")
    #cv2.imshow("image",image)
    #cv2.waitKey(1)
    sleep(2)