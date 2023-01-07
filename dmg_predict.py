import mss
import keyboard
import mouse
import numpy as np
from time import sleep
from random import uniform
import os
import winsound
import tensorflow as tf
from PIL import Image 
import PIL
#[][][][][][][][]

xy = 35
bounding = (640-xy,360-xy,640+xy,360+xy)
im_arr = []
label_arr = []
files = os.listdir('dmg_pics')
num_files = len(files)
stop_flag = False
capture_flag = False
model = tf.keras.models.load_model('dmg_model0.h5')

def on_keypress(event):
    global stop_flag
    stop_flag = True
    print("stop flag")
    winsound.PlaySound("start_stop_beep.wav", winsound.SND_FILENAME)

def on_mouseevent(event):
    global capture_flag
    try:
        if event.button == "left" and event.event_type == "down":
            capture_flag = True
    except AttributeError:
        pass

print("Press [ to start.")
while not keyboard.is_pressed("["):
    pass
keyboard.hook_key("]",on_keypress)
mouse.hook(on_mouseevent)
winsound.PlaySound("start_stop_beep.wav", winsound.SND_FILENAME)

while True:
    if stop_flag == True:
        break
    if capture_flag == True:
        with mss.mss() as sct:
            # Capture the screenshot
            sleep(0.1)
            sct_img = sct.grab(bounding)
            image = np.array(sct_img)
            # creating a image object (main image) 
        im1 = Image.fromarray(image)
        im1 = im1.save("image.png")
        capture_flag = False
        predictions = model.predict(np.array([image]))
        print(predictions, end="\r")
keyboard.unhook_all()
mouse.unhook_all()
#[][]