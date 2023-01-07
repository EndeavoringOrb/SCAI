import mss
import cv2
import keyboard
import mouse
import numpy as np
from time import sleep
from random import uniform
import os
import winsound
#[][][][][][][][]

xy = 35
bounding = (640-xy,360-xy,640+xy,360+xy)
im_arr = []
label_arr = []
files = os.listdir('dmg_pics')
num_files = len(files)
stop_flag = False
capture_flag = False
click_count = 0

def on_keypress(event):
    global stop_flag
    stop_flag = True
    print("stop flag")
    winsound.PlaySound("start_stop_beep.wav", winsound.SND_FILENAME)

def on_mouseevent(event):
    global capture_flag
    global click_count
    try:
        if event.button == "left" and event.event_type == "down":
            capture_flag = True
            click_count += 1
            print(click_count,end="\r")
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
            sleep(uniform(0.1,0.4))
            sct_img = sct.grab(bounding)
            im_arr.append(np.array(sct_img))
        capture_flag = False
keyboard.unhook_all()
mouse.unhook_all()

print("\nsaving")

for i in range(len(im_arr)):
    cv2.imshow("image",im_arr[i])
    cv2.waitKey(1)
    #label_arr.append(0)]
    label_arr.append(int(input(f"{i}/{len(im_arr)}: ")))
    cv2.destroyWindow("image")
    np.save(f"dmg_pics/pic_{i+num_files}",im_arr[i])

for i in range(len(label_arr)):
    if label_arr[i] == 0:
        label_arr[i] = [1,0,0,0]
    elif label_arr[i] == 1:
        label_arr[i] = [0,1,0,0]
    elif label_arr[i] == 2:
        label_arr[i] = [0,0,1,0]
    elif label_arr[i] == 3:
        label_arr[i] = [0,0,0,1]

try:
    oldlabels = np.load("dmg_labels.npy")
    label_arr = np.concatenate((oldlabels,label_arr))
except FileNotFoundError:
    pass
np.save("dmg_labels.npy",label_arr)
count = [0,0,0,0]
for i in label_arr:
    if i[0] == 1:
        count[0] += 1
    elif i[1] == 1:
        count[1] += 1
    elif i[2] == 1:
        count[2] += 1
    elif i[3] == 1:
        count[3] += 1
print(np.array(im_arr).shape)
print(np.array(label_arr).shape)
print(f"Count: {count}")
#[][]]