import mss
import cv2
import keyboard
import mouse
import numpy as np
from time import sleep
from random import uniform
#[][][][][][][][]

input_str = ""
#input_str = input('Enter box "radius": ')
#xy = int(input_str)
#bounding = (640-xy,360-xy,640+xy,360+xy)
bounding = (110,595,150,625)
num = 0

while True:
    #input_str = input('Enter bounding box: ')
    #input_thing = tuple(map(int,input_str.split(",")))
    #bounding = (111,input_thing[0],123,input_thing[1])
    bounding = (111,598,147,617)
    print(f"waiting{num}")
    while not mouse.is_pressed("left"):
        pass
    with mss.mss() as sct:
        im_arr = []
        time = 1
        shots = 1
        # Capture the screenshot
        for i in range(shots):
            sct_img = sct.grab(bounding)
            im_arr.append(np.array(sct_img))
            sleep(time/shots)

        print(im_arr[0].shape)
        for i in range(1):
            print(f"Image {i}")
            cv2.imshow("image",im_arr[i])
            cv2.waitKey(10)
            sleep(5)
            cv2.destroyWindow("image")
    num += 1
#35
#(111,598,147,617)
# 111,123
# 123,135
# 135,147