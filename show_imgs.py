import mss
import cv2
import keyboard
import numpy as np
#[][][][][]

while True:
    while not keyboard.is_pressed("["):
        pass
    with mss.mss() as sct:
        # Capture the screenshot
        #sct_img = sct.grab(sct.monitors[1])[][]
        sct_img = sct.grab((0,0,1280,720))
        im = np.array(sct_img)
        print(im.shape)

        cv2.imshow("image",im)
        cv2.waitKey(10000)