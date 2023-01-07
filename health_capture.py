import keyboard
from time import sleep
import mss
import numpy as np
import cv2
import os
import winsound

def take_screenshot(bounding_box=(0,0,1280,720)):
    with mss.mss() as sct:
        # Capture the screenshot
        sct_img = sct.grab(bounding_box)

        return np.array(sct_img)
#bounding_box = tuple([int(i) for i in input("TLX,TLY,W,H: ").split(",")])

recording = False
break_flag = False

while break_flag == False:
    images = []
    labels = []

    #(111,598,147,617)
    # 111,123
    # 123,135
    # 135,147
    box1 = (111,598,123,617)
    box2 = (123,598,135,617)
    box3 = (135,598,147,617)
    box_num = input("Enter Box Num: ")
    shot_nom = int(input("Enter the number of screenshots: "))
    pic_val = int(input("Enter what number will be displayed in the box the entire recording: "))
    if box_num == "1":
        box = box1
    elif box_num == "2":
        box = box2
    elif box_num == "3":
        box = box3

    print("Press [ to start.")
    while not keyboard.is_pressed("["):
        pass
    recording = True

    print("Recording...")
    winsound.PlaySound("start_stop_beep.wav", winsound.SND_FILENAME)
    while len(images)<shot_nom:
            im = take_screenshot(box)
            images.append(im)
            sleep(0.1)
    winsound.PlaySound("start_stop_beep.wav", winsound.SND_FILENAME)
    print("saving")
    print(len(images))
    length = len(images)
    files = os.listdir('health_pics')
    num_files = len(files)

    for i in range(length):
        # Convert the image to grayscale
        gray_image = cv2.cvtColor(images[i], cv2.COLOR_BGR2GRAY)
        gray_image = np.reshape(gray_image,(19,12,1))
        #cv2.imshow(f"image {i+num_files}/{length}",gray_image)
        label_len = len(labels)
        #while label_len == len(labels):
            #cv2.waitKey(1)
            #labels.append(int(input("Input number shown: ")))
        labels.append(pic_val)
        np.save(f"health_pics/pic_{i+num_files}",gray_image)
        #cv2.destroyWindow(f"image {i+num_files}/{length}")

    for i in range(len(labels)):
        if labels[i] == 0:
            labels[i] = [1,0,0,0,0,0,0,0,0,0,0]
        elif labels[i] == 1:
            labels[i] = [0,1,0,0,0,0,0,0,0,0,0]
        elif labels[i] == 2:
            labels[i] = [0,0,1,0,0,0,0,0,0,0,0]
        elif labels[i] == 3:
            labels[i] = [0,0,0,1,0,0,0,0,0,0,0]
        elif labels[i] == 4:
            labels[i] = [0,0,0,0,1,0,0,0,0,0,0]
        elif labels[i] == 5:
            labels[i] = [0,0,0,0,0,1,0,0,0,0,0]
        elif labels[i] == 6:
            labels[i] = [0,0,0,0,0,0,1,0,0,0,0]
        elif labels[i] == 7:
            labels[i] = [0,0,0,0,0,0,0,1,0,0,0]
        elif labels[i] == 8:
            labels[i] = [0,0,0,0,0,0,0,0,1,0,0]
        elif labels[i] == 9:
            labels[i] = [0,0,0,0,0,0,0,0,0,1,0]
        elif labels[i] == 10:
            labels[i] = [0,0,0,0,0,0,0,0,0,0,1]

    try:
        oldlabels = np.load("health_labels.npy")
        labels = np.concatenate((oldlabels,labels))
    except FileNotFoundError:
        pass

    np.save("health_labels.npy",labels)

    count = [0,0,0,0,0,0,0,0,0,0,0]
    for i in labels:
        if i[0] == 1:
            count[0] += 1
        elif i[1] == 1:
            count[1] += 1
        elif i[2] == 1:
            count[2] += 1
        elif i[3] == 1:
            count[3] += 1
        elif i[4] == 1:
            count[4] += 1
        elif i[5] == 1:
            count[5] += 1
        elif i[6] == 1:
            count[6] += 1
        elif i[7] == 1:
            count[7] += 1
        elif i[8] == 1:
            count[8] += 1
        elif i[9] == 1:
            count[9] += 1
        elif i[10] == 1:
            count[10] += 1
    
    print(f"Count: {count}")
    print(f"IMAGES: {len(images)+num_files}")
    print(f"LABELS: {len(labels)}")
    break
#cv2.imshow("image",im)
#cv2.waitKey()
#110,595,150,625
#300,300,700,700

#3,5,8