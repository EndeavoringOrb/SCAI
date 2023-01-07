import mss
import cv2
import keyboard
import mouse
import numpy as np
from time import sleep
from random import uniform
import os
import winsound
import tkinter as tk
from PIL import Image, ImageTk
#[][][][][][][][]

xy = 35
bounding = (640-xy,360-xy,640+xy,360+xy)
bounding = (0,0,1280,720)
im_arr = []
label_arr = []
files = os.listdir('pos_pics')
num_files = len(files)
stop_flag = False
capture_flag = False
click_count = 0
pos_arr = []

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
            pos = mouse.get_position()
            pos_arr.append(pos[0],pos[1])
            capture_flag = True
            print(pos)
    except AttributeError:
        pass

print("Press [ to start.")
while not keyboard.is_pressed("["):
    pass
keyboard.hook_key("]",on_keypress)
winsound.PlaySound("start_stop_beep.wav", winsound.SND_FILENAME)

while True:
    if stop_flag == True:
        break
    with mss.mss() as sct:
        # Capture the screenshot
        sct_img = sct.grab(bounding)
        im_arr.append(np.array(sct_img))
    sleep(0.1)
keyboard.unhook_all()

print("\nsaving")

try:
    oldlabels = np.load("pos_labels.npy")
    label_arr = np.concatenate((oldlabels,label_arr))
except FileNotFoundError:
    pass
np.save("pos_labels.npy",label_arr)

window = tk.Tk()  #Makes main window
window.overrideredirect(True)
window.wm_attributes("-topmost", True)
window.geometry("1280x720")
display1 = tk.Label(window)
display1.grid(row=1, column=0, padx=0, pady=0)  #Display 1
display1.pack()
count = 0

def show_frame():
    global count
    global num_files
    global im_arr
    global capture_flag
    image = im_arr[count]
    img = Image.fromarray(image)
    imgtk = ImageTk.PhotoImage(master = display1, image=img)
    display1.imgtk = imgtk #Shows frame for display 1
    display1.configure(image=imgtk)

    for i in range(5):
        while capture_flag == False:
            pass
        capture_flag = False
    label_arr.append(pos_arr)

    np.save(f"pos_pics/pic_{count+num_files}",im_arr[count])
    count += 1
    if count != len(im_arr):
        window.after(1, show_frame)

mouse.hook(on_mouseevent)

show_frame()
window.mainloop()

mouse.unhook_all()

#[][]