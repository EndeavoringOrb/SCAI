import mss
import keyboard
import mouse
import time
import numpy as np
import os
from skimage.transform import resize
import winsound


# Create a function to take a screenshot and return it as a greyscale image
def take_screenshot(width, height):
    with mss.mss() as sct:
        # Capture the screenshot
        #sct_img = sct.grab(sct.monitors[1])
        sct_img = sct.grab((0,0,1280,720))

        # Convert the screenshot to greyscale
        return np.array(sct_img)

# Create lists to store keyboard and mouse events
keyboard_events = []
mouse_events = []

# Create list to save screenshots to
screenshots = []

# Create a counter to keep track of the number of screenshots taken
screenshot_count = 0
width = 1920
height = 1080

# Get first mouse position
mouse_pos = mouse.get_position()
# Define hook functions
def on_mouse_event(event):
    mouse_events.append(event)
def on_keyboard_event(event):
    keyboard_events.append(event)

# Wait until the "[" key is pressed[]
print("Ready to start.")
while not keyboard.is_pressed("["):
    pass

# Hook mouse and keyboard
mouse.hook(on_mouse_event)
keyboard.hook(on_keyboard_event)

# Capture screenshots and record keyboard and mouse events for 5 seconds
print("Recording...")
# Play the sound file
winsound.PlaySound("beep.wav", winsound.SND_FILENAME)
start_time = time.time()
while True:
    # Check if the "]" key is pressed to end the loop
    if keyboard.is_pressed("]"):
        mouse.unhook(on_mouse_event)
        keyboard.unhook(on_keyboard_event)
        # Play the sound file
        winsound.PlaySound("beep.wav", winsound.SND_FILENAME)
        break

    # Get the width and height of the primary monitor
    #width = sct.monitors[1]['width']
    #height = sct.monitors[1]['height']

    # Take a screenshot and add it to the list
    screenshot = take_screenshot(width, height)
    screenshots.append(screenshot)
    screenshot_count += 1

    # Sleep for 0.2 seconds to capture screenshots at a rate of 5 per second
    time.sleep(0.2)

# Group the keyboard and mouse events by time
print("Grouping inputs...")
output_groups = []
linear_groups = []
outputs = [0]*18
for k in range(screenshot_count):
    for i in range(len(keyboard_events)):
        if str(type(keyboard_events[i]))[8:12] == "list":
            continue
        if ((keyboard_events[i].time-start_time) >= k * 0.2) and ((keyboard_events[i].time-start_time) < (k + 1) * 0.2):
            if f"{str(keyboard_events[i])[14:20]}" == "w down":
                outputs[0] = 1
            elif f"{str(keyboard_events[i])[14:20]}" == "w up":
                outputs[1] = 1
            elif f"{str(keyboard_events[i])[14:20]}" == "a down":
                outputs[2] = 1
            elif f"{str(keyboard_events[i])[14:20]}" == "a up":
                outputs[3] = 1
            elif f"{str(keyboard_events[i])[14:20]}" == "s down":
                outputs[4] = 1
            elif f"{str(keyboard_events[i])[14:20]}" == "s up":
                outputs[5] = 1
            elif f"{str(keyboard_events[i])[14:20]}" == "d down":
                outputs[6] = 1
            elif f"{str(keyboard_events[i])[14:20]}" == "d up":
                outputs[7] = 1
            elif f"{str(keyboard_events[i])[14:20]}" == "shift down":
                outputs[8] = 1
            elif f"{str(keyboard_events[i])[14:20]}" == "e down":
                outputs[9] = 1
            elif f"{str(keyboard_events[i])[14:20]}" == "q down":
                outputs[10] = 1
            elif f"{str(keyboard_events[i])[14:20]}" == "r down":
                outputs[15] = 1
    for i in range(len(mouse_events)):
        if str(type(mouse_events[i]))[8:12] == "list":
            continue
        if ((mouse_events[i].time-start_time) >= k * 0.2) and ((mouse_events[i].time-start_time) < (k + 1) * 0.2):
            try:
                if f"{mouse_events[i].button} {mouse_events[i].event_type}" == "left down":
                    outputs[11] = 1
                elif f"{mouse_events[i].button} {mouse_events[i].event_type}" == "left up":
                    outputs[12] = 1
                elif f"{mouse_events[i].button} {mouse_events[i].event_type}" == "right down":
                    outputs[13] = 1
                elif f"{mouse_events[i].button} {mouse_events[i].event_type}" == "right up":
                    outputs[14] = 1
            except AttributeError:
                if f"{str(mouse_events[i])[0:9]}" == "MoveEvent":
                    outputs[16] = np.tanh((mouse_events[i].x-mouse_pos[0])/1000) #mouse x movement
                    outputs[17] = np.tanh((mouse_events[i].y-mouse_pos[1])/1000) #mouse y movement
                    mouse_pos = (mouse_events[i].x,mouse_events[i].y)
    output_groups.append(outputs)
    outputs = [0]*18
print("Saving")

def save_array(array, filename, num_files):
    start = time.time()
    formatted_time = time.strftime("%A, %B %d %Y %I:%M:%S %p", time.localtime(start))
    print(f"Start time: {formatted_time}")
    for i in range(len(array)):
        np.save(f'{filename}_{i+num_files}', array[i])
    end = time.time()
    formatted_time = time.strftime("%A, %B %d %Y %I:%M:%S %p", time.localtime(end))
    print(f"End time: {formatted_time}")
    print(f"Avg time per input/output: {(end-start)/total_size}")

files = os.listdir('screenshot_files')
num_files = len(files)

total_size = len(screenshots)
print(f"Total groups (5 per second of recording): {total_size}")

save_array(screenshots,"screenshot_files/screenshots",num_files)
save_array(output_groups,"output_files/outputs",num_files)
print("Finished")
#[][][]][][][][]]]][][]]]