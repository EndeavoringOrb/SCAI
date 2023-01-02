import mss
import keyboard
import mouse
import time
import numpy as np
import os
from skimage.transform import resize
import winsound

screenshot_frequency = 5 # number per second

# Create a function to take a screenshot and return it as a greyscale image
def take_screenshot():
    with mss.mss() as sct:
        # Capture the screenshot
        sct_img = sct.grab((0,0,1280,720))

        # Convert the screenshot to greyscale
        return np.array(sct_img)

def check_pressed(array):
    global left_down
    global right_down
    return_array = array
    if keyboard.is_pressed("w"):
        return_array[0] = 1
    if keyboard.is_pressed("a"):
        return_array[1] = 1
    if keyboard.is_pressed("s"):
        return_array[2] = 1
    if keyboard.is_pressed("d"):
        return_array[3] = 1
    if keyboard.is_pressed("shift"):
        return_array[4] = 1
    if keyboard.is_pressed("e"):
        return_array[5] = 1
    if keyboard.is_pressed("q"):
        return_array[6] = 1
    if keyboard.is_pressed("r"):
        return_array[7] = 1
    if left_down:
        return_array[8] = 1
    if right_down:
        return_array[9] = 1
    return return_array

def check_pressed_loop(screenshot_frequency,check_num,array):
    for i in range(check_num):
        array = check_pressed(array)
        time.sleep((1/screenshot_frequency)/check_num)
    return array


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
left_down = False
right_down = False
def on_mouse_event(event):
    global left_down
    mouse_events.append(event)
    if f"{event.button} {event.event_type}" == "left down":
        left_down = True
    elif f"{event.button} {event.event_type}" == "left up":
        left_down = False
    if f"{event.button} {event.event_type}" == "right down":
        right_down = True
    elif f"{event.button} {event.event_type}" == "right up":
        right_down = False

# Wait until the "[" key is pressed[]
print("Ready to start.")
while not keyboard.is_pressed("["):
    pass

# Hook mouse and keyboard
mouse.hook(on_mouse_event)
array_of_pressed_arrays = []

# Capture screenshots and record keyboard and mouse events for 5 seconds
print("Recording...")
# Play the sound file
winsound.PlaySound("start_stop_beep.wav", winsound.SND_FILENAME)
start_time = time.time()
while True:
    # Check if the "]" key is pressed to end the loop
    if keyboard.is_pressed("]"):
        mouse.unhook(on_mouse_event)
        # Play the sound file
        winsound.PlaySound("start_stop_beep.wav", winsound.SND_FILENAME)
        print("Stopped Recording.")
        break
    
    # Take a screenshot and add it to the list
    screenshot = take_screenshot()
    screenshots.append(screenshot)
    screenshot_count += 1

    #check if things are pressed
    pressed_array = [-1]*12
    pressed_array = check_pressed_loop(screenshot_frequency,10,pressed_array)
    array_of_pressed_arrays.append(pressed_array)

# Group the keyboard and mouse events by time
print("Grouping inputs...")
total_size = len(screenshots)

start_grouping = time.time()
formatted_time = time.strftime("%A, %B %d %Y %I:%M:%S %p", time.localtime(start_grouping))
print(f"Start time: {formatted_time}")

for k in range(len(array_of_pressed_arrays)):
    for i in range(len(mouse_events)):
        #if str(type(mouse_events[i]))[8:12] == "list":
        #    continue
        if ((mouse_events[i].time-start_time) >= k * 0.2) and ((mouse_events[i].time-start_time) < (k + 1) * 0.2):
            if f"{str(mouse_events[i])[0:9]}" == "MoveEvent":
                array_of_pressed_arrays[k][10] = np.tanh((mouse_events[i].x-mouse_pos[0])/200) #mouse x movement
                array_of_pressed_arrays[k][11] = np.tanh((mouse_events[i].y-mouse_pos[1])/200) #mouse y movement
                mouse_pos = (mouse_events[i].x,mouse_events[i].y)

#print how long grouping took
end_grouping = time.time()
formatted_time = time.strftime("%A, %B %d %Y %I:%M:%S %p", time.localtime(end_grouping))
print(f"End time: {formatted_time}")
try:
    print(f"Avg time per input/output: {(end_grouping-start_grouping)/total_size}")
except ZeroDivisionError:
    print(f"Average time for grouping cannot be calculated as there were no screenshots taken. The recording was stopped too quickly.")
print("-------------------------------------------")
print("")

print("Saving...")

def save_array(array, filename, num_files):
    start = time.time()
    formatted_time = time.strftime("%A, %B %d %Y %I:%M:%S %p", time.localtime(start))
    print(filename)
    print(f"Start time: {formatted_time}")
    for i in range(len(array)):
        np.save(f'{filename}_{i+num_files}', array[i])
    end = time.time()
    formatted_time = time.strftime("%A, %B %d %Y %I:%M:%S %p", time.localtime(end))
    print(f"End time: {formatted_time}")
    try:
        print(f"Avg time per input/output: {(end-start)/total_size}")
    except ZeroDivisionError:
        print(f"Average time per input/output cannot be calculated as there were no screenshots taken. The recording was stopped too quickly.")
    print("")

start_saving = time.time()

files = os.listdir('screenshot_files')
num_files = len(files)

print(f"Total groups (5 per second of recording): {total_size}")
print(f"{int(((total_size/5)-(total_size/5)%60)/60)} minutes {((total_size/5)%60):.2f} seconds of recording.")
print("")
start_saving = time.time()

save_array(screenshots,"screenshot_files/screenshots",num_files)
save_array(array_of_pressed_arrays,"output_files/outputs",num_files)

end_saving = time.time()
print(f"Total elapsed time for saving arrays: {int(((end_saving-start_saving)-(end_saving-start_saving)%60)/60)} minutes {(end_saving-start_saving)%60} seconds")
print("Finished")
print(array_of_pressed_arrays)
#[]]]]]]]]]]]][]]]]]]]]]]]