import keyboard
import mouse
import mss
import numpy as np
from datetime import datetime as dt
from time import sleep
import cv2
from keras.models import load_model
import winsound
from os import listdir

def mouse_callback(e):
    mouse_events.append(e)

def keyboard_callback(e):
    global stop_flag
    keyboard_events.append(e)
    if e.name == ']':
        stop_flag = True
    elif e.name == '[':
        stop_flag = False

def get_formatted_action(start_time,end_time):
    actions_dict = {
        'w':0,
        'a':0,
        's':0,
        'd':0,
        'r':0,
        'Lmouse':0
    }
    for i in keyboard_events:
        if i.time > start_time and i.time < end_time:
            if i.event_type == 'down':
                try:
                    actions_dict[i.name] = 1
                except KeyError:
                    pass
            if i.event_type == 'up':
                try:
                    actions_dict[i.name] = 0
                except KeyError:
                    pass
    x_diff = 0
    y_diff = 0
    break_flag = False
    for num,i in enumerate(mouse_events):
        if break_flag == True:
            break
        if str(i)[0] == "M" and i.time > start_time:
            for j in mouse_events[num:]:
                if j.time < end_time and str(j)[0] == 'M':
                    x_diff += j.x-640
                    y_diff += j.y-360
                else:
                    break_flag = True
                    break
        elif str(i)[0] == "B" and i.time > start_time:
            actions_dict['Lmouse'] = 1
            print('lclick')
    ret_arr = [(1,0) if actions_dict[key] else (0,1) for key in actions_dict.keys()]
    new_ret_arr = []
    for i in ret_arr:
        new_ret_arr.append(i[0])
        new_ret_arr.append(i[1])
    new_ret_arr.append(x_diff)
    new_ret_arr.append(y_diff)
    new_ret_arr.append(end_time-start_time)
    return new_ret_arr

def convert_to_int(array):
    array = list(array) #array[0]
    largest_val = max(array)
    number = array.index(largest_val)
    if number == 10:
        return "N"
    return number

def get_state_action_reward_state(miss_reward, body_reward, headshot_reward, kill_reward, HP_reward, health):
    #reward = 0

    with mss.mss() as sct:
        initial_image = sct.grab((0,0,1280,720))
    start = dt.now()
    initial_image = np.array(initial_image)
    initial_image = initial_image[:,:,:3]

    elapsed = dt.now()-start
    elapsed = elapsed.microseconds/1_000_000
    if elapsed < 0.1:
        sleep(0.3-elapsed)

    end = dt.now()
    '''with mss.mss() as sct:
        image = sct.grab((0,0,1280,720))
    image = np.array(image)
    image = image[:,:,:3]

    # get dmg region
    dmg_image = image[325:395, 605:675]

    # get health region (contains all 3 numbers)
    health_image = image[598:617, 111:147]

    gray_image = cv2.cvtColor(np.array(health_image), cv2.COLOR_RGB2GRAY)
    image_array = np.array(np.split(gray_image, 3, axis=1))

    dmg_image = np.array([dmg_image])

    dmgpred = dmgmodel.predict_on_batch(dmg_image)

    healthnum1,healthnum2,healthnum3 = healthmodel.predict_on_batch(image_array)

    prediction_arr = dmgpred[0]
    idx = np.argmax(prediction_arr)
    if idx == 0: # Miss
        reward -= miss_reward
    elif idx == 1: # Body Shot
        reward += body_reward
    elif idx == 2: # Headshot
        reward += headshot_reward
    elif idx == 3: # KILL
        reward += kill_reward

    prev_health = health
    healthnum1 = convert_to_int(healthnum1)
    healthnum2 = convert_to_int(healthnum2)
    healthnum3 = convert_to_int(healthnum3)
    num_arr = [0,0,0]

    if healthnum3 == "N":
        num_arr = [0,0]
    else:
        num_arr[2] = healthnum3
    if healthnum2 == "N":
        num_arr = [0]
    else:
        num_arr[1] = healthnum2
    if healthnum1 == "N" or healthnum1 == 0:
        health = 0
    else:
        num_arr[0] = healthnum1

    if health != 0:
        if len(num_arr) == 3:
            health = num_arr[0]*100+num_arr[1]*10+num_arr[2]
        elif len(num_arr) == 2:
            health = num_arr[0]*10+num_arr[1]
        elif len(num_arr) == 1:
            health = num_arr[0]
        done = False
    else:
        done = True
    
    if health < prev_health:
        reward -= HP_reward*(prev_health-health)
        print(health)'''

    action = get_formatted_action(start.timestamp(),end.timestamp())
    
    return initial_image, action#, reward, image, health

def save_experience(save_chunk_size):
    print("Saving...")
    saved_filenames = listdir('experience')
    max_num = 0
    for name in saved_filenames:
        nums = [int(num) for num in name.split('.')[0].split('-')]
        temp_max_num = max(nums)
        if temp_max_num > max_num:
            max_num = temp_max_num

    global state_action_reward_state_array
    state_action_reward_state_array = np.array(state_action_reward_state_array, dtype=object)
    array_len = len(state_action_reward_state_array)
    for i in range(len(state_action_reward_state_array)//save_chunk_size+1):
        np.save(f'experience/{max_num+i*save_chunk_size}-{max_num+(i+1)*save_chunk_size if save_chunk_size*(i+1) < array_len else max_num+array_len}',state_action_reward_state_array[i*save_chunk_size:(i+1)*save_chunk_size])

mouse_events = []
keyboard_events = []
dmgmodel = load_model('dmg_model.h5') # 2,519,396 params for RGBA, 1,892,196 params for RGB, ONLY MODEL NUMBER 37 AND LATER IS COMPATIBLE
healthmodel = load_model('health_model.h5') #105,119
health = 255
stop_flag = True
keyboard.hook(keyboard_callback)
mouse.hook(mouse_callback)
print('waiting')

while stop_flag==True:
    sleep(1)
stop_flag = False

print("start")
# Play a WAV file
winsound.PlaySound("start_stop_beep.wav", winsound.SND_FILENAME)
state_action_reward_state_array = []
count = 0
while stop_flag==False:
    start = dt.now()
    initial_state, action = get_state_action_reward_state(0,3,6,10,0.,health)
    state_action_reward_state_array.append([initial_state, action])
    print("time taken:",dt.now()-start)
    count += 1
    print(count)

# Play a WAV file
winsound.PlaySound("start_stop_beep.wav", winsound.SND_FILENAME)

save_experience(10)