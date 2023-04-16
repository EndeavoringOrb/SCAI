import gym
from gym import spaces
import numpy as np
from time import sleep, time
import tensorflow as tf
import keyboard
import mouse
import cv2
import mss
import winsound
import win32api
import win32con
import autoit

dmgmodel = tf.keras.models.load_model('dmg_model.h5') # 2,519,396 params for RGBA, 1,892,196 params for RGB, ONLY MODEL NUMBER 37 AND LATER IS COMPATIBLE
healthmodel = tf.keras.models.load_model('health_model.h5') #105,119

class OW2Env(gym.Env):

    def __init__(self, miss_reward, body_reward, headshot_reward, kill_reward, HP_per_point_reward, enable_movement=True):
        super(OW2Env, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        low = np.concatenate((np.zeros(20), np.array([-500, -500]), np.array([0])))
        high = np.concatenate((np.ones(20), np.array([500, 500]), np.array([1])))
        self.action_space = spaces.Box(low=-1, high=1, shape=(15,), dtype=np.float32)
        # Example for using image as input (channel-first; channel-last also works):
        self.observation_space = spaces.Box(low=0, high=255, shape=(720, 1280, 3), dtype=np.uint8) # 
        self.reward = 0
        self.health = 255
        self.image = []

        self.miss_reward = miss_reward
        self.body_reward = body_reward
        self.headshot_reward = headshot_reward
        self.kill_reward = kill_reward
        self.HP_reward = HP_per_point_reward

        self.enable_movement = enable_movement
        
    def convert_to_int(self, array):
        array = list(array) #array[0]
        largest_val = max(array)
        number = array.index(largest_val)
        if number == 10:
            return "N"
        return number

    def step(self, action):
        # do actions (w0,w1,a2,a3,s4,s5,d6,d7,q8,q9,e10,e11,lshift12,lshift13,r14,r15,LM16,LM17,RM18,RM19,MOVEX,MOVEY,MOVETIME)
        if self.enable_movement:
            if action[0] >= action[1]:
                keyboard.send("w",do_press=True,do_release=False)
            else:
                keyboard.send("w",do_press=False,do_release=True)
            if action[2] >= action[3]:
                keyboard.send("a",do_press=True,do_release=False)
            else:
                keyboard.send("a",do_press=False,do_release=True)
            if action[4] >= action[5]:
                keyboard.send("s",do_press=True,do_release=False)
            else:
                keyboard.send("s",do_press=False,do_release=True)
            if action[6] >= action[7]:
                keyboard.send("d",do_press=True,do_release=False)
            else:
                keyboard.send("d",do_press=False,do_release=True)
        #if action[8] >= action[9]:
        #    keyboard.send("q",do_press=True,do_release=True)
        #if action[10] >= action[11]:
        #    keyboard.send("e",do_press=True,do_release=True)
        #if action[12] >= action[13]:
        #    keyboard.send("shift",do_press=True,do_release=True)
        if action[8] >= action[9]: #14,15
            keyboard.send("r",do_press=True,do_release=True)
        if action[10] >= action[11]: #16,17
            # Perform a left click at the current mouse position
            win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
            win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
            #mouse.press("left")
            #mouse.release("left")
        #else:
        #    mouse.release("left")
        #if action[18] >= action[19]:
        #    mouse.press("right")
        #    mouse.release("right")
        #else:
        #    mouse.release("right") ONLY FOR CASSIDY
        movemultiple = 100
        print("move x",movemultiple*action[12])
        print("move y",movemultiple*action[13])
        print("duration",(action[14]+1)/2)
        dx = movemultiple * action[12]
        dy = movemultiple * action[13]
        duration = (action[14] + 1) / 2
        start_pos = win32api.GetCursorPos()
        end_pos = (start_pos[0] + dx, start_pos[1] + dy)
        distance = ((dx ** 2) + (dy ** 2)) ** 0.5
        speed = distance / duration
        print(autoit.mouse_get_pos())
        print(int(end_pos[0]),int(end_pos[1]))
        autoit.mouse_move(int(1),int(1)) # 102, 28

        '''# Move the mouse relative to its current position
        
        
        
        steps = int(duration * 100)  # Number of steps
        step_dx = dx / steps  # Distance to move per step
        step_dy = dy / steps  # Distance to move per step

        for i in range(steps):
            x = int(start_pos[0] + (i + 1) * step_dx)
            y = int(start_pos[1] + (i + 1) * step_dy)
            win32api.SetCursorPos((x, y))
            sleep(1 / speed)

        # Check the final position of the mouse'''
        final_pos = win32api.GetCursorPos()
        print("Final position:", final_pos)
        print("Drift", (final_pos[0]-end_pos[0],final_pos[1]-end_pos[1]))
        #mouse.move(action[12]*movemultiple,action[13]*movemultiple,duration=(action[14]+1)/2) # 20,21,22



        # This uses dxcam
        #self.image = self.camera.grab((0,0,1280,720))
        #while type(self.image) == type(None):
        #    self.image = self.camera.grab((0,0,1280,720))

        # Consider adding a sleep so it doesnt screenshot same screen twice
        #sleep(0.01)

        with mss.mss() as sct:
            self.image = sct.grab((0,0,1280,720))
        self.image = np.array(self.image)
        self.image = self.image[:,:,:3]

        # get dmg region
        dmg_image = self.image[325:395, 605:675]

        # get health region (contains all 3 numbers)
        health_image = self.image[598:617, 111:147]

        gray_image = cv2.cvtColor(np.array(health_image), cv2.COLOR_RGB2GRAY)
        image_array = np.array(np.split(gray_image, 3, axis=1))

        dmg_image = np.array([dmg_image])

        dmgpred = dmgmodel.predict_on_batch(dmg_image)

        healthnum1,healthnum2,healthnum3 = healthmodel.predict_on_batch(image_array)

        prediction_arr = dmgpred[0]
        idx = np.argmax(prediction_arr)
        if idx == 0: # Miss
            self.reward -= self.miss_reward
        elif idx == 1: # Body Shot
            self.reward += self.body_reward
        elif idx == 2: # Headshot
            self.reward += self.headshot_reward
        elif idx == 3: # KILL
            self.reward += self.kill_reward

        self.prev_health = self.health
        healthnum1 = self.convert_to_int(healthnum1)
        healthnum2 = self.convert_to_int(healthnum2)
        healthnum3 = self.convert_to_int(healthnum3)
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
            self.health = 0
        else:
            num_arr[0] = healthnum1

        if self.health != 0:
            if len(num_arr) == 3:
                self.health = num_arr[0]*100+num_arr[1]*10+num_arr[2]
            elif len(num_arr) == 2:
                self.health = num_arr[0]*10+num_arr[1]
            elif len(num_arr) == 1:
                self.health = num_arr[0]
            self.done = False
        else:
            self.done = True
        
        if self.health < self.prev_health:
            self.reward -= self.HP_reward*(self.prev_health-self.health)
            print(self.health)

        info = {}

        # create observation:
        observation = self.image

        return observation, self.reward, self.done, info

    def on_keypress(self, event):
        self.stop_flag = True

    def reset(self):
        """
        keyboard.send("esc",do_press=True,do_release=True)
        sleep(2)
        mouse.move(692,458,absolute=True,duration=1)
        mouse.click()
        sleep(2)
        mouse.move(695,398,absolute=True,duration=1)
        mouse.click()
        sleep(2)
        mouse.move(1199,183,absolute=True,duration=1)
        mouse.click()
        sleep(7)
        mouse.move(1199,183,absolute=True,duration=1)
        mouse.click()
        sleep(2)
        mouse.move(693,534,absolute=True,duration=1)
        mouse.click()
        sleep(2)
        mouse.move(675,652,absolute=True,duration=1)
        mouse.click()
        sleep(40)
        mouse.move(598,557,absolute=True,duration=1)
        mouse.click()
        sleep(2)
        mouse.move(637,655,absolute=True,duration=1)
        mouse.click()
        sleep(15)
        """
        keyboard.send("w",do_press=False,do_release=True)
        keyboard.send("a",do_press=False,do_release=True)
        keyboard.send("s",do_press=False,do_release=True)
        keyboard.send("d",do_press=False,do_release=True)
        mouse.release("left")
        mouse.release("right")
        keyboard.hook_key("[",self.on_keypress)
        self.stop_flag = False
        print("Press [")
        winsound.PlaySound("start_stop_beep.wav", winsound.SND_FILENAME)
        while self.stop_flag == False:
            sleep(0.2)
        winsound.PlaySound("start_stop_beep.wav", winsound.SND_FILENAME)

        self.prev_reward = 0
        self.prev_health = 255 # Just setting it to 255 for now because i am only training on cassidy

        self.done = False

        # USES DXCAM
        #observation = self.camera.grab((0,0,1280,720))
        #while type(observation) == type(None):
        #    observation = self.camera.grab((0,0,1280,720))

        with mss.mss() as sct:
            observation = sct.grab((0,0,1280,720))
        observation = np.array(observation)
        observation = observation[:,:,:3]

        return observation

'''
print("testing...")

start = time()
env = OW2Env()
env.step(action=[0])
stop = time()

print(f"1 frame: {stop-start}")

seconds = 10

start = time()
for i in range(60*seconds):
    stuff = env.step(action=[0])
stop = time()

print(f"{60*seconds} frames: {stop-start}")
print(f"avg FPS: {(60*seconds)/(stop-start)}")
print(f"time per frame: {(stop-start)/(60*seconds)}")

start = time()
env.camera.grab((0,0,1280,720))
stop = time()

print(f"\nfull-screen grab: {stop-start}")

start = time()
env.camera.grab((605,325,675,395))
stop = time()

print(f"dmg grab: {stop-start}")

start = time()
env.camera.grab((111,598,147,617))
stop = time()

print(f"health grab: {stop-start}")

print("\nmss")
sct = mss.mss()

start = time()
sct.grab((0,0,1280,720))
stop = time()

print(f"full-screen grab: {stop-start}")

start = time()
sct.grab((605,325,675,395))
stop = time()

print(f"dmg grab: {stop-start}")

start = time()
sct.grab((111,598,147,617))
stop = time()

print(f"health grab: {stop-start}")
'''