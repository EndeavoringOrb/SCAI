import mss
import tensorflow as tf
import keyboard
import mouse
import numpy as np
import tkinter as tk

action_threshold = 0.9
width = 1920
height = 1080
bounding = (0,0,1280,720)
probabilities = [0.3]*16
labels = ["w↓","w↑","a↓","a↑","s↓","s↑","d↓","d↑","shift","e","q","LM↓","LM↑","RM↓","RM↑","r"]

def take_screenshot():
    with mss.mss() as sct:
        # Capture the screenshot
        sct_img = sct.grab(sct.monitors[1])
        sct_img = sct.grab((0,0,1280,720))

        # Convert the screenshot to greyscale
        return np.array(sct_img)

model = tf.keras.models.load_model('model.h5')

# Create the main window
window = tk.Tk()

# Set the title and size of the window
window_height = 300
window_width = 450
offset_x = 100 # 1450
offset_y = 200
window.title("My Window")
window.geometry(f"{window_width}x{window_height}+{offset_x}+{offset_y}")
#window.attributes("-alpha", 1)
#window.overrideredirect(True)

# Create a canvas to draw the bar chart on
canvas = tk.Canvas(window, width=window_width, height=window_height)
canvas.pack()

# Make window draggable
def start_move(event):
    window.x = event.x
    window.y = event.y

def stop_move():
    window.x = None
    window.y = None

def do_move(event):
    deltax = event.x - window.x
    deltay = event.y - window.y
    x = window.winfo_x() + deltax
    y = window.winfo_y() + deltay
    window.geometry(f"+{x}+{y}")

window.bind("<ButtonPress-1>", start_move)
window.bind("<ButtonRelease-1>", stop_move)
window.bind("<B1-Motion>", do_move)

bar_width = window_width / 16

def update_bar_chart():
    global probabilities
    global labels
    global bar_width
    global window_height
    global window_width
    print("update")
    # Clear the canvas
    canvas.delete("all")

    # Draw the bars and labels on the canvas
    for i in range(len(probabilities)):
        x1 = i * bar_width
        y1 = ((window_height)/2) - (((window_height)/2)*probabilities[i])
        x2 = x1 + bar_width
        y2 = ((window_height)/2)
        canvas.create_rectangle(x1, y1, x2, y2, fill="green")
        canvas.create_text(x1 + bar_width/2, y2, text=f"{labels[i]}", font=("Arial", 8), anchor="n",fill="black")
        canvas.create_text(x1 + bar_width/2, y1, text=f"{probabilities[i]:.2f}", font=("Arial", 8), anchor="s",fill="black")
    canvas.create_line(0,window_height - ((window_height)*action_threshold),window_width,window_height - ((window_height)*action_threshold),fill="red")
    # Move the window to the top
    window.wm_attributes("-topmost", True)

def play_game():
    global probabilities
    # Check if the "]" key is pressed to end the loop
    if keyboard.is_pressed("]"):
        #window.destroy()
        pass
    
    # Predict actions
    predictions = model.predict(np.array([take_screenshot()]))
    predictions = predictions[0]

    probabilities = predictions[0:16]

    #update_bar_chart(predictions,["w↓","w↑","a↓","a↑","s↓","s↑","d↓","d↑","shift","e","q","LM↓","LM↑","RM↓","RM↑","r"])

    # Act on that prediction
    if (predictions[0]+1)/2 >= action_threshold:
        keyboard.send('w',do_press=True,do_release=False)
        print("w")
    if (predictions[1]+1)/2 >= action_threshold:
        keyboard.send('w',do_press=False,do_release=True)
    if (predictions[2]+1)/2 >= action_threshold:
        keyboard.send('a',do_press=True,do_release=False)
        print("a")
    if (predictions[3]+1)/2 >= action_threshold:
        keyboard.send('a',do_press=False,do_release=True)
    if (predictions[4]+1)/2 >= action_threshold:
        keyboard.send('s',do_press=True,do_release=False)
        print("s")
    if (predictions[5]+1)/2 >= action_threshold:
        keyboard.send('s',do_press=False,do_release=True)
    if (predictions[6]+1)/2 >= action_threshold:
        keyboard.send('d',do_press=True,do_release=False)
        print("d")
    if (predictions[7]+1)/2 >= action_threshold:
        keyboard.send('d',do_press=False,do_release=True)
    if (predictions[8]+1)/2 >= action_threshold:
        keyboard.send('shift',do_press=True,do_release=True)
        print("shift")
    if (predictions[9]+1)/2 >= action_threshold:
        keyboard.send('e',do_press=True,do_release=True)
        print("e")
    if (predictions[10]+1)/2 >= action_threshold:
        keyboard.send('q',do_press=True,do_release=True)
        print("q")
    if (predictions[11]+1)/2 >= action_threshold:
        mouse.press(button="left")
        print("left down")
    if (predictions[12]+1)/2 >= action_threshold:
        mouse.release(button="left")
        print("left up")
    if (predictions[13]+1)/2 >= action_threshold:
        mouse.press(button="right")
        print("right down")
    if (predictions[14]+1)/2 >= action_threshold:
        mouse.release(button="right")
        print("right up")
    if (predictions[15]+1)/2 >= action_threshold:
        keyboard.send('r',do_press=True,do_release=False)
        print("r")

    #mouse.move(np.arctanh(predictions[16])*1000, np.arctanh(predictions[17])*1000, absolute=False, duration=0.2)

def run_funcs():
    play_game()
    update_bar_chart()
    window.after(1,run_funcs)

# Run the Tkinter event loop
window.after(1,run_funcs())
window.mainloop()
#