import tkinter as tk
import random

width = 1920
height = 1080
bounding = (0,0,1280,720)
probabilities = [0.3]*16
labels = ["w↓","w↑","a↓","a↑","s↓","s↑","d↓","d↑","shift","e","q","LM↓","LM↑","RM↓","RM↑","r"]
action_threshold = 0.9

# Create the main window
window = tk.Tk()

# Set the title and size of the window
window_height = 300
window_width = 450
offset_x = 0 # 1450
offset_y = 0 # 200
window.title("My Window")
window.geometry(f"{window_width}x{window_height}+{offset_x}+{offset_y}")
window.overrideredirect(True)

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
        y1 = ((window_height-15)/2) - (((window_height-15)/2)*probabilities[i])
        x2 = x1 + bar_width
        y2 = ((window_height-15)/2)
        canvas.create_rectangle(x1, y1, x2, y2, fill="green")
        canvas.create_text(x1 + bar_width/2, y2, text=f"{labels[i]}", font=("Arial", 8), anchor="n",fill="black")
        canvas.create_text(x1 + bar_width/2, y1, text=f"{probabilities[i]:.2f}", font=("Arial", 8), anchor="s",fill="black")
    canvas.create_line(0,window_height-15 - ((window_height-15)*action_threshold),window_width,window_height-15 - ((window_height-15)*action_threshold),fill="red")
    # Move the window to the top
    window.wm_attributes("-topmost", True)

def run_funcs():
    global probabilities
    probabilities = [random.random()*2-1 for i in range(16)]
    update_bar_chart()
    window.after(1,run_funcs)

# Run the Tkinter event loop
window.after(1,run_funcs())
window.mainloop()
#