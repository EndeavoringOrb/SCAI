import mouse
import keyboard

def print_mouse_move(event):
    print(event)

while not keyboard.is_pressed("["):
    pass

mouse.hook(print_mouse_move)

while True:
    # Check if the "]" key is pressed to end the loop
    if keyboard.is_pressed("]"):
        mouse.unhook(print_mouse_move)
        break