from pynput import keyboard

def on_press(key):
    print("KEY:", key)

listener = keyboard.Listener(on_press=on_press)
listener.start()

print("Press keys. Press Esc to quit.")

listener.join()