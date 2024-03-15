import tkinter as tk

def on_button_click():
    label.config(text="Button clicked")

# Create the main application window
app = tk.Tk()
app.title("My Tkinter App")

# Create widgets
label = tk.Label(app, text="Hello, Tkinter!")
label.pack()

button = tk.Button(app, text="Click me!", command=on_button_click)
button.pack()

# Start the main event loop
app.mainloop()
