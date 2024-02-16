import tkinter as tk
from tkinter import ttk
import time

class StopwatchApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Stopwatch")

        self.is_running = False
        self.start_time = 0

        self.time_var = tk.StringVar()
        self.time_var.set("00:00:00")

        self.label = ttk.Label(root, textvariable=self.time_var, font=("Helvetica", 48))
        self.label.pack(padx=10, pady=10)

        self.start_button = ttk.Button(root, text="Start", command=self.start_stop)
        self.start_button.pack(pady=5)

        self.reset_button = ttk.Button(root, text="Reset", command=self.reset)
        self.reset_button.pack(pady=5)

        self.update_time()

    def start_stop(self):
        if self.is_running:
            self.is_running = False
            self.start_button["text"] = "Start"
        else:
            self.is_running = True
            self.start_button["text"] = "Stop"
            self.start_time = time.time()
            self.update_time()

    def reset(self):
        self.is_running = False
        self.start_button["text"] = "Start"
        self.start_time = 0
        self.update_time()

    def update_time(self):
        if self.is_running:
            elapsed_time = time.time() - self.start_time
        else:
            elapsed_time = 0

        hours, remainder = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(remainder, 60)

        time_str = "{:02}:{:02}:{:02}".format(int(hours), int(minutes), int(seconds))
        self.time_var.set(time_str)

        if self.is_running:
            self.root.after(1000, self.update_time)

if __name__ == "__main__":
    root = tk.Tk()
    app = StopwatchApp(root)
    root.mainloop()
