import tkinter as tk
from tkinter import filedialog
from moviepy.video.io.VideoFileClip import VideoFileClip

def extract_clip():
    file_path = filedialog.askopenfilename()
    if file_path:
        try:
            clip = VideoFileClip(file_path)
            start_time = float(entry_start.get())
            end_time = float(entry_end.get())
            if start_time < end_time:
                clip = clip.subclip(start_time, end_time)
                clip.write_videofile("extracted_clip.mp4")
                status_label.config(text="Clip extracted successfully.")
            else:
                status_label.config(text="Start time must be before end time.")
        except Exception as e:
            status_label.config(text=f"Error: {str(e)}")

# Create main window
root = tk.Tk()
root.title("Video Clip Extractor")

# Create UI elements
label_start = tk.Label(root, text="Start Time (seconds):")
label_start.grid(row=0, column=0, padx=10, pady=5, sticky="e")
entry_start = tk.Entry(root)
entry_start.grid(row=0, column=1, padx=10, pady=5)

label_end = tk.Label(root, text="End Time (seconds):")
label_end.grid(row=1, column=0, padx=10, pady=5, sticky="e")
entry_end = tk.Entry(root)
entry_end.grid(row=1, column=1, padx=10, pady=5)

browse_button = tk.Button(root, text="Browse Video", command=extract_clip)
browse_button.grid(row=2, column=0, columnspan=2, padx=10, pady=5)

status_label = tk.Label(root, text="")
status_label.grid(row=3, column=0, columnspan=2)

root.mainloop()
