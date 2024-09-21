import cv2
import tkinter as tk

class UIManager:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("QuickGaze")
        self.root.geometry("640x480")

    def update(self, frame, emotion):
        # Convert the frame to an image that can be displayed in tkinter
        cv2.imshow("Webcam Feed", frame)
        print(f"Predicted Emotion: {emotion}")

    def should_quit(self):
        return cv2.waitKey(1) & 0xFF == ord('q') 
