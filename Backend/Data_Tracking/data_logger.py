import csv
from datetime import datetime

class DataLogger:
    def __init__(self):
        self.file = open('/Users/isaacgutterman/CNN-Teach/Backend/Data_Tracking/emotion_log.csv', mode='w', newline='')
        self.writer = csv.writer(self.file)
        self.writer.writerow(['Timestamp', 'Emotion', 'Slide'])

    def log(self, emotion, slide):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.writer.writerow([timestamp, emotion, slide])
        self.file.flush()  # Ensure data is written immediately

    def close(self):
        self.file.close()
