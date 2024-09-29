import time
import cv2
from Data_Tracking.face_detector import FaceDetector
from Data_Tracking.emotion_classifier import EmotionClassifier
from UI.ui_manager import UIManager
from Data_Tracking.data_logger import DataLogger
import glob

class SlideshowManager:
    def __init__(self, image_folder, slide_duration=10, max_slides=10):
        self.image_paths = glob.glob(f"{image_folder}/*.jpg")
        if not self.image_paths:
            raise FileNotFoundError("No images found in the folder. Make sure the folder contains images with .jpg extension.")
        
        self.current_slide_index = 0
        self.slide_duration = slide_duration
        self.num_slides = min(len(self.image_paths), max_slides)
        print(f"Found {self.num_slides} slides to display from folder: {image_folder}")

    def get_current_slide(self):
        return self.image_paths[self.current_slide_index]

    def next_slide(self):
        self.current_slide_index += 1

    def has_more_slides(self):
        return self.current_slide_index < self.num_slides

def run_slideshow(image_folder):
    face_detector = FaceDetector()
    emotion_classifier = EmotionClassifier()
    ui_manager = UIManager()
    data_logger = DataLogger()

    slideshow_manager = SlideshowManager(image_folder, slide_duration=2, max_slides=10)
    cap = cv2.VideoCapture(0)  # Access webcam for emotion detection

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    try:
        # Loop through slides
        while slideshow_manager.has_more_slides():
            current_slide_path = slideshow_manager.get_current_slide()
            print(f"Displaying slide: {current_slide_path}")
            slide_image = cv2.imread(current_slide_path)

            if slide_image is None:
                print(f"Error: Could not load image {current_slide_path}")
                break

            # Display the image in a window
            cv2.imshow('Slideshow', slide_image)

            # Show the slide for the specified duration
            start_time = time.time()
            while (time.time() - start_time) < slideshow_manager.slide_duration:
                ret, frame = cap.read()
                if not ret:
                    print("Error: Could not read from webcam.")
                    break

                # Detect face and emotion
                face, landmarks = face_detector.detect(frame)
                if face is not None and landmarks is not None:
                    emotion = emotion_classifier.predict(frame, landmarks)
                    print(f"Predicted Emotion: {emotion}")
                    current_slide = slideshow_manager.get_current_slide()
                    ui_manager.update(frame, emotion)
                    data_logger.log(emotion, current_slide)

                # Refresh the window
                cv2.imshow('Slideshow', slide_image)

                # Adding a small sleep to prevent CPU overload
                time.sleep(0.05)  # Sleep for 50 milliseconds

                if cv2.waitKey(1) & 0xFF == ord('q'):  # Set to 1ms wait time
                    print("Exiting slideshow early.")
                    cap.release()
                    cv2.destroyAllWindows()
                    return

            slideshow_manager.next_slide()

        print("Slideshow has finished.")
        
    finally:
        cap.release()
        data_logger.close()
        cv2.destroyAllWindows()
        return '/Users/isaacgutterman/CNN-Teach/Backend/Data_Tracking/emotion_log.csv', slideshow_manager.image_paths
