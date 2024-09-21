import cv2
from Backend.Data_Tracking.face_detector import FaceDetector
from Backend.Data_Tracking.emotion_classifier import EmotionClassifier
from Backend.UI.ui_manager import UIManager
from Backend.Data_Tracking.data_logger import DataLogger
import glob

class SlideshowManager:
    def __init__(self, image_folder, slide_duration=10, max_slides=10):
        """
        Initializes the SlideshowManager with images from the specified folder and the duration for each slide.
        
        Args:
            image_folder (str): The path to the folder containing the slideshow images.
            slide_duration (int): Duration for each slide in seconds (default is 10 seconds).
            max_slides (int): Maximum number of slides to show (default is 10 slides).
        """
        self.image_paths = glob.glob(f"{image_folder}/*.jpg") 
        if not self.image_paths:
            raise FileNotFoundError("No images found in the folder. Make sure the folder contains images with .jpg extension.")
        
        self.current_slide_index = 0
        self.slide_duration = slide_duration 
        self.num_slides = min(len(self.image_paths), max_slides) 
        print(f"Found {self.num_slides} slides to display from folder: {image_folder}")

    def get_current_slide(self):
        """
        Returns the path to the current slide being displayed.
        
        Returns:
            str: The path to the current image file.
        """
        return self.image_paths[self.current_slide_index]

    def next_slide(self):
        """
        Advances to the next slide.
        """
        self.current_slide_index += 1

    def has_more_slides(self):
        """
        Checks if there are more slides to show.
        
        Returns:
            bool: True if there are more slides, False if the slideshow is complete.
        """
        return self.current_slide_index < self.num_slides


def run_slideshow(image_folder):

    face_detector = FaceDetector()
    emotion_classifier = EmotionClassifier()
    ui_manager = UIManager()
    data_logger = DataLogger()

    slideshow_manager = SlideshowManager(image_folder, slide_duration=1, max_slides=10)
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

            cv2.imshow('Slideshow', slide_image)

            for _ in range(slideshow_manager.slide_duration * 10):  # 10 iterations per second
                ret, frame = cap.read()
                if not ret:
                    print("Error: Could not read from webcam.")
                    break

                face, landmarks = face_detector.detect(frame)

                if face is not None and landmarks is not None:
                    emotion = emotion_classifier.predict(frame, landmarks)
                    print(f"Predicted Emotion: {emotion}") 
                    current_slide = slideshow_manager.get_current_slide()
                    ui_manager.update(frame, emotion)
                    data_logger.log(emotion, current_slide)

                if cv2.waitKey(100) & 0xFF == ord('q'):
                    print("Exiting slideshow early.")
                    cap.release()
                    cv2.destroyAllWindows()
                    return  # quit if "q" pressed

            slideshow_manager.next_slide()

        print("Slideshow has finished.")
        
    finally:
        cap.release()
        data_logger.close()
        cv2.destroyAllWindows()
        return 'emotion_log.csv', slideshow_manager.image_paths

