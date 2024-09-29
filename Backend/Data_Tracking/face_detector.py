import dlib
import cv2

class FaceDetector:
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("/Users/isaacgutterman/CNN-Teach/Backend/Models/shape_predictor_68_face_landmarks.dat")

    def detect(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray)
        if len(faces) > 0:
            landmarks = self.predictor(gray, faces[0])  # Get landmarks for the first face
            return faces[0], landmarks
        return None, None
