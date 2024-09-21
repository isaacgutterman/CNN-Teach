import numpy as np
import cv2
import tensorflow as tf

class EmotionClassifier:
    def __init__(self):
        self.model = tf.keras.models.load_model("emotion_model_final.keras")

    def predict(self, full_image, face_landmarks):
        face_rect = face_landmarks.rect  # Get the bounding box from the landmarks
        face_image = self.extract_face(full_image, face_rect)
        face_image = self.preprocess_image(face_image)
        prediction = self.model.predict(face_image)
        emotion = np.argmax(prediction)
        return emotion

    def extract_face(self, full_image, face_rect):
        # Convert the rectangle to coordinates
        x_min = face_rect.left()
        y_min = face_rect.top()
        x_max = face_rect.right()
        y_max = face_rect.bottom()

        # Extract the face image from the full image
        face_image = full_image[y_min:y_max, x_min:x_max]
        return face_image

    def preprocess_image(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (48, 48))
        image = np.expand_dims(image, axis=-1)  
        image = np.array(image).astype('float32') / 255.0  
        image = np.expand_dims(image, axis=0) 
