import os
import cv2
import numpy as np
from app.utils.file_operations import save_image
from app.utils.error_handling import handle_error

class AddPerson:
    def __init__(self, embeddings_db):
        self.embeddings_db = embeddings_db

    def add(self, student_id, images):
        try:
            embeddings = []
            for i, image in enumerate(images):
                embedding = self._extract_embedding(image)
                embeddings.append(embedding)
            self.embeddings_db[student_id] = embeddings
        except Exception as e:
            handle_error(e)

    def delete(self, student_id):
        try:
            if student_id in self.embeddings_db:
                del self.embeddings_db[student_id]
        except Exception as e:
            handle_error(e)

    def _extract_embedding(self, image):
        # Placeholder for embedding extraction logic
        return np.random.rand(128)  # Random embedding for demonstration

class FaceTrackerRecognizer:
    def __init__(self, camera_id):
        self.camera_id = camera_id
        self.cap = cv2.VideoCapture(camera_id)
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()

    def start(self):
        self.cap.open(self.camera_id)
        if not self.cap.isOpened():
            raise Exception("Camera could not be opened")

    def get_frames(self):
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            yield cv2.imencode('.jpg', frame)[1].tobytes()

    def stop(self):
        self.cap.release()