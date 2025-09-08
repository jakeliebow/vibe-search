import cv2
import dlib
from .lazy_loader import LazyLoader

# Face detection models
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Load face detector and landmark predictor
detector = LazyLoader(dlib.get_frontal_face_detector)
predictor = LazyLoader(dlib.shape_predictor, "shape_predictor_68_face_landmarks.dat")
# Dlib face recognition model (128D embeddings). Download from dlib's model zoo.
face_recognition_model = LazyLoader(dlib.face_recognition_model_v1, "dlib_face_recognition_resnet_model_v1.dat")
