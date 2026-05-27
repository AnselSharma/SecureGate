from ultralytics import YOLO
from config import MODEL_NAME

# Load YOLO model
model = YOLO(MODEL_NAME)

def detect_people(frame):

    # Run YOLO
    results = model(frame)[0]

    return results