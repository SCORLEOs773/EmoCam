import cv2
from ultralytics import YOLO
import numpy as np

# Function for emotion prediction using a pre-trained model
def predict_emotion_from_model(face_image):
    # Simulated emotion prediction
    # Replace this with a real emotion detection model
    emotions = ['Happy', 'Sad', 'Angry', 'Surprised', 'Neutral']
    return emotions[np.random.randint(0, len(emotions))]  # Replace with actual prediction logic

# Load YOLO model
model = YOLO('yolov8n.pt')

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Perform object detection
    results = model(frame)

    # Annotate the frame with detection results
    annotated_frame = results[0].plot()

    # Simulate face detection (replace with actual face detection logic)
    face_image = annotated_frame[100:300, 100:300]  # Example bounding box for face
    emotion = predict_emotion_from_model(face_image)

    # Annotate the frame with emotion
    cv2.putText(annotated_frame, emotion, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Display the annotated frame
    cv2.imshow('Webcam with Emotion Detection', annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
