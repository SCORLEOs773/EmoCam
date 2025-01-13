import tensorflow as tf
import cv2
import numpy as np
from ultralytics import YOLO

emotion_model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(224, 224, 3)),
    tf.keras.layers.Rescaling(1./127.5, offset=-1),
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(5, activation='softmax')
])

model = YOLO('yolov8n.pt')

cap = cv2.VideoCapture(0)

def predict_emotion_from_frame(frame):
    resized_frame = cv2.resize(frame, (224, 224))
    normalized_frame = resized_frame / 255.0
    predictions = emotion_model.predict(np.expand_dims(normalized_frame, axis=0))
    emotion = np.argmax(predictions)
    emotions = ['Happy', 'Sad', 'Angry', 'Surprised', 'Neutral']
    return emotions[emotion]

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)

    annotated_frame = results[0].plot()

    face_bbox = results[0].boxes.xywh[0]
    face_image = annotated_frame[int(face_bbox[1]):int(face_bbox[1] + face_bbox[3]),
                                 int(face_bbox[0]):int(face_bbox[0] + face_bbox[2])]

    emotion = predict_emotion_from_frame(face_image)

    cv2.putText(annotated_frame, emotion, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow('Webcam with Emotion and Object Detection', annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
