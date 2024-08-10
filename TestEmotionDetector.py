import sys

sys.path.append('C:\msys64\mingw64\lib\python3.10\site-packages')
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import model_from_json
import sys

sys.path.append('C:\msys64\mingw64\lib\python3.10\site-packages')

# Dictionary to map emotions to labels
emotion_dict = {
    0: "Angry", 1: "Disgusted", 2: "Fearful",
    3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"
}

# Load the model architecture and weights with error handling
try:
    # Use raw strings to avoid 'unicodeescape' errors in file paths
    json_file_path = r'C:\Users\asary\Downloads\Emotion_detection_with_CNN-main\model\emotion_model.json'
    h5_file_path = r'C:\Users\asary\Downloads\Emotion_detection_with_CNN-main\model\emotion_model.h5'

    # Load the model within a custom object scope to ensure 'Sequential' is recognized
    with tf.keras.utils.custom_object_scope({
        'Sequential': tf.keras.models.Sequential,
        # Add any other custom objects or classes if needed
    }):
        with open(json_file_path, 'r') as json_file:
            loaded_model_json = json_file.read()

        # Load model from JSON
        emotion_model = model_from_json(loaded_model_json)

        # Load model weights
        emotion_model.load_weights(h5_file_path)
        print("Model loaded successfully from disk")

except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

cap = cv2.VideoCapture(0)

# Start video capture (from a file or webcam)
#video_path = r"pratyush.mp4"  # Modify as needed
#cap = cv2.VideoCapture(video_path)

# Check if the video source is valid
if not cap.isOpened():
    print("Error: Could not open video file or webcam")
    exit(1)

# Haar cascade for face detection
face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# Process video frames
while True:
    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read frame")
        break

    frame = cv2.resize(frame, (1280, 720))  # Optional, for consistent frame size
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces with optimal parameters
    num_faces = face_detector.detectMultiScale(
        gray_frame,
        scaleFactor=1.1,  # Adjust as needed for better face detection
        minNeighbors=5,
        minSize=(30, 30)  # Minimum face size to consider
    )

    # Process each detected face
    for (x, y, w, h) in num_faces:
        # Draw rectangle around the face
        cv2.rectangle(frame, (x, y-50), (x + w, y + h + 10), (0, 255, 0), 4)

        # Extract and preprocess the face region of interest
        roi_gray_frame = gray_frame[y:y + h, x:x + w]
        cropped_img = np.expand_dims(
            np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1),
            0
        )

        # Predict emotion for the face
        emotion_prediction = emotion_model.predict(cropped_img)
        maxindex = int(np.argmax(emotion_prediction))
        emotion_text = emotion_dict[maxindex]

        # Display the predicted emotion on the frame
        cv2.putText(
            frame, emotion_text,
            (x + 5, y - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 0, 0),
            2,
            cv2.LINE_AA
        )

    # Show the video frame with emotion labels
    cv2.imshow('Emotion Detection', frame)

    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
