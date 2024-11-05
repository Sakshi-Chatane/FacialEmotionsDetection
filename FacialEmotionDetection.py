# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 23:18:44 2024

@author: Sakshi Chatane
"""

import cv2
from fer import FER

# Initialize the emotion detector
emotion_detector = FER()

# Open the webcam
cap = cv2.VideoCapture(0)

while True:
    # Capture a frame from the webcam
    ret, frame = cap.read()
    
    # Check if the frame was captured successfully
    if not ret:
        break

    # Detect emotions in the current frame
    emotions = emotion_detector.detect_emotions(frame)

    # Process each detected face and its emotions
    for result in emotions:
        # Get bounding box coordinates
        x, y, w, h = result["box"]

        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Get the dominant emotion and its score
        emotion, score = max(result["emotions"].items(), key=lambda item: item[1])
        
        # Put the emotion label on the frame
        cv2.putText(frame, f"{emotion}: {score:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Display the frame with the detected emotions
    cv2.imshow("Facial Emotion Detection", frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
