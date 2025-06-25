import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model("eye_status_model.h5")

# Constants
IMAGE_SIZE = 64

# Initialize webcam
cap = cv2.VideoCapture(0)

print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Define region of interest (manually or use your face/eye detection later)
    x, y, w, h = 200, 100, 100, 100  # Adjust based on your webcam view
    roi = gray[y:y+h, x:x+w]

    # Preprocess the ROI
    roi_resized = cv2.resize(roi, (IMAGE_SIZE, IMAGE_SIZE))
    roi_normalized = roi_resized / 255.0
    roi_reshaped = np.reshape(roi_normalized, (1, IMAGE_SIZE, IMAGE_SIZE, 1))

    # Predict
    prediction = model.predict(roi_reshaped)
    label = "Open" if np.argmax(prediction) == 1 else "Closed"

    # Display result
    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
    cv2.putText(frame, label, (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Eye Status", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
