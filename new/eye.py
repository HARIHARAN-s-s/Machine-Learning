import cv2
import os

# Choose label
label = 'open'  # change to 'closed' when collecting closed eye images

# Create directory if not exists
save_dir = f'eye_dataset/{label}'
os.makedirs(save_dir, exist_ok=True)

cap = cv2.VideoCapture(0)
count = 0

print("Press 's' to save the eye image. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip for natural mirror view
    frame = cv2.flip(frame, 1)

    # Define manual eye ROI (Region of Interest)
    # You can adjust this ROI box to match your face
    x, y, w, h = 300, 200, 80, 40
    eye = frame[y:y+h, x:x+w]

    # Show frames
    cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
    cv2.imshow("Webcam", frame)
    cv2.imshow("Eye Region", eye)

    key = cv2.waitKey(1)
    if key == ord('s'):
        # Save image
        file_path = os.path.join(save_dir, f'{label}_{count}.jpg')
        cv2.imwrite(file_path, eye)
        print(f'Saved: {file_path}')
        count += 1
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
