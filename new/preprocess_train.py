import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical # type: ignore

# Constants
IMAGE_SIZE = 64
DATASET_PATH = "eye_dataset"

def load_data():
    images = []
    labels = []

    for label_name in ["closed", "open"]:
        folder_path = os.path.join(DATASET_PATH, label_name)
        label = 0 if label_name == "closed" else 1

        for filename in os.listdir(folder_path):
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            if img is None:
                continue

            img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
            img = img / 255.0  # Normalize
            images.append(img)
            labels.append(label)

    images = np.array(images).reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 1)
    labels = to_categorical(labels, 2)  # One-hot encode: [1, 0] for closed, [0, 1] for open

    return train_test_split(images, labels, test_size=0.2, random_state=42)

# Load the data
X_train, X_test, y_train, y_test = load_data()
print("Data loaded! Training:", len(X_train), "Testing:", len(X_test))
