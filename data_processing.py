import cv2
import os
import numpy as np

def preprocess_video(video_path, output_dir, target_size=(224, 224)):
    video_capture = cv2.VideoCapture(video_path)
    frame_count = 0
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
        frame = preprocess_frame(frame, target_size)
        cv2.imwrite(os.path.join(output_dir, f"frame_{frame_count}.jpg"), frame)
        frame_count += 1
    video_capture.release()

def preprocess_frame(frame, target_size):
    # Resize to target size
    frame = cv2.resize(frame, target_size)
    # Convert to grayscale
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Normalize pixel values
    frame = frame / 255.0
    return frame

def generate_dataset(image_dir, labels, output_dir):
    data = []
    for image_file, label in zip(os.listdir(image_dir), labels):
        image_path = os.path.join(image_dir, image_file)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        data.append((image, label))
    np.savez(output_dir, data=data)
