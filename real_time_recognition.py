import cv2
import numpy as np
from model import build_model

def load_model(model_path):
    return tf.keras.models.load_model(model_path)

def preprocess_frame(frame, target_size=(224, 224)):
    # Resize to target size
    frame = cv2.resize(frame, target_size)
    # Convert to grayscale
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Normalize pixel values
    frame = frame / 255.0
    # Expand dimensions to match model input shape
    frame = np.expand_dims(frame, axis=-1)
    frame = np.expand_dims(frame, axis=0)
    return frame

def recognize_gesture(frame, model):
    predictions = model.predict(frame)
    # Post-process predictions
    return predictions.argmax()

def main():
    model_path = 'path/to/your/model.h5'
    model = load_model(model_path)
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        preprocessed_frame = preprocess_frame(frame)
        prediction = recognize_gesture(preprocessed_frame, model)
        # Display prediction on frame
        cv2.putText(frame, f"Prediction: {prediction}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Air Writing Recognition', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
