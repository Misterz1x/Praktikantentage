import cv2
from ultralytics import YOLO

# Load classification model
model = YOLO('best.pt')

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run classification
    results = model(frame)
    probs = results[0].probs  # classification probabilities

    if probs is not None:
        # Get all class probabilities as a list
        class_names = results[0].names
        confs = probs.data.tolist()  # convert tensor → Python list

        # Pair each emotion with its confidence
        emotions = list(zip(class_names.values(), confs))

        # Sort by confidence, highest first
        emotions.sort(key=lambda x: x[1], reverse=True)

        # Draw main emotion (big & green)
        top_emotion, top_conf = emotions[0]
        top_label = f"{top_emotion} ({top_conf:.2f})"
        cv2.putText(frame, top_label, (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 4, cv2.LINE_AA)

        # Draw other emotions (smaller & white)
        y_offset = 90
        for emotion, conf in emotions[1:]:
            label = f"{emotion} ({conf:.2f})"
            cv2.putText(frame, label, (40, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
            y_offset += 30

    # Show webcam feed
    cv2.imshow("Emotion Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
