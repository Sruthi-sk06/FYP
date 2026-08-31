import cv2
from mediapipe.python.solutions import pose
from mediapipe.python.solutions import drawing_utils

# Create MediaPipe Pose detector
pose_detector = pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Open webcam
cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()

    if not ret:
        print("Could not access camera")
        break

    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect body landmarks
    results = pose_detector.process(rgb_frame)

    # Draw body landmarks
    if results.pose_landmarks:
        drawing_utils.draw_landmarks(
            frame,
            results.pose_landmarks,
            pose.POSE_CONNECTIONS
        )

    # Display camera
    cv2.imshow("Stroke Rehabilitation - Pose Detection", frame)

    # Press Q to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
pose_detector.close()
cv2.destroyAllWindows()