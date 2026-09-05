import cv2
import mediapipe as mp
import math

from collections import deque
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# --------------------------------------------------
# PATH TO POSE MODEL
# --------------------------------------------------

MODEL_PATH = r"C:\Users\SKS\OneDrive\Desktop\FYP\models\pose_landmarker_full.task"


# --------------------------------------------------
# CREATE POSE LANDMARKER
# --------------------------------------------------

base_options = python.BaseOptions(
    model_asset_path=MODEL_PATH
)

options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_poses=1
)

pose_landmarker = vision.PoseLandmarker.create_from_options(options)


# --------------------------------------------------
# ANGLE FUNCTION
# --------------------------------------------------

def calculate_angle(point1, point2, point3):

    angle = math.degrees(
        math.atan2(
            point3[1] - point2[1],
            point3[0] - point2[0]
        )
        -
        math.atan2(
            point1[1] - point2[1],
            point1[0] - point2[0]
        )
    )

    angle = abs(angle)

    if angle > 180:
        angle = 360 - angle

    return angle


# --------------------------------------------------
# OPEN WEBCAM
# --------------------------------------------------

cap = cv2.VideoCapture(0)

window_name = "Stroke Rehabilitation - Repetition Counter"

cv2.namedWindow(
    window_name,
    cv2.WINDOW_NORMAL
)

cv2.resizeWindow(
    window_name,
    1000,
    700
)


# --------------------------------------------------
# WRIST PATH
# --------------------------------------------------

left_wrist_path = deque(maxlen=50)
right_wrist_path = deque(maxlen=50)


# --------------------------------------------------
# ANGLE SMOOTHING
# --------------------------------------------------

left_angle_history = deque(maxlen=7)


# --------------------------------------------------
# REPETITION VARIABLES
# --------------------------------------------------

left_reps = 0

left_stage = "START"

# Number of frames the arm must stay in a position
bent_frames = 0
straight_frames = 0

# Required stable frames
REQUIRED_FRAMES = 4


# --------------------------------------------------
# TIMESTAMP
# --------------------------------------------------

timestamp = 0


# --------------------------------------------------
# MAIN LOOP
# --------------------------------------------------

while True:

    ret, frame = cap.read()

    if not ret:
        print("Could not access camera")
        break


    # Mirror the camera
    frame = cv2.flip(frame, 1)


    # Convert BGR → RGB
    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


    # Convert to MediaPipe image
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )


    # Detect pose
    timestamp += 1

    results = pose_landmarker.detect_for_video(
        mp_image,
        timestamp
    )


    # --------------------------------------------------
    # PROCESS LANDMARKS
    # --------------------------------------------------

    if results.pose_landmarks:

        landmarks = results.pose_landmarks[0]

        h, w, _ = frame.shape


        # --------------------------------------------------
        # LEFT ARM
        # --------------------------------------------------

        left_shoulder = landmarks[11]
        left_elbow = landmarks[13]
        left_wrist = landmarks[15]


        # --------------------------------------------------
        # RIGHT ARM
        # --------------------------------------------------

        right_shoulder = landmarks[12]
        right_elbow = landmarks[14]
        right_wrist = landmarks[16]


        # --------------------------------------------------
        # CONVERT TO PIXELS
        # --------------------------------------------------

        ls = (
            int(left_shoulder.x * w),
            int(left_shoulder.y * h)
        )

        le = (
            int(left_elbow.x * w),
            int(left_elbow.y * h)
        )

        lw = (
            int(left_wrist.x * w),
            int(left_wrist.y * h)
        )


        rs = (
            int(right_shoulder.x * w),
            int(right_shoulder.y * h)
        )

        re = (
            int(right_elbow.x * w),
            int(right_elbow.y * h)
        )

        rw = (
            int(right_wrist.x * w),
            int(right_wrist.y * h)
        )


        # --------------------------------------------------
        # CALCULATE ELBOW ANGLES
        # --------------------------------------------------

        left_angle = calculate_angle(
            ls,
            le,
            lw
        )

        right_angle = calculate_angle(
            rs,
            re,
            rw
        )


        # --------------------------------------------------
        # SMOOTH LEFT ELBOW ANGLE
        # --------------------------------------------------

        left_angle_history.append(left_angle)

        smooth_left_angle = sum(left_angle_history) / len(
            left_angle_history
        )


        # --------------------------------------------------
        # STABLE REPETITION COUNTER
        # --------------------------------------------------

        # BENT POSITION
        if smooth_left_angle < 80:

            bent_frames += 1
            straight_frames = 0

            if bent_frames >= REQUIRED_FRAMES:
                left_stage = "BENT"


        # STRAIGHT POSITION
        elif smooth_left_angle > 150:

            straight_frames += 1
            bent_frames = 0

            if (
                straight_frames >= REQUIRED_FRAMES
                and left_stage == "BENT"
            ):

                left_stage = "STRAIGHT"

                left_reps += 1


        # MIDDLE / MOVING POSITION
        else:

            bent_frames = 0
            straight_frames = 0


        # --------------------------------------------------
        # STORE WRIST POSITIONS
        # --------------------------------------------------

        left_wrist_path.append(lw)
        right_wrist_path.append(rw)


        # --------------------------------------------------
        # DRAW ARM CONNECTIONS
        # --------------------------------------------------

        cv2.line(
            frame,
            ls,
            le,
            (0, 255, 0),
            3
        )

        cv2.line(
            frame,
            le,
            lw,
            (0, 255, 0),
            3
        )

        cv2.line(
            frame,
            rs,
            re,
            (0, 255, 0),
            3
        )

        cv2.line(
            frame,
            re,
            rw,
            (0, 255, 0),
            3
        )


        # --------------------------------------------------
        # DRAW LANDMARKS
        # --------------------------------------------------

        for point in [ls, le, lw, rs, re, rw]:

            cv2.circle(
                frame,
                point,
                8,
                (0, 255, 0),
                -1
            )


        # --------------------------------------------------
        # DRAW WRIST PATH
        # --------------------------------------------------

        for i in range(1, len(left_wrist_path)):

            cv2.line(
                frame,
                left_wrist_path[i - 1],
                left_wrist_path[i],
                (255, 0, 0),
                2
            )


        # --------------------------------------------------
        # DISPLAY ELBOW ANGLES
        # --------------------------------------------------

        cv2.putText(
            frame,
            f"Left Elbow: {int(smooth_left_angle)} deg",
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Right Elbow: {int(right_angle)} deg",
            (30, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )


        # --------------------------------------------------
        # DISPLAY REPETITIONS
        # --------------------------------------------------

        cv2.putText(
            frame,
            f"Repetitions: {left_reps}",
            (30, 125),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 255),
            3
        )


        # --------------------------------------------------
        # DISPLAY CURRENT STAGE
        # --------------------------------------------------

        cv2.putText(
            frame,
            f"Stage: {left_stage}",
            (30, 165),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )


        # --------------------------------------------------
        # DISPLAY WRIST POSITION
        # --------------------------------------------------

        cv2.putText(
            frame,
            f"Left Wrist: X={lw[0]} Y={lw[1]}",
            (30, 205),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 0),
            2
        )


    # --------------------------------------------------
    # DISPLAY CAMERA
    # --------------------------------------------------

    cv2.imshow(
        window_name,
        frame
    )


    # Press Q to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# --------------------------------------------------
# RELEASE
# --------------------------------------------------

cap.release()

pose_landmarker.close()

cv2.destroyAllWindows()