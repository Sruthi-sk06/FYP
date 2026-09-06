import cv2
import mediapipe as mp
import random
import math
import time

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# =========================
# SETTINGS
# =========================

MODEL_PATH = "models/pose_landmarker_full.task"

GAME_TIME = 60

# Difficulty settings
DIFFICULTIES = {
    "Easy": 120,
    "Medium": 220,
    "Hard": 320
}

# Balloon size
BALLOON_RADIUS = 35


# =========================
# MEDIAPIPE SETUP
# =========================

base_options = python.BaseOptions(
    model_asset_path=MODEL_PATH
)

options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_poses=1
)

landmarker = vision.PoseLandmarker.create_from_options(options)


# =========================
# CAMERA
# =========================

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)


# =========================
# VARIABLES
# =========================

game_started = False
game_finished = False

difficulty = "Easy"

score = 0
targets_reached = 0
total_attempts = 0

start_time = 0
last_timestamp = 0

balloon_x = 0
balloon_y = 0


# =========================
# DIFFICULTY SELECTION
# =========================

print("\n==============================")
print("BALLOON REACH REHABILITATION")
print("==============================")
print("Select difficulty:")
print("1 - Easy")
print("2 - Medium")
print("3 - Hard")

choice = input("Enter choice: ")

if choice == "2":
    difficulty = "Medium"

elif choice == "3":
    difficulty = "Hard"

else:
    difficulty = "Easy"

print("Selected difficulty:", difficulty)


# =========================
# CREATE BALLOON
# =========================

def create_balloon(wrist_x, wrist_y, width, height):

    max_distance = DIFFICULTIES[difficulty]

    # Generate a random angle
    angle = random.uniform(0, 2 * math.pi)

    # Generate a random distance
    distance = random.randint(
        int(max_distance * 0.5),
        max_distance
    )

    x = int(
        wrist_x + math.cos(angle) * distance
    )

    y = int(
        wrist_y + math.sin(angle) * distance
    )

    # Keep balloon inside screen
    x = max(
        BALLOON_RADIUS + 10,
        min(width - BALLOON_RADIUS - 10, x)
    )

    y = max(
        BALLOON_RADIUS + 10,
        min(height - BALLOON_RADIUS - 10, y)
    )

    return x, y


# =========================
# DRAW BALLOON
# =========================

def draw_balloon(frame, x, y):

    # Balloon body
    cv2.circle(
        frame,
        (x, y),
        BALLOON_RADIUS,
        (0, 0, 255),
        -1
    )

    # Highlight
    cv2.circle(
        frame,
        (x - 10, y - 10),
        7,
        (255, 255, 255),
        -1
    )

    # Balloon string
    cv2.line(
        frame,
        (x, y + BALLOON_RADIUS),
        (x, y + BALLOON_RADIUS + 25),
        (255, 255, 255),
        2
    )


# =========================
# GAME LOOP
# =========================

while cap.isOpened():

    success, frame = cap.read()

    if not success:
        break

    frame = cv2.flip(frame, 1)

    height, width, _ = frame.shape

    # =========================
    # MEDIAPIPE
    # =========================

    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    timestamp = int(
        time.time() * 1000
    )

    # Ensure timestamp always increases
    if timestamp <= last_timestamp:
        timestamp = last_timestamp + 1

    last_timestamp = timestamp

    result = landmarker.detect_for_video(
        mp_image,
        timestamp
    )

    # =========================
    # FIND WRIST
    # =========================

    wrist_x = None
    wrist_y = None

    if result.pose_landmarks:

        landmarks = result.pose_landmarks[0]

        # Left wrist = landmark 15
        wrist = landmarks[15]

        wrist_x = int(
            wrist.x * width
        )

        wrist_y = int(
            wrist.y * height
        )

        # Draw wrist
        cv2.circle(
            frame,
            (wrist_x, wrist_y),
            12,
            (0, 255, 0),
            -1
        )

    # =========================
    # START SCREEN
    # =========================

    if not game_started and not game_finished:

        cv2.putText(
            frame,
            "BALLOON REACH",
            (width // 2 - 200, 200),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (255, 255, 255),
            3
        )

        cv2.putText(
            frame,
            f"Difficulty: {difficulty}",
            (width // 2 - 150, 260),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            "Move your LEFT hand to the balloon",
            (width // 2 - 280, 330),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            "Press SPACE to START",
            (width // 2 - 180, 400),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 255),
            2
        )

        cv2.putText(
            frame,
            "Press Q to QUIT",
            (width // 2 - 140, 450),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

    # =========================
    # START GAME
    # =========================

    if game_started and not game_finished:

        elapsed_time = time.time() - start_time

        remaining_time = max(
            0,
            int(GAME_TIME - elapsed_time)
        )

        # =========================
        # DRAW BALLOON
        # =========================

        draw_balloon(
            frame,
            balloon_x,
            balloon_y
        )

        # =========================
        # CHECK COLLISION
        # =========================

        if wrist_x is not None:

            distance = math.sqrt(
                (wrist_x - balloon_x) ** 2
                +
                (wrist_y - balloon_y) ** 2
            )

            # Wrist reached balloon
            if distance < BALLOON_RADIUS + 20:

                score += 10
                targets_reached += 1

                # Create next target
                balloon_x, balloon_y = create_balloon(
                    wrist_x,
                    wrist_y,
                    width,
                    height
                )

        # =========================
        # DISPLAY INFORMATION
        # =========================

        cv2.putText(
            frame,
            f"Score: {score}",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Targets: {targets_reached}",
            (30, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Difficulty: {difficulty}",
            (30, 130),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Time: {remaining_time}s",
            (width - 200, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2
        )

        # =========================
        # GAME OVER
        # =========================

        if elapsed_time >= GAME_TIME:

            game_started = False
            game_finished = True

    # =========================
    # RESULT SCREEN
    # =========================

    if game_finished:

        # Calculate accuracy
        if total_attempts > 0:
            accuracy = (
                targets_reached /
                total_attempts
            ) * 100
        else:
            accuracy = 0

        cv2.putText(
            frame,
            "SESSION COMPLETE",
            (width // 2 - 220, 180),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.3,
            (255, 255, 255),
            3
        )

        cv2.putText(
            frame,
            f"Score: {score}",
            (width // 2 - 120, 260),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Targets Reached: {targets_reached}",
            (width // 2 - 180, 310),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Difficulty: {difficulty}",
            (width // 2 - 150, 360),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            "Press R to PLAY AGAIN",
            (width // 2 - 190, 440),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )

        cv2.putText(
            frame,
            "Press Q to QUIT",
            (width // 2 - 130, 490),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

    # =========================
    # SHOW FRAME
    # =========================

    cv2.imshow(
        "Balloon Reach Rehabilitation",
        frame
    )

    key = cv2.waitKey(1) & 0xFF

    # =========================
    # KEY CONTROLS
    # =========================

    if key == ord("q"):
        break

    # Start
    if key == 32 and not game_started and not game_finished:

        if wrist_x is not None:

            balloon_x, balloon_y = create_balloon(
                wrist_x,
                wrist_y,
                width,
                height
            )

        else:

            balloon_x = random.randint(
                100,
                width - 100
            )

            balloon_y = random.randint(
                100,
                height - 100
            )

        score = 0
        targets_reached = 0
        total_attempts = 0

        start_time = time.time()

        game_started = True
        game_finished = False

    # Restart
    if key == ord("r") and game_finished:

        score = 0
        targets_reached = 0
        total_attempts = 0

        game_started = False
        game_finished = False


# =========================
# CLEANUP
# =========================

cap.release()
cv2.destroyAllWindows()
landmarker.close()