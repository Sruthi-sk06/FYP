import cv2
import mediapipe as mp
import random
import math
import time
import os
import pandas as pd

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "pose_landmarker_full.task"
)

CSV_PATH = os.path.join(
    BASE_DIR,
    "module 3",
    "patient_progress.csv"
)


# ============================================================
# SETTINGS
# ============================================================

GAME_TIME = 60

DIFFICULTIES = {
    "Easy": 120,
    "Medium": 220,
    "Hard": 320
}

BALLOON_RADIUS = 35

# Distance from balloon center at which wrist is considered
# to have reached the balloon.
COLLISION_DISTANCE = BALLOON_RADIUS + 20

# Small delay after a successful target before checking again.
# This prevents one held position from counting repeatedly.
TARGET_COOLDOWN = 0.35


# ============================================================
# MEDIAPIPE SETUP
# ============================================================

base_options = python.BaseOptions(
    model_asset_path=MODEL_PATH
)

options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_poses=1
)

landmarker = vision.PoseLandmarker.create_from_options(
    options
)


# ============================================================
# CAMERA
# ============================================================

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)


# ============================================================
# PATIENT INFORMATION
# ============================================================

print("\n========================================")
print("MOVENTRA - BALLOON REACH")
print("========================================")

patient_id = input(
    "Enter Patient ID (example: P003): "
).strip()

patient_name = input(
    "Enter Patient Name: "
).strip()

while True:
    session_input = input(
        "Enter Session Number: "
    ).strip()

    try:
        session_number = int(session_input)
        break
    except ValueError:
        print("Please enter a valid session number.")


print("----------------------------------------")
print(f"Patient ID   : {patient_id}")
print(f"Patient Name : {patient_name}")
print(f"Session      : {session_number}")
print("----------------------------------------")


# ============================================================
# DIFFICULTY SELECTION
# ============================================================

print("\nSelect difficulty:")
print("1 - Easy")
print("2 - Medium")
print("3 - Hard")

choice = input("Enter choice: ").strip()

if choice == "2":
    difficulty = "Medium"

elif choice == "3":
    difficulty = "Hard"

else:
    difficulty = "Easy"

print("Selected difficulty:", difficulty)


# ============================================================
# GAME VARIABLES
# ============================================================

game_started = False
game_finished = False

score = 0
targets_reached = 0
total_attempts = 0

start_time = 0
last_timestamp = 0

balloon_x = 0
balloon_y = 0

last_target_time = 0

results_saved = False
save_success = False


# ============================================================
# CREATE BALLOON
# ============================================================

def create_balloon(
    wrist_x,
    wrist_y,
    width,
    height
):

    max_distance = DIFFICULTIES[difficulty]

    angle = random.uniform(
        0,
        2 * math.pi
    )

    distance = random.randint(
        int(max_distance * 0.5),
        max_distance
    )

    x = int(
        wrist_x +
        math.cos(angle) * distance
    )

    y = int(
        wrist_y +
        math.sin(angle) * distance
    )

    # Keep balloon inside screen

    x = max(
        BALLOON_RADIUS + 10,
        min(
            width - BALLOON_RADIUS - 10,
            x
        )
    )

    y = max(
        BALLOON_RADIUS + 10,
        min(
            height - BALLOON_RADIUS - 10,
            y
        )
    )

    return x, y


# ============================================================
# DRAW BALLOON
# ============================================================

def draw_balloon(
    frame,
    x,
    y
):

    # Balloon body
    cv2.circle(
        frame,
        (x, y),
        BALLOON_RADIUS,
        (0, 0, 255),
        -1
    )

    # Balloon highlight
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
        (
            x,
            y + BALLOON_RADIUS + 25
        ),
        (255, 255, 255),
        2
    )


# ============================================================
# SAVE RESULTS TO CSV
# ============================================================

def save_balloon_result():

    global save_success

    try:

        # ----------------------------------------------------
        # Check CSV
        # ----------------------------------------------------

        if os.path.exists(CSV_PATH):

            df = pd.read_csv(CSV_PATH)

        else:

            df = pd.DataFrame()


        # ----------------------------------------------------
        # Required columns
        # ----------------------------------------------------

        required_columns = [

            "patient_id",
            "patient_name",
            "session_number",

            "balloon_score",
            "balloon_attempts",
            "balloon_successful_targets",
            "balloon_accuracy",
            "balloon_time_sec",

            "path_score",
            "path_accuracy",
            "path_checkpoints_completed",
            "path_time_sec",
            "path_distance_moved",

            "sorting_score",
            "sorting_accuracy",
            "sorting_correct",
            "sorting_wrong",
            "sorting_targets_completed",
            "sorting_time_sec",
            "sorting_distance_moved",

            "is_sample_data"
        ]


        # ----------------------------------------------------
        # Add missing columns
        # ----------------------------------------------------

        for column in required_columns:

            if column not in df.columns:

                if column in [
                    "patient_id",
                    "patient_name",
                    "is_sample_data"
                ]:

                    df[column] = ""

                else:

                    df[column] = 0


        # ----------------------------------------------------
        # Fix column data types
        # ----------------------------------------------------

        integer_columns = [

            "session_number",

            "balloon_score",
            "balloon_attempts",
            "balloon_successful_targets",

            "path_score",
            "path_checkpoints_completed",

            "sorting_score",
            "sorting_correct",
            "sorting_wrong",
            "sorting_targets_completed"
        ]

        float_columns = [

            "balloon_accuracy",
            "balloon_time_sec",

            "path_accuracy",
            "path_time_sec",
            "path_distance_moved",

            "sorting_accuracy",
            "sorting_time_sec",
            "sorting_distance_moved"
        ]


        for column in integer_columns:

            df[column] = (
                pd.to_numeric(
                    df[column],
                    errors="coerce"
                )
                .fillna(0)
                .astype(int)
            )


        for column in float_columns:

            df[column] = (
                pd.to_numeric(
                    df[column],
                    errors="coerce"
                )
                .fillna(0.0)
                .astype(float)
            )


        df["patient_id"] = (
            df["patient_id"]
            .fillna("")
            .astype(str)
        )

        df["patient_name"] = (
            df["patient_name"]
            .fillna("")
            .astype(str)
        )

        df["is_sample_data"] = (
            df["is_sample_data"]
            .fillna("NO")
            .astype(str)
        )


        # ----------------------------------------------------
        # Find existing patient + session
        # ----------------------------------------------------

        mask = (
            (df["patient_id"] == patient_id) &
            (df["session_number"] == session_number)
        )


        if mask.any():

            row_index = df.index[mask][0]

        else:

            # Create new row

            new_row = {
                column: 0
                for column in required_columns
            }

            new_row["patient_id"] = patient_id
            new_row["patient_name"] = patient_name
            new_row["session_number"] = session_number
            new_row["is_sample_data"] = "NO"

            df = pd.concat(
                [
                    df,
                    pd.DataFrame([new_row])
                ],
                ignore_index=True
            )

            row_index = df.index[-1]


        # ----------------------------------------------------
        # Update patient name
        # ----------------------------------------------------

        df.at[
            row_index,
            "patient_name"
        ] = patient_name


        # ----------------------------------------------------
        # Save Balloon Reach results
        # ----------------------------------------------------

        df.at[
            row_index,
            "balloon_score"
        ] = int(score)

        df.at[
            row_index,
            "balloon_attempts"
        ] = int(total_attempts)

        df.at[
            row_index,
            "balloon_successful_targets"
        ] = int(targets_reached)

        df.at[
            row_index,
            "balloon_accuracy"
        ] = float(accuracy)

        df.at[
            row_index,
            "balloon_time_sec"
        ] = float(GAME_TIME)


        df.at[
            row_index,
            "is_sample_data"
        ] = "NO"


        # ----------------------------------------------------
        # Save CSV
        # ----------------------------------------------------

        os.makedirs(
            os.path.dirname(CSV_PATH),
            exist_ok=True
        )

        df.to_csv(
            CSV_PATH,
            index=False
        )

        save_success = True

        print("\n========================================")
        print("BALLOON REACH RESULTS SAVED")
        print("========================================")
        print(
            f"Patient : {patient_id} - {patient_name}"
        )
        print(
            f"Session : {session_number}"
        )
        print(
            f"Score   : {score}"
        )
        print(
            f"Attempts: {total_attempts}"
        )
        print(
            f"Targets : {targets_reached}"
        )
        print(
            f"Accuracy: {accuracy:.1f}%"
        )
        print(
            f"Time    : {GAME_TIME} sec"
        )
        print(
            f"CSV     : {CSV_PATH}"
        )
        print("========================================")

        return True


    except Exception as e:

        save_success = False

        print("\n========================================")
        print("ERROR: Could not save Balloon Reach results.")
        print("========================================")
        print("Reason:", e)
        print("========================================")

        return False


# ============================================================
# GAME LOOP
# ============================================================

while cap.isOpened():

    success, frame = cap.read()

    if not success:
        break


    # Mirror camera view
    frame = cv2.flip(
        frame,
        1
    )

    height, width, _ = frame.shape


    # ========================================================
    # MEDIAPIPE
    # ========================================================

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

    if timestamp <= last_timestamp:

        timestamp = (
            last_timestamp + 1
        )

    last_timestamp = timestamp


    result = landmarker.detect_for_video(
        mp_image,
        timestamp
    )


    # ========================================================
    # FIND LEFT WRIST
    # ========================================================

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


    # ========================================================
    # START SCREEN
    # ========================================================

    if (
        not game_started
        and not game_finished
    ):

        cv2.putText(
            frame,
            "BALLOON REACH",
            (
                width // 2 - 200,
                180
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (255, 255, 255),
            3
        )

        cv2.putText(
            frame,
            f"Patient: {patient_id} - {patient_name}",
            (
                width // 2 - 250,
                235
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Session: {session_number}",
            (
                width // 2 - 120,
                275
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Difficulty: {difficulty}",
            (
                width // 2 - 150,
                320
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            "Move your LEFT hand to the balloon",
            (
                width // 2 - 280,
                380
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            "Press SPACE to START",
            (
                width // 2 - 180,
                440
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 255),
            2
        )

        cv2.putText(
            frame,
            "Press Q to QUIT",
            (
                width // 2 - 140,
                490
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )


    # ========================================================
    # START GAME
    # ========================================================

    if (
        game_started
        and not game_finished
    ):

        elapsed_time = (
            time.time() -
            start_time
        )

        remaining_time = max(
            0,
            int(
                GAME_TIME -
                elapsed_time
            )
        )


        # ----------------------------------------------------
        # Draw balloon
        # ----------------------------------------------------

        draw_balloon(
            frame,
            balloon_x,
            balloon_y
        )


        # ----------------------------------------------------
        # Collision check
        # ----------------------------------------------------

        if wrist_x is not None:

            distance = math.sqrt(

                (wrist_x - balloon_x) ** 2
                +
                (wrist_y - balloon_y) ** 2

            )


            # Count an attempt only when the player
            # actually reaches the target.

            if (
                distance <= COLLISION_DISTANCE
                and (
                    time.time() -
                    last_target_time
                ) >= TARGET_COOLDOWN
            ):

                total_attempts += 1

                targets_reached += 1

                score += 10

                last_target_time = (
                    time.time()
                )


                # Create next balloon

                balloon_x, balloon_y = (
                    create_balloon(
                        wrist_x,
                        wrist_y,
                        width,
                        height
                    )
                )


        # ----------------------------------------------------
        # Calculate accuracy
        # ----------------------------------------------------

        if total_attempts > 0:

            accuracy = (
                targets_reached /
                total_attempts
            ) * 100

        else:

            accuracy = 0.0


        # ----------------------------------------------------
        # Display information
        # ----------------------------------------------------

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
            f"Attempts: {total_attempts}",
            (30, 130),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Accuracy: {accuracy:.1f}%",
            (30, 170),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Difficulty: {difficulty}",
            (30, 210),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Time: {remaining_time}s",
            (
                width - 200,
                50
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2
        )


        # ----------------------------------------------------
        # GAME OVER
        # ----------------------------------------------------

        if elapsed_time >= GAME_TIME:

            game_started = False

            game_finished = True


    # ========================================================
    # RESULT SCREEN
    # ========================================================

    if game_finished:

        # Calculate final accuracy

        if total_attempts > 0:

            accuracy = (
                targets_reached /
                total_attempts
            ) * 100

        else:

            accuracy = 0.0


        # Save results only once

        if not results_saved:

            save_success = (
                save_balloon_result()
            )

            results_saved = True


        cv2.putText(
            frame,
            "SESSION COMPLETE",
            (
                width // 2 - 220,
                150
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.3,
            (255, 255, 255),
            3
        )

        cv2.putText(
            frame,
            f"Score: {score}",
            (
                width // 2 - 120,
                220
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Targets Reached: {targets_reached}",
            (
                width // 2 - 190,
                270
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Attempts: {total_attempts}",
            (
                width // 2 - 120,
                320
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Accuracy: {accuracy:.1f}%",
            (
                width // 2 - 130,
                370
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Difficulty: {difficulty}",
            (
                width // 2 - 150,
                420
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2
        )


        if save_success:

            cv2.putText(
                frame,
                "Results saved to dashboard",
                (
                    width // 2 - 190,
                    465
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

        else:

            cv2.putText(
                frame,
                "CSV save failed - check terminal",
                (
                    width // 2 - 220,
                    465
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )


        cv2.putText(
            frame,
            "Press R to PLAY AGAIN",
            (
                width // 2 - 190,
                525
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )

        cv2.putText(
            frame,
            "Press Q to QUIT",
            (
                width // 2 - 130,
                570
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )


    # ========================================================
    # SHOW FRAME
    # ========================================================

    cv2.imshow(
        "Balloon Reach Rehabilitation",
        frame
    )


    key = cv2.waitKey(1) & 0xFF


    # ========================================================
    # KEY CONTROLS
    # ========================================================

    # Quit

    if key == ord("q"):

        break


    # Start game

    if (
        key == 32
        and not game_started
        and not game_finished
    ):

        if wrist_x is not None:

            balloon_x, balloon_y = (
                create_balloon(
                    wrist_x,
                    wrist_y,
                    width,
                    height
                )
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

        last_target_time = 0

        results_saved = False
        save_success = False

        game_started = True
        game_finished = False


    # Restart

    if (
        key == ord("r")
        and game_finished
    ):

        score = 0
        targets_reached = 0
        total_attempts = 0

        start_time = 0
        last_target_time = 0

        results_saved = False
        save_success = False

        game_started = False
        game_finished = False


# ============================================================
# CLEANUP
# ============================================================

cap.release()

cv2.destroyAllWindows()

landmarker.close()

print("\nMOVENTRA - Balloon Reach closed.")