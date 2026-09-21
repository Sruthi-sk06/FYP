import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
import os
import random
import time

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ============================================================
# MOVENTRA - OBJECT SORTING
# ============================================================

print("\n" + "=" * 60)
print("             MOVENTRA - OBJECT SORTING")
print("=" * 60)


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
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

CAM_INDEX = 0

SESSION_SECONDS = 60

WINDOW_NAME = "MOVENTRA - Object Sorting"

# MediaPipe Pose Landmarker indices
LEFT_WRIST_INDEX = 15
RIGHT_WRIST_INDEX = 16


# ============================================================
# DIFFICULTY
# ============================================================

DIFFICULTY_SETTINGS = {

    "EASY": {
        "count": 3,
        "radius": 65,
        "tolerance": 35,
        "points": 50
    },

    "MEDIUM": {
        "count": 5,
        "radius": 50,
        "tolerance": 25,
        "points": 75
    },

    "HARD": {
        "count": 7,
        "radius": 40,
        "tolerance": 18,
        "points": 100
    }
}


# ============================================================
# CSV COLUMNS
# ============================================================

REQUIRED_COLUMNS = [

    "patient_id",
    "patient_name",
    "age",
    "training_arm",
    "session_number",
    "session_date",

    "balloon_score",
    "balloon_accuracy",
    "balloon_attempts",
    "balloon_successful_targets",
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


# ============================================================
# COLORS
# ============================================================

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

GREEN = (0, 220, 0)
RED = (0, 0, 255)
BLUE = (255, 100, 0)
YELLOW = (0, 255, 255)
CYAN = (255, 255, 0)

OBJECT_COLORS = [
    (255, 80, 80),
    (80, 180, 255),
    (100, 255, 100),
    (255, 100, 255),
    (100, 255, 255),
    (255, 200, 80),
    (180, 100, 255),
    (80, 220, 180)
]


# ============================================================
# PATIENT DATA
# ============================================================

patient_id = ""
patient_name = ""
patient_age = 0
training_arm = ""
session_number = 1


# ============================================================
# GAME VARIABLES
# ============================================================

difficulty = "EASY"

objects = []
target_index = 0

score = 0
correct = 0
wrong = 0
targets_completed = 0

total_distance = 0.0

previous_wrist_original = None

game_start_time = None

last_touch_time = 0

result_saved = False


# ============================================================
# PATIENT REGISTRATION
# ============================================================

def get_patient_details():

    global patient_id
    global patient_name
    global patient_age
    global training_arm
    global session_number

    print("\nPatient Registration")
    print("-" * 40)

    # --------------------------------------------------------
    # Patient ID
    # --------------------------------------------------------

    while True:

        patient_id = input(
            "Enter Patient ID: "
        ).strip()

        if patient_id:
            break

        print("Patient ID cannot be empty.")

    # --------------------------------------------------------
    # Patient Name
    # --------------------------------------------------------

    while True:

        patient_name = input(
            "Enter Patient Name: "
        ).strip()

        if patient_name:
            break

        print("Patient Name cannot be empty.")

    # --------------------------------------------------------
    # Age
    # --------------------------------------------------------

    while True:

        try:

            patient_age = int(
                input(
                    "Enter Patient Age: "
                ).strip()
            )

            if patient_age > 0:
                break

            print("Age must be greater than 0.")

        except ValueError:

            print(
                "Please enter a valid age."
            )

    # --------------------------------------------------------
    # Training Arm
    # --------------------------------------------------------

    while True:

        arm_input = input(
            "Enter Training Arm (LEFT/RIGHT): "
        ).strip().upper()

        if arm_input in ["LEFT", "RIGHT"]:

            training_arm = arm_input

            break

        print(
            "Please enter LEFT or RIGHT."
        )

    # --------------------------------------------------------
    # Session Number
    # --------------------------------------------------------

    while True:

        try:

            session_number = int(
                input(
                    "Enter Session Number: "
                ).strip()
            )

            if session_number > 0:
                break

            print(
                "Session number must be greater than 0."
            )

        except ValueError:

            print(
                "Please enter a valid session number."
            )

    print("\n" + "-" * 50)

    print(
        "Patient      :",
        patient_name
    )

    print(
        "Patient ID   :",
        patient_id
    )

    print(
        "Age          :",
        patient_age
    )

    print(
        "Training Arm :",
        training_arm
    )

    print(
        "Session      :",
        session_number
    )

    print("-" * 50)


# ============================================================
# GENERATE OBJECTS
# ============================================================

def generate_objects(width, height):

    global objects
    global target_index

    settings = DIFFICULTY_SETTINGS[
        difficulty
    ]

    count = settings["count"]

    objects = []

    attempts = 0

    while (
        len(objects) < count
        and attempts < 500
    ):

        attempts += 1

        radius = settings["radius"]

        x = random.randint(
            radius + 50,
            max(
                radius + 51,
                width - radius - 50
            )
        )

        y = random.randint(
            radius + 130,
            max(
                radius + 131,
                height - radius - 80
            )
        )

        valid = True

        for obj in objects:

            distance = np.sqrt(
                (x - obj["x"]) ** 2
                +
                (y - obj["y"]) ** 2
            )

            if distance < (
                radius * 2 + 30
            ):

                valid = False

                break

        if valid:

            objects.append(
                {
                    "x": x,
                    "y": y,
                    "radius": radius,
                    "color": random.choice(
                        OBJECT_COLORS
                    )
                }
            )

    if objects:

        target_index = random.randint(
            0,
            len(objects) - 1
        )


# ============================================================
# CHANGE TARGET
# ============================================================

def generate_new_target():

    global target_index

    if len(objects) <= 1:
        return

    available_indices = [
        i
        for i in range(len(objects))
        if i != target_index
    ]

    target_index = random.choice(
        available_indices
    )


# ============================================================
# DRAW TEXT
# ============================================================

def draw_text(
    frame,
    text,
    position,
    scale=0.7,
    color=WHITE,
    thickness=2
):

    cv2.putText(
        frame,
        str(text),
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA
    )


# ============================================================
# DRAW OBJECTS
# ============================================================

def draw_objects(frame):

    for i, obj in enumerate(objects):

        center = (
            int(obj["x"]),
            int(obj["y"])
        )

        radius = int(
            obj["radius"]
        )

        # ----------------------------------------------------
        # Target
        # ----------------------------------------------------

        if i == target_index:

            cv2.circle(
                frame,
                center,
                radius + 12,
                YELLOW,
                4
            )

            cv2.circle(
                frame,
                center,
                radius,
                obj["color"],
                -1
            )

            draw_text(
                frame,
                "TARGET",
                (
                    center[0] - 40,
                    center[1] + 5
                ),
                0.45,
                BLACK,
                2
            )

        # ----------------------------------------------------
        # Normal object
        # ----------------------------------------------------

        else:

            cv2.circle(
                frame,
                center,
                radius,
                obj["color"],
                -1
            )


# ============================================================
# GET SELECTED WRIST
#
# IMPORTANT:
# MediaPipe processes the ORIGINAL camera frame.
#
# LEFT  = landmark 15
# RIGHT = landmark 16
# ============================================================

def get_wrist_position_original(
    result,
    width,
    height
):

    if not result.pose_landmarks:

        return None

    landmarks = result.pose_landmarks[0]

    # --------------------------------------------------------
    # IMPORTANT SIDE SELECTION
    # --------------------------------------------------------

    if training_arm == "LEFT":

        wrist_index = LEFT_WRIST_INDEX

    else:

        wrist_index = RIGHT_WRIST_INDEX

    # --------------------------------------------------------
    # Get selected landmark
    # --------------------------------------------------------

    if wrist_index >= len(landmarks):

        return None

    wrist = landmarks[
        wrist_index
    ]

    # --------------------------------------------------------
    # Convert normalized coordinates
    # to ORIGINAL camera coordinates
    # --------------------------------------------------------

    x = int(
        wrist.x * width
    )

    y = int(
        wrist.y * height
    )

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    if (
        x < 0
        or x >= width
        or y < 0
        or y >= height
    ):

        return None

    return x, y


# ============================================================
# CONVERT ORIGINAL WRIST TO MIRRORED DISPLAY
# ============================================================

def mirror_point(
    point,
    width
):

    if point is None:

        return None

    x, y = point

    mirrored_x = (
        width - 1 - x
    )

    return (
        mirrored_x,
        y
    )


# ============================================================
# UPDATE MOVEMENT DISTANCE
#
# Distance is calculated using ORIGINAL coordinates.
# ============================================================

def update_distance(
    wrist_original
):

    global previous_wrist_original
    global total_distance

    if wrist_original is None:

        return

    if previous_wrist_original is not None:

        dx = (
            wrist_original[0]
            -
            previous_wrist_original[0]
        )

        dy = (
            wrist_original[1]
            -
            previous_wrist_original[1]
        )

        distance = np.sqrt(
            dx * dx +
            dy * dy
        )

        # Ignore sudden tracking jumps
        if distance < 150:

            total_distance += distance

    previous_wrist_original = (
        wrist_original
    )


# ============================================================
# CHECK TARGET TOUCH
#
# Target positions are displayed in mirrored coordinates.
# Therefore the selected wrist must also be converted
# to mirrored coordinates before comparison.
# ============================================================

def check_target_touch(
    wrist_display
):

    global score
    global correct
    global wrong
    global targets_completed
    global last_touch_time

    if wrist_display is None:

        return

    if not objects:

        return

    current_time = time.time()

    # Prevent repeated detection
    if (
        current_time
        -
        last_touch_time
        <
        0.35
    ):

        return

    target = objects[
        target_index
    ]

    settings = DIFFICULTY_SETTINGS[
        difficulty
    ]

    # --------------------------------------------------------
    # Distance to target
    # --------------------------------------------------------

    target_distance = np.sqrt(
        (
            wrist_display[0]
            -
            target["x"]
        ) ** 2
        +
        (
            wrist_display[1]
            -
            target["y"]
        ) ** 2
    )

    # --------------------------------------------------------
    # CORRECT TARGET
    # --------------------------------------------------------

    if target_distance <= (
        target["radius"]
        +
        settings["tolerance"]
    ):

        score += settings["points"]

        correct += 1

        targets_completed += 1

        last_touch_time = (
            current_time
        )

        generate_new_target()

        return

    # --------------------------------------------------------
    # WRONG TARGET
    # --------------------------------------------------------

    for i, obj in enumerate(objects):

        if i == target_index:

            continue

        distance = np.sqrt(
            (
                wrist_display[0]
                -
                obj["x"]
            ) ** 2
            +
            (
                wrist_display[1]
                -
                obj["y"]
            ) ** 2
        )

        if distance <= obj["radius"]:

            wrong += 1

            last_touch_time = (
                current_time
            )

            break


# ============================================================
# SAVE RESULT
# ============================================================

def save_result():

    global result_saved

    if result_saved:

        return

    print(
        "\nSaving Object Sorting result..."
    )

    try:

        # ----------------------------------------------------
        # Ensure module 3 directory exists
        # ----------------------------------------------------

        os.makedirs(
            os.path.dirname(
                CSV_PATH
            ),
            exist_ok=True
        )

        # ----------------------------------------------------
        # Accuracy
        # ----------------------------------------------------

        attempts = (
            correct
            +
            wrong
        )

        if attempts > 0:

            accuracy = (
                correct
                /
                attempts
            ) * 100

        else:

            accuracy = 0.0

        # ----------------------------------------------------
        # Time
        # ----------------------------------------------------

        if game_start_time is not None:

            elapsed = (
                time.time()
                -
                game_start_time
            )

        else:

            elapsed = 0

        elapsed = min(
            elapsed,
            SESSION_SECONDS
        )

        # ----------------------------------------------------
        # Read existing CSV
        # ----------------------------------------------------

        if os.path.exists(
            CSV_PATH
        ):

            try:

                df = pd.read_csv(
                    CSV_PATH
                )

            except Exception:

                df = pd.DataFrame()

        else:

            df = pd.DataFrame()

        # ----------------------------------------------------
        # Add missing columns
        # ----------------------------------------------------

        for column in REQUIRED_COLUMNS:

            if column not in df.columns:

                df[column] = np.nan

        # ----------------------------------------------------
        # New result
        # ----------------------------------------------------

        new_data = {

            "patient_id":
                patient_id,

            "patient_name":
                patient_name,

            "age":
                patient_age,

            "training_arm":
                training_arm,

            "session_number":
                session_number,

            "session_date":
                time.strftime(
                    "%Y-%m-%d"
                ),

            "sorting_score":
                score,

            "sorting_accuracy":
                accuracy,

            "sorting_correct":
                correct,

            "sorting_wrong":
                wrong,

            "sorting_targets_completed":
                targets_completed,

            "sorting_time_sec":
                elapsed,

            "sorting_distance_moved":
                total_distance,

            "is_sample_data":
                "NO"
        }

        # ----------------------------------------------------
        # Find existing patient/session
        #
        # Case-insensitive patient ID matching
        # ----------------------------------------------------

        if not df.empty:

            patient_match = (
                df["patient_id"]
                .astype(str)
                .str.strip()
                .str.lower()
                ==
                str(patient_id)
                .strip()
                .lower()
            )

            session_match = (
                pd.to_numeric(
                    df["session_number"],
                    errors="coerce"
                )
                ==
                session_number
            )

            matching_rows = (
                patient_match
                &
                session_match
            )

        else:

            matching_rows = pd.Series(
                dtype=bool
            )

        # ----------------------------------------------------
        # UPDATE EXISTING SESSION
        # ----------------------------------------------------

        if (
            not matching_rows.empty
            and
            matching_rows.any()
        ):

            row_index = df.index[
                matching_rows
            ][0]

            for key, value in (
                new_data.items()
            ):

                df.loc[
                    row_index,
                    key
                ] = value

            print(
                "Existing patient/session updated."
            )

        # ----------------------------------------------------
        # ADD NEW SESSION
        # ----------------------------------------------------

        else:

            new_row = {}

            for column in REQUIRED_COLUMNS:

                new_row[column] = (
                    new_data.get(
                        column,
                        0
                    )
                )

            df = pd.concat(
                [
                    df,
                    pd.DataFrame(
                        [new_row]
                    )
                ],
                ignore_index=True
            )

            print(
                "New patient/session added."
            )

        # ----------------------------------------------------
        # Make sure all columns exist
        # ----------------------------------------------------

        for column in REQUIRED_COLUMNS:

            if column not in df.columns:

                df[column] = 0

        # ----------------------------------------------------
        # Arrange columns
        # ----------------------------------------------------

        df = df[
            REQUIRED_COLUMNS
        ]

        # ----------------------------------------------------
        # Save
        # ----------------------------------------------------

        df.to_csv(
            CSV_PATH,
            index=False
        )

        result_saved = True

        # ----------------------------------------------------
        # Terminal confirmation
        # ----------------------------------------------------

        print("\n" + "=" * 60)
        print(
            "          RESULT SAVED SUCCESSFULLY"
        )
        print("=" * 60)

        print(
            "Patient ID       :",
            patient_id
        )

        print(
            "Patient Name     :",
            patient_name
        )

        print(
            "Age              :",
            patient_age
        )

        print(
            "Training Arm     :",
            training_arm
        )

        print(
            "Session Number   :",
            session_number
        )

        print(
            "Sorting Score    :",
            score
        )

        print(
            "Sorting Accuracy :",
            f"{accuracy:.2f}%"
        )

        print(
            "Correct          :",
            correct
        )

        print(
            "Wrong            :",
            wrong
        )

        print(
            "Targets          :",
            targets_completed
        )

        print(
            "CSV File         :",
            CSV_PATH
        )

        print("=" * 60)

    except Exception as e:

        print(
            "\nERROR WHILE SAVING CSV:"
        )

        print(e)


# ============================================================
# MENU
# ============================================================

def show_menu(frame):

    height, width = frame.shape[:2]

    overlay = frame.copy()

    cv2.rectangle(
        overlay,
        (0, 0),
        (width, height),
        BLACK,
        -1
    )

    frame = cv2.addWeighted(
        overlay,
        0.65,
        frame,
        0.35,
        0
    )

    draw_text(
        frame,
        "MOVENTRA - OBJECT SORTING",
        (
            width // 2 - 260,
            100
        ),
        1.0,
        CYAN,
        3
    )

    draw_text(
        frame,
        "Select Difficulty",
        (
            width // 2 - 120,
            175
        ),
        0.75,
        WHITE,
        2
    )

    draw_text(
        frame,
        "1 - EASY",
        (
            width // 2 - 90,
            235
        ),
        0.7,
        GREEN,
        2
    )

    draw_text(
        frame,
        "2 - MEDIUM",
        (
            width // 2 - 90,
            285
        ),
        0.7,
        YELLOW,
        2
    )

    draw_text(
        frame,
        "3 - HARD",
        (
            width // 2 - 90,
            335
        ),
        0.7,
        RED,
        2
    )

    draw_text(
        frame,
        "Current: " + difficulty,
        (
            width // 2 - 90,
            395
        ),
        0.7,
        WHITE,
        2
    )

    draw_text(
        frame,
        "Press SPACE to start",
        (
            width // 2 - 145,
            475
        ),
        0.7,
        WHITE,
        2
    )

    draw_text(
        frame,
        "Press Q to quit",
        (
            width // 2 - 95,
            525
        ),
        0.6,
        WHITE,
        2
    )

    return frame


# ============================================================
# START SCREEN
# ============================================================

def show_start_screen(frame):

    height, width = frame.shape[:2]

    overlay = frame.copy()

    cv2.rectangle(
        overlay,
        (0, 0),
        (width, height),
        BLACK,
        -1
    )

    frame = cv2.addWeighted(
        overlay,
        0.65,
        frame,
        0.35,
        0
    )

    draw_text(
        frame,
        "GET READY!",
        (
            width // 2 - 120,
            140
        ),
        1.0,
        CYAN,
        3
    )

    draw_text(
        frame,
        "Training Arm: "
        + training_arm,
        (
            width // 2 - 150,
            215
        ),
        0.75,
        WHITE,
        2
    )

    draw_text(
        frame,
        "Difficulty: "
        + difficulty,
        (
            width // 2 - 120,
            265
        ),
        0.7,
        YELLOW,
        2
    )

    draw_text(
        frame,
        "Touch the highlighted TARGET",
        (
            width // 2 - 180,
            340
        ),
        0.7,
        WHITE,
        2
    )

    draw_text(
        frame,
        "Use your "
        + training_arm
        + " hand",
        (
            width // 2 - 150,
            390
        ),
        0.7,
        GREEN,
        2
    )

    draw_text(
        frame,
        "Press SPACE to begin",
        (
            width // 2 - 145,
            485
        ),
        0.7,
        WHITE,
        2
    )

    draw_text(
        frame,
        "Press Q to quit",
        (
            width // 2 - 95,
            535
        ),
        0.6,
        WHITE,
        2
    )

    return frame


# ============================================================
# GAME SCREEN
# ============================================================

def show_game(
    frame,
    wrist_display
):

    height, width = frame.shape[:2]

    # --------------------------------------------------------
    # Draw objects
    # --------------------------------------------------------

    draw_objects(
        frame
    )

    # --------------------------------------------------------
    # Draw selected wrist
    # --------------------------------------------------------

    if wrist_display is not None:

        cv2.circle(
            frame,
            wrist_display,
            14,
            GREEN,
            -1
        )

        cv2.circle(
            frame,
            wrist_display,
            22,
            WHITE,
            2
        )

    # --------------------------------------------------------
    # Calculate time
    # --------------------------------------------------------

    elapsed = (
        time.time()
        -
        game_start_time
    )

    remaining = max(
        0,
        SESSION_SECONDS
        -
        elapsed
    )

    attempts = (
        correct
        +
        wrong
    )

    if attempts > 0:

        accuracy = (
            correct
            /
            attempts
        ) * 100

    else:

        accuracy = 0

    # --------------------------------------------------------
    # Top information bar
    # --------------------------------------------------------

    cv2.rectangle(
        frame,
        (0, 0),
        (width, 85),
        BLACK,
        -1
    )

    draw_text(
        frame,
        "MOVENTRA",
        (20, 32),
        0.7,
        CYAN,
        2
    )

    draw_text(
        frame,
        f"Score: {score}",
        (180, 32),
        0.6,
        WHITE,
        2
    )

    draw_text(
        frame,
        f"Accuracy: {accuracy:.1f}%",
        (300, 32),
        0.6,
        WHITE,
        2
    )

    draw_text(
        frame,
        f"Correct: {correct}",
        (480, 32),
        0.6,
        GREEN,
        2
    )

    draw_text(
        frame,
        f"Wrong: {wrong}",
        (620, 32),
        0.6,
        RED,
        2
    )

    draw_text(
        frame,
        f"Time: {remaining:.0f}s",
        (750, 32),
        0.6,
        YELLOW,
        2
    )

    draw_text(
        frame,
        training_arm,
        (900, 32),
        0.6,
        CYAN,
        2
    )

    draw_text(
        frame,
        "Q = Quit",
        (1020, 32),
        0.5,
        WHITE,
        1
    )

    return frame


# ============================================================
# COMPLETION SCREEN
# ============================================================

def show_completion(frame):

    height, width = frame.shape[:2]

    overlay = np.zeros_like(
        frame
    )

    frame = cv2.addWeighted(
        frame,
        0.25,
        overlay,
        0.75,
        0
    )

    attempts = (
        correct
        +
        wrong
    )

    if attempts > 0:

        accuracy = (
            correct
            /
            attempts
        ) * 100

    else:

        accuracy = 0

    draw_text(
        frame,
        "SESSION COMPLETE",
        (
            width // 2 - 190,
            130
        ),
        1.0,
        CYAN,
        3
    )

    draw_text(
        frame,
        f"Score: {score}",
        (
            width // 2 - 80,
            205
        ),
        0.8,
        WHITE,
        2
    )

    draw_text(
        frame,
        f"Accuracy: {accuracy:.1f}%",
        (
            width // 2 - 105,
            255
        ),
        0.75,
        WHITE,
        2
    )

    draw_text(
        frame,
        f"Correct: {correct}",
        (
            width // 2 - 90,
            305
        ),
        0.7,
        GREEN,
        2
    )

    draw_text(
        frame,
        f"Wrong: {wrong}",
        (
            width // 2 - 80,
            355
        ),
        0.7,
        RED,
        2
    )

    draw_text(
        frame,
        f"Targets Completed: "
        f"{targets_completed}",
        (
            width // 2 - 165,
            405
        ),
        0.7,
        YELLOW,
        2
    )

    draw_text(
        frame,
        "Result saved successfully",
        (
            width // 2 - 160,
            475
        ),
        0.65,
        GREEN,
        2
    )

    draw_text(
        frame,
        "Press R to play again",
        (
            width // 2 - 130,
            530
        ),
        0.6,
        WHITE,
        2
    )

    draw_text(
        frame,
        "Press Q to quit",
        (
            width // 2 - 90,
            575
        ),
        0.6,
        WHITE,
        2
    )

    return frame


# ============================================================
# RESET GAME
# ============================================================

def reset_game(
    width,
    height
):

    global score
    global correct
    global wrong
    global targets_completed
    global total_distance
    global previous_wrist_original
    global game_start_time
    global result_saved
    global last_touch_time

    score = 0

    correct = 0

    wrong = 0

    targets_completed = 0

    total_distance = 0.0

    previous_wrist_original = None

    last_touch_time = 0

    game_start_time = (
        time.time()
    )

    result_saved = False

    generate_objects(
        width,
        height
    )


# ============================================================
# MAIN
# ============================================================

def main():

    global difficulty

    # --------------------------------------------------------
    # Patient registration
    # --------------------------------------------------------

    get_patient_details()

    # --------------------------------------------------------
    # Check MediaPipe model
    # --------------------------------------------------------

    if not os.path.exists(
        MODEL_PATH
    ):

        print(
            "\nERROR: MediaPipe model not found."
        )

        print(
            "Expected location:"
        )

        print(
            MODEL_PATH
        )

        input(
            "\nPress Enter to exit..."
        )

        return

    # --------------------------------------------------------
    # Create MediaPipe landmarker
    # --------------------------------------------------------

    try:

        base_options = (
            python.BaseOptions(
                model_asset_path=
                MODEL_PATH
            )
        )

        options = (
            vision.PoseLandmarkerOptions(
                base_options=
                base_options,

                running_mode=
                vision.RunningMode.IMAGE,

                num_poses=1
            )
        )

        landmarker = (
            vision.PoseLandmarker
            .create_from_options(
                options
            )
        )

    except Exception as e:

        print(
            "\nERROR creating MediaPipe:"
        )

        print(e)

        input(
            "\nPress Enter to exit..."
        )

        return

    # --------------------------------------------------------
    # Open camera
    # --------------------------------------------------------

    cap = cv2.VideoCapture(
        CAM_INDEX,
        cv2.CAP_DSHOW
    )

    if not cap.isOpened():

        cap = cv2.VideoCapture(
            CAM_INDEX
        )

    if not cap.isOpened():

        print(
            "\nERROR: Could not open camera."
        )

        landmarker.close()

        input(
            "\nPress Enter to exit..."
        )

        return

    # --------------------------------------------------------
    # Camera resolution
    # --------------------------------------------------------

    cap.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        1280
    )

    cap.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        720
    )

    # --------------------------------------------------------
    # OpenCV window
    # --------------------------------------------------------

    cv2.namedWindow(
        WINDOW_NAME,
        cv2.WINDOW_NORMAL
    )

    cv2.resizeWindow(
        WINDOW_NAME,
        1280,
        720
    )

    # --------------------------------------------------------
    # Initial state
    # --------------------------------------------------------

    state = "MENU"

    print(
        "\nCamera started successfully."
    )

    print(
        "Selected training arm:",
        training_arm
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "LEFT  -> MediaPipe landmark 15"
    )

    print(
        "RIGHT -> MediaPipe landmark 16"
    )

    try:

        while True:

            # ------------------------------------------------
            # Read ORIGINAL camera frame
            # ------------------------------------------------

            ret, original_frame = (
                cap.read()
            )

            if not ret:

                print(
                    "\nUnable to read camera frame."
                )

                break

            height, width = (
                original_frame.shape[:2]
            )

            # =================================================
            # MENU
            # =================================================

            if state == "MENU":

                # Display can be mirrored
                display_frame = (
                    cv2.flip(
                        original_frame,
                        1
                    )
                )

                display = show_menu(
                    display_frame
                )

            # =================================================
            # START SCREEN
            # =================================================

            elif state == "START":

                display_frame = (
                    cv2.flip(
                        original_frame,
                        1
                    )
                )

                display = (
                    show_start_screen(
                        display_frame
                    )
                )

            # =================================================
            # PLAYING
            # =================================================

            elif state == "PLAYING":

                # ------------------------------------------------
                # IMPORTANT:
                #
                # MediaPipe receives ORIGINAL frame.
                # We DO NOT flip before detection.
                # ------------------------------------------------

                rgb_frame = cv2.cvtColor(
                    original_frame,
                    cv2.COLOR_BGR2RGB
                )

                mp_image = mp.Image(
                    image_format=
                    mp.ImageFormat.SRGB,
                    data=rgb_frame
                )

                result = (
                    landmarker.detect(
                        mp_image
                    )
                )

                # ------------------------------------------------
                # Get anatomical selected wrist
                # ------------------------------------------------

                wrist_original = (
                    get_wrist_position_original(
                        result,
                        width,
                        height
                    )
                )

                # ------------------------------------------------
                # Calculate movement using original coordinates
                # ------------------------------------------------

                update_distance(
                    wrist_original
                )

                # ------------------------------------------------
                # Mirror ONLY the display
                # ------------------------------------------------

                display_frame = (
                    cv2.flip(
                        original_frame,
                        1
                    )
                )

                # ------------------------------------------------
                # Convert wrist position to mirrored display
                # coordinates
                # ------------------------------------------------

                wrist_display = (
                    mirror_point(
                        wrist_original,
                        width
                    )
                )

                # ------------------------------------------------
                # Check target using display coordinates
                # ------------------------------------------------

                check_target_touch(
                    wrist_display
                )

                # ------------------------------------------------
                # Draw game
                # ------------------------------------------------

                display = show_game(
                    display_frame,
                    wrist_display
                )

                # ------------------------------------------------
                # Check timer
                # ------------------------------------------------

                elapsed = (
                    time.time()
                    -
                    game_start_time
                )

                if (
                    elapsed
                    >=
                    SESSION_SECONDS
                ):

                    save_result()

                    state = "COMPLETE"

            # =================================================
            # COMPLETE
            # =================================================

            else:

                display_frame = (
                    cv2.flip(
                        original_frame,
                        1
                    )
                )

                display = (
                    show_completion(
                        display_frame
                    )
                )

            # ------------------------------------------------
            # Show camera
            # ------------------------------------------------

            cv2.imshow(
                WINDOW_NAME,
                display
            )

            # ------------------------------------------------
            # Keyboard
            # ------------------------------------------------

            key = (
                cv2.waitKey(1)
                &
                0xFF
            )

            # =================================================
            # QUIT
            # =================================================

            if key == ord("q"):

                if state == "PLAYING":

                    save_result()

                break

            # ESC
            if key == 27:

                if state == "PLAYING":

                    save_result()

                break

            # =================================================
            # MENU CONTROLS
            # =================================================

            if state == "MENU":

                if key == ord("1"):

                    difficulty = "EASY"

                elif key == ord("2"):

                    difficulty = "MEDIUM"

                elif key == ord("3"):

                    difficulty = "HARD"

                elif key == 32:

                    state = "START"

            # =================================================
            # START SCREEN CONTROLS
            # =================================================

            elif state == "START":

                if key == 32:

                    reset_game(
                        width,
                        height
                    )

                    state = "PLAYING"

            # =================================================
            # COMPLETION CONTROLS
            # =================================================

            elif state == "COMPLETE":

                if key == ord("r"):

                    reset_game(
                        width,
                        height
                    )

                    state = "PLAYING"

    except KeyboardInterrupt:

        print(
            "\nProgram interrupted."
        )

        if state == "PLAYING":

            save_result()

    except Exception as e:

        print(
            "\nUnexpected error:"
        )

        print(e)

        if state == "PLAYING":

            save_result()

    finally:

        # ----------------------------------------------------
        # Safety save
        # ----------------------------------------------------

        if state == "PLAYING":

            save_result()

        # ----------------------------------------------------
        # Cleanup
        # ----------------------------------------------------

        cap.release()

        cv2.destroyAllWindows()

        landmarker.close()

        print(
            "\nCamera closed."
        )

        print(
            "MOVENTRA Object Sorting ended."
        )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()