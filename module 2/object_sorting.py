"""
=====================================================================
MOVENTRA - AR/VR Rehabilitation System for Stroke Patients
Module 2: Gamified Rehabilitation
Game 3: OBJECT SORTING

Camera-based, non-immersive upper-limb rehabilitation game.

The patient uses the LEFT hand/wrist to select the object that
matches the target color.

Tech stack:
    Python
    OpenCV
    MediaPipe Pose Landmarker Tasks API
    Pandas

Controls:
    1 / 2 / 3  -> Easy / Medium / Hard
    SPACE      -> Start
    R          -> Play again
    Q          -> Quit
=====================================================================
"""

import cv2
import time
import math
import os
import sys
import random
import numpy as np
import tkinter as tk
import pandas as pd

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ================================================================
# PROJECT PATHS
# ================================================================

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


# ================================================================
# CONFIGURATION
# ================================================================

CAM_INDEX = 0

SESSION_SECONDS = 60

WINDOW_NAME = "MOVENTRA - Object Sorting"

FEEDBACK_SECONDS = 0.6

LEFT_WRIST_INDEX = 15


# ================================================================
# COLORS - BGR FORMAT
# ================================================================

COLOR_BG_TEXT = (255, 255, 255)

COLOR_WRIST_NEUTRAL = (255, 255, 255)

COLOR_CORRECT = (0, 255, 0)

COLOR_WRONG = (0, 0, 255)

COLOR_WARNING = (0, 0, 255)


COLOR_PALETTE = {

    "RED": (0, 0, 255),

    "GREEN": (0, 200, 0),

    "BLUE": (255, 120, 0),

    "YELLOW": (0, 220, 220),

    "PURPLE": (200, 0, 200),

    "ORANGE": (0, 140, 255),

    "CYAN": (255, 255, 0),

    "PINK": (200, 150, 255),

}


# ================================================================
# DIFFICULTY SETTINGS
# ================================================================

DIFFICULTY_SETTINGS = {

    "EASY": {
        "count": 3,
        "radius": 70,
        "hit_tolerance": 35,
        "points": 50
    },

    "MEDIUM": {
        "count": 5,
        "radius": 50,
        "hit_tolerance": 20,
        "points": 75
    },

    "HARD": {
        "count": 7,
        "radius": 35,
        "hit_tolerance": 8,
        "points": 100
    }

}


# ================================================================
# GAME STATES
# ================================================================

STATE_MENU = "MENU"

STATE_START_SCREEN = "START_SCREEN"

STATE_PLAYING = "PLAYING"

STATE_COMPLETE = "SESSION_COMPLETE"


# ================================================================
# GENERATE OBJECTS
# ================================================================

def generate_objects(
    difficulty,
    width,
    height
):

    cfg = DIFFICULTY_SETTINGS[difficulty]

    count = cfg["count"]

    radius = cfg["radius"]

    hit_tolerance = cfg["hit_tolerance"]


    # Select unique colors

    color_names = random.sample(
        list(COLOR_PALETTE.keys()),
        count
    )


    left_margin = radius + 60

    right_margin = (
        width -
        radius -
        60
    )

    usable_width = max(
        1,
        right_margin -
        left_margin
    )

    mid_y = height // 2


    objects = []


    for i in range(count):

        if count == 1:

            x = width // 2

        else:

            x = int(
                left_margin +
                i *
                (
                    usable_width /
                    (count - 1)
                )
            )


        y_offset = (
            60
            if i % 2 == 0
            else -60
        )

        y = mid_y + y_offset


        objects.append({

            "pos": (x, y),

            "color_name":
                color_names[i],

            "color_bgr":
                COLOR_PALETTE[
                    color_names[i]
                ]

        })


    target_index = random.randrange(
        count
    )


    return (
        objects,
        target_index,
        radius,
        hit_tolerance
    )


# ================================================================
# FIND TOUCHED OBJECT
# ================================================================

def find_touched_object(
    wrist_pos,
    objects,
    radius,
    hit_tolerance
):

    hit_radius = (
        radius +
        hit_tolerance
    )


    for idx, obj in enumerate(objects):

        ox, oy = obj["pos"]


        dist = math.hypot(

            wrist_pos[0] - ox,

            wrist_pos[1] - oy

        )


        if dist <= hit_radius:

            return idx


    return None


# ================================================================
# FULLSCREEN LETTERBOX
# ================================================================

def resize_with_letterbox(
    frame,
    target_w,
    target_h
):

    h, w = frame.shape[:2]


    scale = min(

        target_w / w,

        target_h / h

    )


    new_w = max(
        1,
        int(w * scale)
    )

    new_h = max(
        1,
        int(h * scale)
    )


    resized = cv2.resize(

        frame,

        (new_w, new_h),

        interpolation=cv2.INTER_LINEAR

    )


    canvas = np.zeros(

        (
            target_h,
            target_w,
            3
        ),

        dtype=np.uint8

    )


    x_offset = (
        target_w -
        new_w
    ) // 2

    y_offset = (
        target_h -
        new_h
    ) // 2


    canvas[
        y_offset:
        y_offset + new_h,

        x_offset:
        x_offset + new_w
    ] = resized


    return canvas


# ================================================================
# DRAW CENTERED TEXT
# ================================================================

def draw_text_center(
    frame,
    text,
    y,
    scale=1.0,
    color=COLOR_BG_TEXT,
    thickness=2
):

    (
        text_w,
        _
    ), _ = cv2.getTextSize(

        text,

        cv2.FONT_HERSHEY_SIMPLEX,

        scale,

        thickness

    )


    x = (
        frame.shape[1] -
        text_w
    ) // 2


    cv2.putText(

        frame,

        text,

        (x, y),

        cv2.FONT_HERSHEY_SIMPLEX,

        scale,

        color,

        thickness,

        cv2.LINE_AA

    )


# ================================================================
# MENU SCREEN
# ================================================================

def draw_menu_screen(frame):

    overlay = frame.copy()


    cv2.rectangle(

        overlay,

        (0, 0),

        (
            frame.shape[1],
            frame.shape[0]
        ),

        (30, 30, 30),

        -1

    )


    cv2.addWeighted(

        overlay,

        0.85,

        frame,

        0.15,

        0,

        frame

    )


    draw_text_center(

        frame,

        "OBJECT SORTING",

        100,

        1.4,

        (0, 255, 255),

        3

    )


    draw_text_center(

        frame,

        "MOVENTRA - Gamified Rehabilitation",

        140,

        0.7,

        (200, 200, 200),

        1

    )


    draw_text_center(

        frame,

        "Select Difficulty",

        220,

        1.0,

        COLOR_BG_TEXT,

        2

    )


    draw_text_center(

        frame,

        "1 - Easy",

        280,

        0.9,

        (0, 255, 0),

        2

    )


    draw_text_center(

        frame,

        "2 - Medium",

        320,

        0.9,

        (0, 165, 255),

        2

    )


    draw_text_center(

        frame,

        "3 - Hard",

        360,

        0.9,

        (0, 0, 255),

        2

    )


    draw_text_center(

        frame,

        "Press Q to Quit",

        frame.shape[0] - 40,

        0.7,

        (180, 180, 180),

        1

    )


# ================================================================
# START SCREEN
# ================================================================

def draw_start_screen(
    frame,
    difficulty,
    patient_id,
    patient_name,
    session_number
):

    draw_text_center(

        frame,

        "OBJECT SORTING",

        60,

        1.1,

        (0, 255, 255),

        2

    )


    draw_text_center(

        frame,

        f"Patient: {patient_id} - {patient_name}",

        100,

        0.7,

        COLOR_BG_TEXT,

        2

    )


    draw_text_center(

        frame,

        f"Session: {session_number}",

        135,

        0.7,

        COLOR_BG_TEXT,

        2

    )


    draw_text_center(

        frame,

        f"Difficulty: {difficulty}",

        175,

        0.8,

        COLOR_BG_TEXT,

        2

    )


    draw_text_center(

        frame,

        "Reach the object that matches the target color",

        230,

        0.7,

        COLOR_BG_TEXT,

        1

    )


    draw_text_center(

        frame,

        "Use your LEFT hand",

        frame.shape[0] - 100,

        0.8,

        (0, 255, 0),

        2

    )


    draw_text_center(

        frame,

        "Press SPACE to Start",

        frame.shape[0] - 60,

        0.9,

        (0, 255, 0),

        2

    )


    draw_text_center(

        frame,

        "Press Q to Quit",

        frame.shape[0] - 25,

        0.6,

        (180, 180, 180),

        1

    )


# ================================================================
# DRAW OBJECTS
# ================================================================

def draw_objects(

    frame,
    objects,
    target_index,
    radius,
    feedback_active,
    feedback_type,
    touched_index

):

    for idx, obj in enumerate(objects):

        pos = obj["pos"]

        color = obj["color_bgr"]


        cv2.circle(

            frame,

            pos,

            radius,

            color,

            -1

        )


        cv2.circle(

            frame,

            pos,

            radius,

            (255, 255, 255),

            2

        )


        if (

            feedback_active
            and
            idx == touched_index

        ):

            if feedback_type == "correct":

                ring_color = COLOR_CORRECT

            else:

                ring_color = COLOR_WRONG


            cv2.circle(

                frame,

                pos,

                radius + 12,

                ring_color,

                4

            )


# ================================================================
# TARGET INSTRUCTION
# ================================================================

def draw_target_instruction(
    frame,
    target_color_name
):

    h, w = frame.shape[:2]


    text = (
        f"Find the "
        f"{target_color_name} Object"
    )


    color = COLOR_PALETTE[
        target_color_name
    ]


    (
        text_w,
        text_h
    ), _ = cv2.getTextSize(

        text,

        cv2.FONT_HERSHEY_SIMPLEX,

        0.9,

        2

    )


    box_y = h - 80


    cv2.rectangle(

        frame,

        (
            w // 2 -
            text_w // 2 -
            20,

            box_y -
            text_h -
            15
        ),

        (
            w // 2 +
            text_w // 2 +
            20,

            box_y + 15
        ),

        (20, 20, 20),

        -1

    )


    cv2.putText(

        frame,

        text,

        (
            w // 2 -
            text_w // 2,

            box_y
        ),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.9,

        color,

        2,

        cv2.LINE_AA

    )


# ================================================================
# FEEDBACK
# ================================================================

def draw_feedback_banner(
    frame,
    feedback_type
):

    if feedback_type == "correct":

        draw_text_center(

            frame,

            "CORRECT!",

            100,

            1.3,

            COLOR_CORRECT,

            3

        )

    else:

        draw_text_center(

            frame,

            "WRONG OBJECT - TRY AGAIN",

            100,

            1.0,

            COLOR_WRONG,

            3

        )


# ================================================================
# HUD
# ================================================================

def draw_hud(

    frame,
    score,
    accuracy,
    time_left,
    targets_completed

):

    h, w = frame.shape[:2]


    overlay = frame.copy()


    cv2.rectangle(

        overlay,

        (0, 0),

        (w, 70),

        (20, 20, 20),

        -1

    )


    cv2.addWeighted(

        overlay,

        0.6,

        frame,

        0.4,

        0,

        frame

    )


    cv2.putText(

        frame,

        f"Score: {score}",

        (20, 30),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.7,

        (0, 255, 0),

        2

    )


    cv2.putText(

        frame,

        f"Accuracy: {accuracy:.1f}%",

        (20, 58),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.7,

        (0, 200, 255),

        2

    )


    targets_text = (

        f"Targets Completed: "
        f"{targets_completed}"
    )


    (
        tw,
        _
    ), _ = cv2.getTextSize(

        targets_text,

        cv2.FONT_HERSHEY_SIMPLEX,

        0.7,

        2

    )


    cv2.putText(

        frame,

        targets_text,

        (
            w // 2 -
            tw // 2,

            40
        ),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.7,

        COLOR_BG_TEXT,

        2

    )


    timer_text = (
        f"Time: "
        f"{int(time_left)}s"
    )


    (
        tw3,
        _
    ), _ = cv2.getTextSize(

        timer_text,

        cv2.FONT_HERSHEY_SIMPLEX,

        0.7,

        2

    )


    cv2.putText(

        frame,

        timer_text,

        (
            w -
            tw3 -
            20,

            30
        ),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.7,

        (255, 255, 0),

        2

    )


# ================================================================
# WRIST WARNING
# ================================================================

def draw_wrist_not_detected_warning(
    frame
):

    draw_text_center(

        frame,

        "LEFT WRIST NOT DETECTED",

        110,

        0.8,

        COLOR_WARNING,

        2

    )


    draw_text_center(

        frame,

        "Please make sure your left arm is visible",

        140,

        0.6,

        COLOR_WARNING,

        1

    )


# ================================================================
# SAVE RESULTS TO CSV
# ================================================================

def save_object_sorting_result(
    patient_id,
    patient_name,
    session_number,
    difficulty,
    score,
    accuracy,
    correct_count,
    wrong_count,
    targets_completed,
    time_taken,
    distance_travelled
):

    try:

        # --------------------------------------------------------
        # Read existing CSV
        # --------------------------------------------------------

        if os.path.exists(CSV_PATH):

            df = pd.read_csv(
                CSV_PATH
            )

        else:

            df = pd.DataFrame()


        # --------------------------------------------------------
        # Required CSV columns
        # --------------------------------------------------------

        required_columns = [

            "patient_id",

            "patient_name",

            "session_number",

            # Balloon Reach
            "balloon_score",
            "balloon_attempts",
            "balloon_successful_targets",
            "balloon_accuracy",
            "balloon_time_sec",

            # Follow the Path
            "path_score",
            "path_accuracy",
            "path_checkpoints_completed",
            "path_time_sec",
            "path_distance_moved",

            # Object Sorting
            "sorting_score",
            "sorting_accuracy",
            "sorting_correct",
            "sorting_wrong",
            "sorting_targets_completed",
            "sorting_time_sec",
            "sorting_distance_moved",

            "is_sample_data"

        ]


        # --------------------------------------------------------
        # Add missing columns
        # --------------------------------------------------------

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


        # --------------------------------------------------------
        # Fix numeric column types
        #
        # This prevents the pandas dtype error that happened
        # previously in Follow the Path.
        # --------------------------------------------------------

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


        # --------------------------------------------------------
        # Text columns
        # --------------------------------------------------------

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


        # --------------------------------------------------------
        # Find patient + session
        # --------------------------------------------------------

        mask = (

            (df["patient_id"] == patient_id)

            &

            (
                df["session_number"]
                == session_number
            )

        )


        if mask.any():

            row_index = df.index[
                mask
            ][0]

        else:

            # Create new row

            new_row = {

                column: 0

                for column
                in required_columns

            }


            new_row[
                "patient_id"
            ] = patient_id


            new_row[
                "patient_name"
            ] = patient_name


            new_row[
                "session_number"
            ] = session_number


            new_row[
                "is_sample_data"
            ] = "NO"


            df = pd.concat(

                [

                    df,

                    pd.DataFrame(
                        [new_row]
                    )

                ],

                ignore_index=True

            )


            row_index = df.index[-1]


        # --------------------------------------------------------
        # Update patient details
        # --------------------------------------------------------

        df.at[

            row_index,

            "patient_name"

        ] = patient_name


        df.at[

            row_index,

            "is_sample_data"

        ] = "NO"


        # --------------------------------------------------------
        # Save Object Sorting values
        # --------------------------------------------------------

        df.at[

            row_index,

            "sorting_score"

        ] = int(score)


        df.at[

            row_index,

            "sorting_accuracy"

        ] = float(accuracy)


        df.at[

            row_index,

            "sorting_correct"

        ] = int(correct_count)


        df.at[

            row_index,

            "sorting_wrong"

        ] = int(wrong_count)


        df.at[

            row_index,

            "sorting_targets_completed"

        ] = int(targets_completed)


        df.at[

            row_index,

            "sorting_time_sec"

        ] = float(time_taken)


        df.at[

            row_index,

            "sorting_distance_moved"

        ] = float(distance_travelled)


        # --------------------------------------------------------
        # Save CSV
        # --------------------------------------------------------

        os.makedirs(

            os.path.dirname(
                CSV_PATH
            ),

            exist_ok=True

        )


        df.to_csv(

            CSV_PATH,

            index=False

        )


        # --------------------------------------------------------
        # Terminal output
        # --------------------------------------------------------

        print("\n========================================")

        print(
            "OBJECT SORTING RESULTS SAVED"
        )

        print("========================================")

        print(
            f"Patient : "
            f"{patient_id} - "
            f"{patient_name}"
        )

        print(
            f"Session : "
            f"{session_number}"
        )

        print(
            f"Difficulty: "
            f"{difficulty}"
        )

        print(
            f"Score   : "
            f"{score}"
        )

        print(
            f"Correct : "
            f"{correct_count}"
        )

        print(
            f"Wrong   : "
            f"{wrong_count}"
        )

        print(
            f"Targets : "
            f"{targets_completed}"
        )

        print(
            f"Accuracy: "
            f"{accuracy:.1f}%"
        )

        print(
            f"Time    : "
            f"{time_taken:.1f} sec"
        )

        print(
            f"Distance: "
            f"{distance_travelled:.0f} px"
        )

        print(
            f"CSV     : "
            f"{CSV_PATH}"
        )

        print(
            "========================================"
        )


        return True


    except Exception as e:

        print("\n========================================")

        print(
            "ERROR: Could not save "
            "Object Sorting results."
        )

        print("Reason:", e)

        print(
            "========================================"
        )

        return False


# ================================================================
# SESSION COMPLETE SCREEN
# ================================================================

def draw_session_complete(
    frame,
    results,
    save_success
):

    overlay = frame.copy()


    cv2.rectangle(

        overlay,

        (0, 0),

        (
            frame.shape[1],
            frame.shape[0]
        ),

        (20, 20, 20),

        -1

    )


    cv2.addWeighted(

        overlay,

        0.9,

        frame,

        0.1,

        0,

        frame

    )


    draw_text_center(

        frame,

        "SESSION COMPLETE",

        80,

        1.3,

        (0, 255, 255),

        3

    )


    lines = [

        f"Difficulty: {results['difficulty']}",

        f"Score: {results['score']}",

        f"Accuracy: {results['accuracy']:.1f}%",

        f"Correct Selections: {results['correct']}",

        f"Wrong Selections: {results['wrong']}",

        f"Targets Completed: {results['targets_completed']}",

        f"Time Taken: {results['time_taken']:.1f} sec",

        f"Distance Travelled: {results['distance']:.0f} px"

    ]


    y = 140


    for line in lines:

        draw_text_center(

            frame,

            line,

            y,

            0.75,

            COLOR_BG_TEXT,

            2

        )

        y += 38


    if save_success:

        draw_text_center(

            frame,

            "Results saved to dashboard",

            y + 5,

            0.65,

            (0, 255, 0),

            2

        )

    else:

        draw_text_center(

            frame,

            "CSV save failed - check terminal",

            y + 5,

            0.65,

            (0, 0, 255),

            2

        )


    draw_text_center(

        frame,

        "Press R to PLAY AGAIN   |   Press Q to QUIT",

        frame.shape[0] - 40,

        0.7,

        (0, 255, 0),

        2

    )


# ================================================================
# MAIN
# ================================================================

def main():

    # ------------------------------------------------------------
    # Patient information
    # ------------------------------------------------------------

    print("\n========================================")

    print(
        "MOVENTRA - OBJECT SORTING"
    )

    print("========================================")


    patient_id = input(

        "Enter Patient ID "
        "(example: P003): "

    ).strip()


    patient_name = input(

        "Enter Patient Name: "

    ).strip()


    while True:

        session_input = input(

            "Enter Session Number: "

        ).strip()


        try:

            session_number = int(
                session_input
            )

            break

        except ValueError:

            print(
                "Please enter a valid "
                "session number."
            )


    print("----------------------------------------")

    print(
        f"Patient ID   : "
        f"{patient_id}"
    )

    print(
        f"Patient Name : "
        f"{patient_name}"
    )

    print(
        f"Session      : "
        f"{session_number}"
    )

    print("----------------------------------------")


    # ------------------------------------------------------------
    # Check model
    # ------------------------------------------------------------

    if not os.path.exists(
        MODEL_PATH
    ):

        print(
            f"ERROR: Pose model not found:"
        )

        print(
            MODEL_PATH
        )

        sys.exit(1)


    # ------------------------------------------------------------
    # MediaPipe
    # ------------------------------------------------------------

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
            vision.RunningMode.VIDEO,

            num_poses=1,

            min_pose_detection_confidence=
            0.5,

            min_pose_presence_confidence=
            0.5,

            min_tracking_confidence=
            0.5

        )
    )


    landmarker = (
        vision.PoseLandmarker
        .create_from_options(
            options
        )
    )


    # ------------------------------------------------------------
    # Camera
    # ------------------------------------------------------------

    cap = cv2.VideoCapture(
        CAM_INDEX
    )


    if not cap.isOpened():

        print(
            "ERROR: Could not open webcam."
        )

        landmarker.close()

        sys.exit(1)


    cap.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        1280
    )

    cap.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        720
    )


    # ------------------------------------------------------------
    # Screen resolution
    # ------------------------------------------------------------

    root = tk.Tk()

    root.withdraw()

    screen_w = (
        root.winfo_screenwidth()
    )

    screen_h = (
        root.winfo_screenheight()
    )

    root.destroy()


    # ------------------------------------------------------------
    # Fullscreen window
    # ------------------------------------------------------------

    cv2.namedWindow(

        WINDOW_NAME,

        cv2.WINDOW_NORMAL

    )


    cv2.setWindowProperty(

        WINDOW_NAME,

        cv2.WND_PROP_FULLSCREEN,

        cv2.WINDOW_FULLSCREEN

    )


    # ------------------------------------------------------------
    # Initial game state
    # ------------------------------------------------------------

    state = STATE_MENU

    difficulty = None


    objects = []

    target_index = 0

    obj_radius = 50

    hit_tolerance = 20


    score = 0

    correct_count = 0

    wrong_count = 0

    targets_completed = 0

    distance_travelled = 0.0


    prev_wrist_pos = None


    feedback_active = False

    feedback_type = None

    feedback_until = 0.0

    feedback_touched_index = None


    last_touched_index = None


    session_start_time = 0.0

    frame_timestamp_ms = 0


    results = {}

    save_success = False

    results_saved = False


    print(
        "MOVENTRA - Object Sorting started."
    )

    print(
        "Press Q at any time to quit."
    )


    # ============================================================
    # MAIN LOOP
    # ============================================================

    while True:

        ret, raw_frame = cap.read()


        if not ret:

            print(
                "ERROR: Failed to read "
                "frame from webcam."
            )

            break


        h, w = raw_frame.shape[:2]


        # --------------------------------------------------------
        # MediaPipe detection
        #
        # IMPORTANT:
        # Detection happens BEFORE flipping the frame so landmark
        # 15 remains the patient's anatomical LEFT wrist.
        # --------------------------------------------------------

        wrist_pos = None

        wrist_detected = False


        if state in (

            STATE_START_SCREEN,

            STATE_PLAYING

        ):

            rgb_frame = cv2.cvtColor(

                raw_frame,

                cv2.COLOR_BGR2RGB

            )


            mp_image = mp.Image(

                image_format=
                mp.ImageFormat.SRGB,

                data=rgb_frame

            )


            frame_timestamp_ms += 33


            pose_result = (
                landmarker
                .detect_for_video(

                    mp_image,

                    frame_timestamp_ms

                )
            )


            if pose_result.pose_landmarks:

                landmarks = (
                    pose_result
                    .pose_landmarks[0]
                )


                wrist_lm = (
                    landmarks[
                        LEFT_WRIST_INDEX
                    ]
                )


                if (

                    wrist_lm.visibility
                    is None

                    or

                    wrist_lm.visibility
                    > 0.3

                ):

                    raw_x = int(
                        wrist_lm.x * w
                    )

                    raw_y = int(
                        wrist_lm.y * h
                    )


                    # Mirror X coordinate
                    # for display.

                    wrist_x = (
                        w - raw_x
                    )

                    wrist_y = raw_y


                    wrist_pos = (

                        wrist_x,

                        wrist_y

                    )


                    wrist_detected = True


        # --------------------------------------------------------
        # Mirror display
        # --------------------------------------------------------

        frame = cv2.flip(
            raw_frame,
            1
        )


        # ========================================================
        # MENU
        # ========================================================

        if state == STATE_MENU:

            draw_menu_screen(
                frame
            )


        # ========================================================
        # START SCREEN
        # ========================================================

        elif state == STATE_START_SCREEN:

            draw_start_screen(

                frame,

                difficulty,

                patient_id,

                patient_name,

                session_number

            )


            if wrist_detected:

                cv2.circle(

                    frame,

                    wrist_pos,

                    12,

                    COLOR_WRIST_NEUTRAL,

                    -1

                )

            else:

                draw_wrist_not_detected_warning(
                    frame
                )


        # ========================================================
        # PLAYING
        # ========================================================

        elif state == STATE_PLAYING:

            elapsed = (

                time.time()

                -

                session_start_time

            )


            time_left = max(

                0,

                SESSION_SECONDS -
                elapsed

            )


            # ----------------------------------------------------
            # Feedback timer
            # ----------------------------------------------------

            if (

                feedback_active

                and

                time.time()
                >= feedback_until

            ):

                if feedback_type == "correct":

                    objects, target_index, obj_radius, hit_tolerance = generate_objects(

                        difficulty,

                        w,

                        h

                    )


                    last_touched_index = None


                feedback_active = False

                feedback_type = None

                feedback_touched_index = None


            # ----------------------------------------------------
            # Draw objects
            # ----------------------------------------------------

            draw_objects(

                frame,

                objects,

                target_index,

                obj_radius,

                feedback_active,

                feedback_type,

                feedback_touched_index

            )


            draw_target_instruction(

                frame,

                objects[
                    target_index
                ]["color_name"]

            )


            # ----------------------------------------------------
            # Wrist tracking
            # ----------------------------------------------------

            touched_index = None


            if wrist_detected:

                # Distance travelled

                if prev_wrist_pos is not None:

                    step_distance = math.hypot(

                        wrist_pos[0]
                        -
                        prev_wrist_pos[0],

                        wrist_pos[1]
                        -
                        prev_wrist_pos[1]

                    )


                    distance_travelled += (
                        step_distance
                    )


                prev_wrist_pos = wrist_pos


                # ------------------------------------------------
                # Check object touch
                # ------------------------------------------------

                if not feedback_active:

                    touched_index = (
                        find_touched_object(

                            wrist_pos,

                            objects,

                            obj_radius,

                            hit_tolerance

                        )
                    )


                    # Debounce:
                    # only count when wrist ENTERS
                    # an object.

                    if (

                        touched_index
                        is not None

                        and

                        touched_index
                        != last_touched_index

                    ):

                        # Correct object

                        if (
                            touched_index
                            == target_index
                        ):

                            correct_count += 1

                            targets_completed += 1

                            score += (
                                DIFFICULTY_SETTINGS[
                                    difficulty
                                ]["points"]
                            )

                            feedback_type = (
                                "correct"
                            )


                        # Wrong object

                        else:

                            wrong_count += 1

                            feedback_type = (
                                "wrong"
                            )


                        feedback_active = True

                        feedback_until = (

                            time.time()

                            +

                            FEEDBACK_SECONDS

                        )


                        feedback_touched_index = (
                            touched_index
                        )


                    last_touched_index = (
                        touched_index
                    )


                # ------------------------------------------------
                # Wrist marker
                # ------------------------------------------------

                if feedback_active:

                    if feedback_type == "correct":

                        marker_color = (
                            COLOR_CORRECT
                        )

                    else:

                        marker_color = (
                            COLOR_WRONG
                        )

                else:

                    marker_color = (
                        COLOR_WRIST_NEUTRAL
                    )


                cv2.circle(

                    frame,

                    wrist_pos,

                    15,

                    marker_color,

                    -1

                )


                cv2.circle(

                    frame,

                    wrist_pos,

                    15,

                    (255, 255, 255),

                    2

                )


            else:

                draw_wrist_not_detected_warning(
                    frame
                )

                prev_wrist_pos = None

                last_touched_index = None


            # ----------------------------------------------------
            # Feedback banner
            # ----------------------------------------------------

            if feedback_active:

                draw_feedback_banner(

                    frame,

                    feedback_type

                )


            # ----------------------------------------------------
            # Accuracy
            # ----------------------------------------------------

            total_attempts = (

                correct_count

                +

                wrong_count

            )


            if total_attempts > 0:

                accuracy = (

                    correct_count
                    /
                    total_attempts
                    *
                    100
                )

            else:

                accuracy = 0.0


            # ----------------------------------------------------
            # HUD
            # ----------------------------------------------------

            draw_hud(

                frame,

                score,

                accuracy,

                time_left,

                targets_completed

            )


            # ----------------------------------------------------
            # Session end
            # ----------------------------------------------------

            if time_left <= 0:

                results = {

                    "difficulty":
                        difficulty,

                    "score":
                        score,

                    "accuracy":
                        accuracy,

                    "correct":
                        correct_count,

                    "wrong":
                        wrong_count,

                    "targets_completed":
                        targets_completed,

                    "time_taken":
                        elapsed,

                    "distance":
                        distance_travelled

                }


                state = STATE_COMPLETE


        # ========================================================
        # COMPLETE SCREEN
        # ========================================================

        elif state == STATE_COMPLETE:

            # Save only once

            if not results_saved:

                save_success = (
                    save_object_sorting_result(

                        patient_id,

                        patient_name,

                        session_number,

                        results["difficulty"],

                        results["score"],

                        results["accuracy"],

                        results["correct"],

                        results["wrong"],

                        results[
                            "targets_completed"
                        ],

                        results["time_taken"],

                        results["distance"]

                    )
                )


                results_saved = True


            draw_session_complete(

                frame,

                results,

                save_success

            )


        # ========================================================
        # DISPLAY
        # ========================================================

        display_frame = (
            resize_with_letterbox(

                frame,

                screen_w,

                screen_h

            )
        )


        cv2.imshow(

            WINDOW_NAME,

            display_frame

        )


        # ========================================================
        # KEYBOARD
        # ========================================================

        key = (
            cv2.waitKey(1)
            &
            0xFF
        )


        # Quit

        if key == ord("q"):

            break


        # ========================================================
        # MENU CONTROLS
        # ========================================================

        if state == STATE_MENU:

            if key in (

                ord("1"),

                ord("2"),

                ord("3")

            ):

                difficulty = {

                    ord("1"): "EASY",

                    ord("2"): "MEDIUM",

                    ord("3"): "HARD"

                }[key]


                state = (
                    STATE_START_SCREEN
                )


        # ========================================================
        # START SCREEN CONTROLS
        # ========================================================

        elif state == STATE_START_SCREEN:

            if key == ord(" "):

                (

                    objects,

                    target_index,

                    obj_radius,

                    hit_tolerance

                ) = generate_objects(

                    difficulty,

                    w,

                    h

                )


                score = 0

                correct_count = 0

                wrong_count = 0

                targets_completed = 0

                distance_travelled = 0.0

                prev_wrist_pos = None

                last_touched_index = None

                feedback_active = False

                feedback_type = None

                feedback_touched_index = None

                results_saved = False

                save_success = False


                session_start_time = (
                    time.time()
                )


                state = STATE_PLAYING


        # ========================================================
        # COMPLETE SCREEN CONTROLS
        # ========================================================

        elif state == STATE_COMPLETE:

            if key == ord("r"):

                state = STATE_MENU

                difficulty = None

                results_saved = False

                save_success = False


    # ============================================================
    # CLEANUP
    # ============================================================

    cap.release()

    cv2.destroyAllWindows()

    landmarker.close()


    print(
        "MOVENTRA - Object Sorting closed."
    )


# ================================================================
# RUN
# ================================================================

if __name__ == "__main__":

    main()