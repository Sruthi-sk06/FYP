"""
=====================================================================
MOVENTRA - AR/VR Rehabilitation System for Stroke Patients
Module 2: Gamified Rehabilitation
Game 2: FOLLOW THE PATH

Camera-based, non-immersive upper-limb rehabilitation game.

Features:
    - MediaPipe Pose Landmarker
    - LEFT / RIGHT wrist tracking
    - Easy / Medium / Hard difficulty
    - Path-following accuracy
    - Checkpoint completion
    - Distance travelled
    - Average path deviation
    - Session timing
    - Patient/session identification
    - Age and training-arm information
    - Automatic Module 3 CSV integration

Controls:
    1 / 2 / 3  -> Easy / Medium / Hard
    SPACE      -> Start
    R          -> Restart after completion
    Q          -> Quit
=====================================================================
"""

# ================================================================
# IMPORTS
# ================================================================

import cv2
import time
import math
import os
import sys
import numpy as np
import tkinter as tk
import pandas as pd

from datetime import date

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

WINDOW_NAME = "MOVENTRA - Follow the Path"

# MediaPipe anatomical wrist landmarks
LEFT_WRIST_INDEX = 15
RIGHT_WRIST_INDEX = 16


# ================================================================
# COLOURS - BGR FORMAT
# ================================================================

COLOR_BG_TEXT = (255, 255, 255)

COLOR_PATH = (200, 200, 0)

COLOR_PATH_DONE = (0, 200, 0)

COLOR_CHECKPOINT = (0, 165, 255)

COLOR_CHECKPOINT_DONE = (0, 255, 0)

COLOR_WRIST_ON = (0, 255, 0)

COLOR_WRIST_OFF = (0, 0, 255)

COLOR_WARNING = (0, 0, 255)


# ================================================================
# CSV STRUCTURE
# ================================================================

CSV_COLUMNS = [

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


# ================================================================
# PATIENT / SESSION INFORMATION
# ================================================================

def get_patient_session_info():

    print("\n========================================")
    print("       MOVENTRA PATIENT SESSION")
    print("========================================")

    # ------------------------------------------------------------
    # PATIENT ID
    # ------------------------------------------------------------

    patient_id = input(
        "Enter Patient ID: "
    ).strip()

    while not patient_id:

        print("Patient ID cannot be empty.")

        patient_id = input(
            "Enter Patient ID: "
        ).strip()

    # ------------------------------------------------------------
    # PATIENT NAME
    # ------------------------------------------------------------

    patient_name = input(
        "Enter Patient Name: "
    ).strip()

    while not patient_name:

        print("Patient Name cannot be empty.")

        patient_name = input(
            "Enter Patient Name: "
        ).strip()

    # ------------------------------------------------------------
    # AGE
    # ------------------------------------------------------------

    while True:

        age_input = input(
            "Enter Patient Age: "
        ).strip()

        try:

            age = int(age_input)

            if age < 1 or age > 120:

                print(
                    "Please enter a valid age between 1 and 120."
                )

                continue

            break

        except ValueError:

            print(
                "Please enter a valid age."
            )

    # ------------------------------------------------------------
    # TRAINING ARM
    # ------------------------------------------------------------

    while True:

        training_arm = input(
            "Enter Training Arm (LEFT/RIGHT): "
        ).strip().upper()

        if training_arm in ("LEFT", "RIGHT"):

            break

        print(
            "Please enter LEFT or RIGHT."
        )

    # ------------------------------------------------------------
    # SESSION NUMBER
    # ------------------------------------------------------------

    while True:

        session_input = input(
            "Enter Session Number: "
        ).strip()

        try:

            session_number = int(
                session_input
            )

            if session_number < 1:

                print(
                    "Session number must be 1 or greater."
                )

                continue

            break

        except ValueError:

            print(
                "Please enter a valid number."
            )

    # ------------------------------------------------------------
    # DISPLAY INFORMATION
    # ------------------------------------------------------------

    print("----------------------------------------")

    print(
        f"Patient ID   : {patient_id}"
    )

    print(
        f"Patient Name : {patient_name}"
    )

    print(
        f"Age          : {age}"
    )

    print(
        f"Training Arm : {training_arm}"
    )

    print(
        f"Session      : {session_number}"
    )

    print("----------------------------------------\n")

    return (
        patient_id,
        patient_name,
        age,
        training_arm,
        session_number
    )


# ================================================================
# GET PATIENT INFORMATION
# ================================================================

(
    patient_id,
    patient_name,
    patient_age,
    training_arm,
    session_number
) = get_patient_session_info()


# ================================================================
# SELECT WRIST LANDMARK
# ================================================================

if training_arm == "LEFT":

    SELECTED_WRIST_INDEX = LEFT_WRIST_INDEX

else:

    SELECTED_WRIST_INDEX = RIGHT_WRIST_INDEX


# ================================================================
# GAME STATES
# ================================================================

STATE_MENU = "MENU"

STATE_START_SCREEN = "START_SCREEN"

STATE_PLAYING = "PLAYING"

STATE_COMPLETE = "SESSION_COMPLETE"


# ================================================================
# PATH GENERATION
# ================================================================

def generate_path(
    difficulty,
    width,
    height
):

    settings = {

        "EASY": {

            "amp": 60,

            "freq": 1,

            "checkpoints": 6,

            "tolerance": 60
        },

        "MEDIUM": {

            "amp": 90,

            "freq": 2,

            "checkpoints": 8,

            "tolerance": 45
        },

        "HARD": {

            "amp": 120,

            "freq": 3,

            "checkpoints": 10,

            "tolerance": 30
        }
    }

    cfg = settings[difficulty]

    left_margin = 80

    right_margin = width - 80

    mid_y = height // 2

    path_points = []

    num_points = 300

    for i in range(num_points):

        t = (
            i /
            (num_points - 1)
        )

        x = int(
            left_margin
            +
            t *
            (
                right_margin
                -
                left_margin
            )
        )

        y = int(
            mid_y
            +
            cfg["amp"]
            *
            math.sin(
                2
                *
                math.pi
                *
                cfg["freq"]
                *
                t
            )
        )

        path_points.append(
            (x, y)
        )

    checkpoints = []

    num_cp = cfg["checkpoints"]

    for i in range(num_cp):

        idx = int(
            i
            *
            (num_points - 1)
            /
            (num_cp - 1)
        )

        checkpoints.append(
            path_points[idx]
        )

    return (
        path_points,
        checkpoints,
        cfg["tolerance"]
    )


# ================================================================
# DISTANCE FROM POINT TO PATH
# ================================================================

def distance_point_to_polyline(
    point,
    polyline
):

    px, py = point

    min_dist = float("inf")

    for i in range(
        len(polyline) - 1
    ):

        x1, y1 = polyline[i]

        x2, y2 = polyline[i + 1]

        seg_len_sq = (
            (x2 - x1) ** 2
            +
            (y2 - y1) ** 2
        )

        if seg_len_sq == 0:

            dist = math.hypot(
                px - x1,
                py - y1
            )

        else:

            t = (

                (px - x1)
                *
                (x2 - x1)

                +

                (py - y1)
                *
                (y2 - y1)

            ) / seg_len_sq

            t = max(
                0,
                min(1, t)
            )

            proj_x = (
                x1
                +
                t *
                (x2 - x1)
            )

            proj_y = (
                y1
                +
                t *
                (y2 - y1)
            )

            dist = math.hypot(
                px - proj_x,
                py - proj_y
            )

        if dist < min_dist:

            min_dist = dist

    return min_dist


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
        target_w - new_w
    ) // 2

    y_offset = (
        target_h - new_h
    ) // 2

    canvas[
        y_offset:
        y_offset + new_h,

        x_offset:
        x_offset + new_w
    ] = resized

    return canvas


# ================================================================
# TEXT HELPER
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
        frame.shape[1]
        -
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
        "FOLLOW THE PATH",
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
        f"Training Arm: {training_arm}",
        180,
        0.8,
        (255, 255, 0),
        2
    )

    draw_text_center(
        frame,
        "Select Difficulty",
        240,
        1.0,
        COLOR_BG_TEXT,
        2
    )

    draw_text_center(
        frame,
        "1 - Easy",
        300,
        0.9,
        (0, 255, 0),
        2
    )

    draw_text_center(
        frame,
        "2 - Medium",
        340,
        0.9,
        (0, 165, 255),
        2
    )

    draw_text_center(
        frame,
        "3 - Hard",
        380,
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
    difficulty
):

    draw_text_center(
        frame,
        "FOLLOW THE PATH",
        60,
        1.1,
        (0, 255, 255),
        2
    )

    draw_text_center(
        frame,
        f"Difficulty: {difficulty}",
        100,
        0.8,
        COLOR_BG_TEXT,
        2
    )

    draw_text_center(
        frame,
        f"Use your {training_arm} hand to follow the path",
        140,
        0.7,
        COLOR_BG_TEXT,
        1
    )

    draw_text_center(
        frame,
        f"Patient: {patient_name}",
        180,
        0.65,
        (200, 200, 200),
        1
    )

    draw_text_center(
        frame,
        f"Training Arm: {training_arm}",
        215,
        0.65,
        (255, 255, 0),
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
# DRAW PATH
# ================================================================

def draw_path(
    frame,
    path_points,
    checkpoints,
    current_cp_index
):

    checkpoint_section = (
        len(path_points)
        //
        max(
            1,
            len(checkpoints)
        )
    )

    for i in range(
        len(path_points) - 1
    ):

        if (
            i
            <
            current_cp_index
            *
            checkpoint_section
        ):

            color = COLOR_PATH_DONE

        else:

            color = COLOR_PATH

        cv2.line(
            frame,
            path_points[i],
            path_points[i + 1],
            color,
            4
        )

    for idx, cp in enumerate(
        checkpoints
    ):

        if idx < current_cp_index:

            color = COLOR_CHECKPOINT_DONE

        elif idx == current_cp_index:

            color = COLOR_CHECKPOINT

        else:

            color = (
                120,
                120,
                120
            )

        radius = (
            14
            if idx == current_cp_index
            else 8
        )

        cv2.circle(
            frame,
            cp,
            radius,
            color,
            -1
        )

        cv2.circle(
            frame,
            cp,
            radius,
            (255, 255, 255),
            2
        )


# ================================================================
# HUD
# ================================================================

def draw_hud(
    frame,
    score,
    accuracy,
    time_left,
    checkpoints_reached,
    total_checkpoints,
    wrist_on_path,
    avg_deviation
):

    h, w = frame.shape[:2]

    overlay = frame.copy()

    cv2.rectangle(
        overlay,
        (0, 0),
        (w, 115),
        (20, 20, 20),
        -1
    )

    cv2.addWeighted(
        overlay,
        0.65,
        frame,
        0.35,
        0,
        frame
    )

    # ------------------------------------------------------------
    # TRAINING ARM
    # ------------------------------------------------------------

    cv2.putText(
        frame,
        f"Arm: {training_arm}",
        (20, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 0),
        2
    )

    # ------------------------------------------------------------
    # SCORE
    # ------------------------------------------------------------

    cv2.putText(
        frame,
        f"Score: {score}",
        (20, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    # ------------------------------------------------------------
    # ACCURACY
    # ------------------------------------------------------------

    cv2.putText(
        frame,
        f"Accuracy: {accuracy:.0f}%",
        (20, 85),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 200, 255),
        2
    )

    # ------------------------------------------------------------
    # DEVIATION
    # ------------------------------------------------------------

    cv2.putText(
        frame,
        f"Deviation: {avg_deviation:.1f}px",
        (20, 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 200, 0),
        2
    )

    # ------------------------------------------------------------
    # CHECKPOINTS
    # ------------------------------------------------------------

    cp_text = (
        f"Checkpoints: "
        f"{checkpoints_reached}/"
        f"{total_checkpoints}"
    )

    (
        tw,
        _
    ), _ = cv2.getTextSize(
        cp_text,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        2
    )

    cv2.putText(
        frame,
        cp_text,
        (
            w // 2 - tw // 2,
            35
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        COLOR_BG_TEXT,
        2
    )

    # ------------------------------------------------------------
    # STATUS
    # ------------------------------------------------------------

    status_text = (
        "ON PATH"
        if wrist_on_path
        else
        "OFF PATH - move back to the line"
    )

    status_color = (
        COLOR_WRIST_ON
        if wrist_on_path
        else
        COLOR_WRIST_OFF
    )

    (
        tw2,
        _
    ), _ = cv2.getTextSize(
        status_text,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        2
    )

    cv2.putText(
        frame,
        status_text,
        (
            w // 2 - tw2 // 2,
            65
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        status_color,
        2
    )

    # ------------------------------------------------------------
    # TIMER
    # ------------------------------------------------------------

    timer_text = (
        f"Time: {int(time_left)}s"
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
            w - tw3 - 20,
            35
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
        f"{training_arm} WRIST NOT DETECTED",
        150,
        0.8,
        COLOR_WARNING,
        2
    )

    draw_text_center(
        frame,
        f"Please make sure your {training_arm.lower()} arm is visible",
        180,
        0.6,
        COLOR_WARNING,
        1
    )


# ================================================================
# SESSION COMPLETE SCREEN
# ================================================================

def draw_session_complete(
    frame,
    results
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

        f"Patient: {patient_name}",

        f"Training Arm: {training_arm}",

        f"Difficulty: "
        f"{results['difficulty']}",

        f"Score: "
        f"{results['score']}",

        f"Path-Following Accuracy: "
        f"{results['accuracy']:.1f}%",

        f"Average Path Deviation: "
        f"{results['average_deviation']:.1f} px",

        f"Time Taken: "
        f"{results['time_taken']:.1f} sec",

        f"Distance Travelled: "
        f"{results['distance']:.0f} px",

        f"Checkpoints Reached: "
        f"{results['checkpoints_reached']}/"
        f"{results['total_checkpoints']}",
    ]

    y = 125

    for line in lines:

        draw_text_center(
            frame,
            line,
            y,
            0.68,
            COLOR_BG_TEXT,
            2
        )

        y += 32

    draw_text_center(
        frame,
        "Results saved to patient progress",
        y + 10,
        0.65,
        (0, 255, 0),
        2
    )

    draw_text_center(
        frame,
        "Press R to Restart   |   Press Q to Quit",
        frame.shape[0] - 40,
        0.7,
        (0, 255, 0),
        2
    )


# ================================================================
# SAVE FOLLOW THE PATH RESULT
# ================================================================

def save_follow_path_result(
    results
):

    try:

        # ========================================================
        # READ EXISTING CSV
        # ========================================================

        if os.path.exists(CSV_PATH):

            df = pd.read_csv(
                CSV_PATH
            )

        else:

            df = pd.DataFrame(
                columns=CSV_COLUMNS
            )

        # ========================================================
        # ADD MISSING COLUMNS
        # ========================================================

        for column in CSV_COLUMNS:

            if column not in df.columns:

                if column == "age":

                    df[column] = 0

                elif column == "training_arm":

                    df[column] = ""

                else:

                    df[column] = 0

        # ========================================================
        # INTEGER COLUMNS
        # ========================================================

        integer_columns = [

            "age",

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

        # ========================================================
        # FLOAT COLUMNS
        # ========================================================

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

        # ========================================================
        # CONVERT INTEGER COLUMNS
        # ========================================================

        for column in integer_columns:

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

            df[column] = (
                df[column]
                .fillna(0)
                .astype(int)
            )

        # ========================================================
        # CONVERT FLOAT COLUMNS
        # ========================================================

        for column in float_columns:

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

            df[column] = (
                df[column]
                .fillna(0.0)
                .astype(float)
            )

        # ========================================================
        # FIND EXISTING PATIENT + SESSION
        # ========================================================

        mask = (

            df["patient_id"]
            .astype(str)
            .str.strip()
            .str.lower()
            .eq(
                str(patient_id)
                .strip()
                .lower()
            )

            &

            df["session_number"]
            .astype(str)
            .eq(
                str(session_number)
            )
        )

        # ========================================================
        # EXISTING SESSION
        # ========================================================

        if mask.any():

            row_index = df.index[
                mask
            ][0]

            df.at[
                row_index,
                "patient_name"
            ] = patient_name

            df.at[
                row_index,
                "age"
            ] = patient_age

            df.at[
                row_index,
                "training_arm"
            ] = training_arm

            df.at[
                row_index,
                "session_date"
            ] = date.today().isoformat()

        # ========================================================
        # NEW SESSION
        # ========================================================

        else:

            new_row = {}

            for column in CSV_COLUMNS:

                if column in integer_columns:

                    new_row[column] = 0

                elif column in float_columns:

                    new_row[column] = 0.0

                else:

                    new_row[column] = ""

            # Patient information
            new_row["patient_id"] = (
                patient_id
            )

            new_row["patient_name"] = (
                patient_name
            )

            new_row["age"] = (
                patient_age
            )

            new_row["training_arm"] = (
                training_arm
            )

            new_row["session_number"] = (
                int(session_number)
            )

            new_row["session_date"] = (
                date.today().isoformat()
            )

            new_row["is_sample_data"] = (
                "NO"
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

            row_index = df.index[-1]

        # ========================================================
        # FOLLOW THE PATH VALUES
        # ========================================================

        path_score = int(
            results["score"]
        )

        path_accuracy = float(
            results["accuracy"]
        )

        checkpoints_completed = int(
            results[
                "checkpoints_reached"
            ]
        )

        path_time = float(
            results[
                "time_taken"
            ]
        )

        path_distance = float(
            results[
                "distance"
            ]
        )

        # ========================================================
        # SAVE FOLLOW THE PATH DATA
        # ========================================================

        df.at[
            row_index,
            "path_score"
        ] = path_score

        df.at[
            row_index,
            "path_accuracy"
        ] = path_accuracy

        df.at[
            row_index,
            "path_checkpoints_completed"
        ] = checkpoints_completed

        df.at[
            row_index,
            "path_time_sec"
        ] = path_time

        df.at[
            row_index,
            "path_distance_moved"
        ] = path_distance

        df.at[
            row_index,
            "is_sample_data"
        ] = "NO"

        # ========================================================
        # FINAL PATIENT INFORMATION
        # ========================================================

        df.at[
            row_index,
            "patient_id"
        ] = patient_id

        df.at[
            row_index,
            "patient_name"
        ] = patient_name

        df.at[
            row_index,
            "age"
        ] = patient_age

        df.at[
            row_index,
            "training_arm"
        ] = training_arm

        df.at[
            row_index,
            "session_number"
        ] = int(session_number)

        # ========================================================
        # FINAL TYPE CONVERSION
        # ========================================================

        for column in integer_columns:

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

            df[column] = (
                df[column]
                .fillna(0)
                .round()
                .astype(int)
            )

        for column in float_columns:

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

            df[column] = (
                df[column]
                .fillna(0.0)
                .astype(float)
            )

        # ========================================================
        # KEEP REQUIRED COLUMN ORDER
        # ========================================================

        df = df[
            CSV_COLUMNS
        ]

        # ========================================================
        # SAVE
        # ========================================================

        df.to_csv(
            CSV_PATH,
            index=False
        )

        # ========================================================
        # SUCCESS MESSAGE
        # ========================================================

        print(
            "\n========================================"
        )

        print(
            "FOLLOW THE PATH RESULTS SAVED"
        )

        print(
            "========================================"
        )

        print(
            f"Patient      : "
            f"{patient_name}"
        )

        print(
            f"Patient ID   : "
            f"{patient_id}"
        )

        print(
            f"Age          : "
            f"{patient_age}"
        )

        print(
            f"Training Arm : "
            f"{training_arm}"
        )

        print(
            f"Session      : "
            f"{session_number}"
        )

        print(
            f"Score        : "
            f"{path_score}"
        )

        print(
            f"Accuracy     : "
            f"{path_accuracy:.1f}%"
        )

        print(
            f"Time         : "
            f"{path_time:.1f} sec"
        )

        print(
            f"Distance     : "
            f"{path_distance:.0f} px"
        )

        print(
            f"CSV          : "
            f"{CSV_PATH}"
        )

        print(
            "========================================\n"
        )

        return True

    except Exception as e:

        print(
            "\nERROR: Could not save "
            "Follow the Path results."
        )

        print(
            f"Reason: {e}"
        )

        return False


# ================================================================
# MAIN PROGRAM
# ================================================================

def main():

    # ============================================================
    # CHECK MODEL
    # ============================================================

    if not os.path.exists(
        MODEL_PATH
    ):

        print(
            "ERROR: Pose model not found."
        )

        print(
            f"Expected location:\n"
            f"{MODEL_PATH}"
        )

        sys.exit(1)

    # ============================================================
    # CREATE MEDIAPIPE LANDMARKER
    # ============================================================

    base_options = (
        python.BaseOptions(
            model_asset_path=MODEL_PATH
        )
    )

    options = (
        vision.PoseLandmarkerOptions(

            base_options=base_options,

            running_mode=(
                vision.RunningMode.VIDEO
            ),

            num_poses=1,

            min_pose_detection_confidence=0.5,

            min_pose_presence_confidence=0.5,

            min_tracking_confidence=0.5
        )
    )

    landmarker = (
        vision.PoseLandmarker
        .create_from_options(
            options
        )
    )

    # ============================================================
    # OPEN WEBCAM
    # ============================================================

    cap = cv2.VideoCapture(
        CAM_INDEX
    )

    if not cap.isOpened():

        print(
            "ERROR: Could not open webcam."
        )

        landmarker.close()

        sys.exit(1)

    # ============================================================
    # GET SCREEN RESOLUTION
    # ============================================================

    root = tk.Tk()

    root.withdraw()

    screen_w = (
        root.winfo_screenwidth()
    )

    screen_h = (
        root.winfo_screenheight()
    )

    root.destroy()

    # ============================================================
    # CREATE FULLSCREEN WINDOW
    # ============================================================

    cv2.namedWindow(
        WINDOW_NAME,
        cv2.WINDOW_NORMAL
    )

    cv2.setWindowProperty(
        WINDOW_NAME,
        cv2.WND_PROP_FULLSCREEN,
        cv2.WINDOW_FULLSCREEN
    )

    # ============================================================
    # GAME VARIABLES
    # ============================================================

    state = STATE_MENU

    difficulty = None

    path_points = []

    checkpoints = []

    tolerance_px = 50

    current_cp_index = 0

    score = 0

    on_path_frames = 0

    total_frames = 0

    distance_travelled = 0.0

    prev_wrist_pos = None

    total_path_deviation = 0.0

    deviation_samples = 0

    session_start_time = 0.0

    frame_timestamp_ms = 0

    results = {}

    results_saved = False

    print(
        "MOVENTRA - Follow the Path started."
    )

    print(
        f"Training arm: {training_arm}"
    )

    print(
        f"Tracking wrist landmark: "
        f"{SELECTED_WRIST_INDEX}"
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
                "ERROR: Failed to read webcam frame."
            )

            break

        h, w = raw_frame.shape[:2]

        # ========================================================
        # WRIST TRACKING
        # ========================================================

        wrist_pos = None

        wrist_detected = False

        if state in (
            STATE_START_SCREEN,
            STATE_PLAYING
        ):

            # IMPORTANT:
            #
            # MediaPipe detection is performed
            # BEFORE the camera image is mirrored.
            #
            # This ensures:
            #
            # LEFT wrist  = landmark 15
            # RIGHT wrist = landmark 16
            #
            # according to anatomical orientation.

            rgb_frame = cv2.cvtColor(
                raw_frame,
                cv2.COLOR_BGR2RGB
            )

            mp_image = mp.Image(
                image_format=(
                    mp.ImageFormat.SRGB
                ),
                data=rgb_frame
            )

            frame_timestamp_ms += 33

            pose_result = (
                landmarker.detect_for_video(
                    mp_image,
                    frame_timestamp_ms
                )
            )

            if pose_result.pose_landmarks:

                landmarks = (
                    pose_result.pose_landmarks[0]
                )

                # ------------------------------------------------
                # SELECT LEFT OR RIGHT WRIST
                # ------------------------------------------------

                wrist_lm = (
                    landmarks[
                        SELECTED_WRIST_INDEX
                    ]
                )

                # ------------------------------------------------
                # CHECK VISIBILITY
                # ------------------------------------------------

                if (
                    wrist_lm.visibility
                    is None
                    or
                    wrist_lm.visibility > 0.3
                ):

                    # Coordinates from ORIGINAL frame
                    raw_x = int(
                        wrist_lm.x * w
                    )

                    raw_y = int(
                        wrist_lm.y * h
                    )

                    # ------------------------------------------------
                    # MIRROR X FOR DISPLAY
                    # ------------------------------------------------

                    wrist_x = (
                        w - raw_x
                    )

                    wrist_y = raw_y

                    wrist_pos = (
                        wrist_x,
                        wrist_y
                    )

                    wrist_detected = True

        # ========================================================
        # FLIP DISPLAY FRAME
        # ========================================================

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
                difficulty
            )

            if wrist_detected:

                cv2.circle(
                    frame,
                    wrist_pos,
                    12,
                    COLOR_WRIST_ON,
                    -1
                )

                cv2.putText(
                    frame,
                    f"{training_arm} HAND DETECTED",
                    (
                        20,
                        frame.shape[0] - 100
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    COLOR_WRIST_ON,
                    2
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
                SESSION_SECONDS
                -
                elapsed
            )

            # ----------------------------------------------------
            # DRAW PATH
            # ----------------------------------------------------

            draw_path(
                frame,
                path_points,
                checkpoints,
                current_cp_index
            )

            wrist_on_path = False

            # ----------------------------------------------------
            # WRIST DETECTED
            # ----------------------------------------------------

            if wrist_detected:

                total_frames += 1

                # ------------------------------------------------
                # DISTANCE TRAVELLED
                # ------------------------------------------------

                if prev_wrist_pos is not None:

                    step_dist = math.hypot(

                        wrist_pos[0]
                        -
                        prev_wrist_pos[0],

                        wrist_pos[1]
                        -
                        prev_wrist_pos[1]
                    )

                    distance_travelled += (
                        step_dist
                    )

                prev_wrist_pos = (
                    wrist_pos
                )

                # ------------------------------------------------
                # PATH DEVIATION
                # ------------------------------------------------

                dist_to_path = (
                    distance_point_to_polyline(
                        wrist_pos,
                        path_points
                    )
                )

                total_path_deviation += (
                    dist_to_path
                )

                deviation_samples += 1

                # ------------------------------------------------
                # PATH ACCURACY
                # ------------------------------------------------

                if (
                    dist_to_path
                    <= tolerance_px
                ):

                    wrist_on_path = True

                    on_path_frames += 1

                    score += 1

                # ------------------------------------------------
                # CHECKPOINT DETECTION
                # ------------------------------------------------

                if (
                    current_cp_index
                    <
                    len(checkpoints)
                ):

                    cp_x, cp_y = (
                        checkpoints[
                            current_cp_index
                        ]
                    )

                    dist_to_cp = math.hypot(

                        wrist_pos[0]
                        -
                        cp_x,

                        wrist_pos[1]
                        -
                        cp_y
                    )

                    if (
                        dist_to_cp
                        <= tolerance_px
                    ):

                        current_cp_index += 1

                        score += 50

                # ------------------------------------------------
                # DRAW WRIST
                # ------------------------------------------------

                marker_color = (

                    COLOR_WRIST_ON

                    if wrist_on_path

                    else

                    COLOR_WRIST_OFF
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

            # ----------------------------------------------------
            # WRIST NOT DETECTED
            # ----------------------------------------------------

            else:

                draw_wrist_not_detected_warning(
                    frame
                )

                prev_wrist_pos = None

            # ----------------------------------------------------
            # ACCURACY
            # ----------------------------------------------------

            accuracy = (

                on_path_frames
                /
                total_frames
                *
                100

            ) if total_frames > 0 else 0.0

            # ----------------------------------------------------
            # AVERAGE DEVIATION
            # ----------------------------------------------------

            average_deviation = (

                total_path_deviation
                /
                deviation_samples

            ) if deviation_samples > 0 else 0.0

            # ----------------------------------------------------
            # HUD
            # ----------------------------------------------------

            draw_hud(

                frame,

                score,

                accuracy,

                time_left,

                current_cp_index,

                len(checkpoints),

                wrist_on_path,

                average_deviation
            )

            # ----------------------------------------------------
            # END CONDITION
            # ----------------------------------------------------

            if (
                time_left <= 0
                or
                current_cp_index
                >= len(checkpoints)
            ):

                results = {

                    "difficulty":
                        difficulty,

                    "score":
                        score,

                    "accuracy":
                        accuracy,

                    "average_deviation":
                        average_deviation,

                    "time_taken":
                        elapsed,

                    "distance":
                        distance_travelled,

                    "checkpoints_reached":
                        current_cp_index,

                    "total_checkpoints":
                        len(checkpoints)
                }

                # ------------------------------------------------
                # SAVE ONLY ONCE
                # ------------------------------------------------

                if not results_saved:

                    save_success = (
                        save_follow_path_result(
                            results
                        )
                    )

                    results_saved = True

                    if not save_success:

                        print(
                            "WARNING: "
                            "Results were not saved."
                        )

                state = (
                    STATE_COMPLETE
                )

        # ========================================================
        # SESSION COMPLETE
        # ========================================================

        elif state == STATE_COMPLETE:

            draw_session_complete(
                frame,
                results
            )

        # ========================================================
        # FULLSCREEN DISPLAY
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

        # ========================================================
        # QUIT
        # ========================================================

        if key == ord("q"):

            # ------------------------------------------------
            # SAVE PARTIAL SESSION WHEN Q IS PRESSED
            # ------------------------------------------------
            # If the game is already playing, save the current
            # progress before closing. This prevents the session
            # from being lost when the user quits manually.

            if (
                state == STATE_PLAYING
                and not results_saved
            ):

                elapsed = (
                    time.time()
                    -
                    session_start_time
                )

                accuracy = (

                    (
                        on_path_frames
                        /
                        total_frames
                    )
                    *
                    100.0

                ) if total_frames > 0 else 0.0

                average_deviation = (

                    total_path_deviation
                    /
                    deviation_samples

                ) if deviation_samples > 0 else 0.0

                results = {

                    "difficulty":
                        difficulty,

                    "score":
                        score,

                    "accuracy":
                        accuracy,

                    "average_deviation":
                        average_deviation,

                    "time_taken":
                        elapsed,

                    "distance":
                        distance_travelled,

                    "checkpoints_reached":
                        current_cp_index,

                    "total_checkpoints":
                        len(checkpoints)
                }

                save_success = (
                    save_follow_path_result(
                        results
                    )
                )

                results_saved = True

                if not save_success:

                    print(
                        "WARNING: "
                        "Results were not saved."
                    )

            break

        # ========================================================
        # MENU INPUT
        # ========================================================

        if state == STATE_MENU:

            if key in (
                ord("1"),
                ord("2"),
                ord("3")
            ):

                difficulty = {

                    ord("1"):
                        "EASY",

                    ord("2"):
                        "MEDIUM",

                    ord("3"):
                        "HARD"

                }[key]

                (
                    path_points,
                    checkpoints,
                    tolerance_px
                ) = generate_path(
                    difficulty,
                    w,
                    h
                )

                state = (
                    STATE_START_SCREEN
                )

        # ========================================================
        # START SCREEN INPUT
        # ========================================================

        elif state == STATE_START_SCREEN:

            if key == ord(" "):

                # Reset metrics
                current_cp_index = 0

                score = 0

                on_path_frames = 0

                total_frames = 0

                distance_travelled = 0.0

                prev_wrist_pos = None

                total_path_deviation = 0.0

                deviation_samples = 0

                session_start_time = (
                    time.time()
                )

                results_saved = False

                state = STATE_PLAYING

        # ========================================================
        # COMPLETE SCREEN INPUT
        # ========================================================

        elif state == STATE_COMPLETE:

            if key == ord("r"):

                state = STATE_MENU

                difficulty = None

                results = {}

                results_saved = False

    # ============================================================
    # CLEANUP
    # ============================================================

    cap.release()

    cv2.destroyAllWindows()

    landmarker.close()

    print(
        "MOVENTRA - Follow the Path closed."
    )


# ================================================================
# PROGRAM ENTRY POINT
# ================================================================

if __name__ == "__main__":

    main()