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

COLLISION_DISTANCE = BALLOON_RADIUS + 20

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
    age_input = input(
        "Enter Patient Age: "
    ).strip()

    try:
        age = int(age_input)

        if age < 1:
            print("Please enter a valid age.")
            continue

        break

    except ValueError:
        print("Please enter a valid age.")


while True:
    training_arm = input(
        "Enter Training Arm (LEFT/RIGHT): "
    ).strip().upper()

    if training_arm in ["LEFT", "RIGHT"]:
        break

    print("Please enter LEFT or RIGHT.")


while True:
    session_input = input(
        "Enter Session Number: "
    ).strip()

    try:
        session_number = int(session_input)

        if session_number < 1:
            print("Please enter a valid session number.")
            continue

        break

    except ValueError:
        print("Please enter a valid session number.")


print("----------------------------------------")
print(f"Patient ID   : {patient_id}")
print(f"Patient Name : {patient_name}")
print(f"Age          : {age}")
print(f"Training Arm : {training_arm}")
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
# WRIST LANDMARK
# ============================================================

LEFT_WRIST_INDEX = 15
RIGHT_WRIST_INDEX = 16

SELECTED_WRIST_INDEX = (
    RIGHT_WRIST_INDEX
    if training_arm == "RIGHT"
    else LEFT_WRIST_INDEX
)


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

    cv2.circle(
        frame,
        (x, y),
        BALLOON_RADIUS,
        (0, 0, 255),
        -1
    )

    cv2.circle(
        frame,
        (x - 10, y - 10),
        7,
        (255, 255, 255),
        -1
    )

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

def save_balloon_result(final_time=None):

    global save_success

    try:

        if os.path.exists(CSV_PATH):
            df = pd.read_csv(CSV_PATH)
        else:
            df = pd.DataFrame()


        required_columns = [

            "patient_id",
            "patient_name",
            "age",
            "training_arm",
            "session_number",
            "session_date",

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


        for column in required_columns:

            if column not in df.columns:

                if column in [
                    "patient_id",
                    "patient_name",
                    "training_arm",
                    "session_date",
                    "is_sample_data"
                ]:

                    df[column] = ""

                else:

                    df[column] = 0


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

        df["training_arm"] = (
            df["training_arm"]
            .fillna("")
            .astype(str)
        )

        df["session_date"] = (
            df["session_date"]
            .fillna("")
            .astype(str)
        )

        df["is_sample_data"] = (
            df["is_sample_data"]
            .fillna("NO")
            .astype(str)
        )


        # Case-insensitive patient ID matching.
        mask = (
            df["patient_id"]
            .str.strip()
            .str.lower()
            .eq(patient_id.strip().lower())
        ) & (
            df["session_number"] == session_number
        )


        if mask.any():

            row_index = df.index[mask][0]

        else:

            new_row = {
                column: 0
                for column in required_columns
            }

            new_row["patient_id"] = patient_id
            new_row["patient_name"] = patient_name
            new_row["age"] = age
            new_row["training_arm"] = training_arm
            new_row["session_number"] = session_number
            new_row["session_date"] = time.strftime("%Y-%m-%d")
            new_row["is_sample_data"] = "NO"

            df = pd.concat(
                [
                    df,
                    pd.DataFrame([new_row])
                ],
                ignore_index=True
            )

            row_index = df.index[-1]


        # Update patient/session information.
        df.at[
            row_index,
            "patient_name"
        ] = patient_name

        df.at[
            row_index,
            "age"
        ] = age

        df.at[
            row_index,
            "training_arm"
        ] = training_arm

        if (
            not str(
                df.at[row_index, "session_date"]
            ).strip()
            or str(
                df.at[row_index, "session_date"]
            ).lower() == "nan"
        ):

            df.at[
                row_index,
                "session_date"
            ] = time.strftime("%Y-%m-%d")


        # Final time:
        # If the game was completed normally, use 60 sec.
        # If Q was pressed during the game, save the actual
        # elapsed time.
        if final_time is None:
            final_time = GAME_TIME


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
        ] = float(final_time)

        df.at[
            row_index,
            "is_sample_data"
        ] = "NO"


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
            f"Patient     : {patient_id} - {patient_name}"
        )
        print(
            f"Age         : {age}"
        )
        print(
            f"Training Arm: {training_arm}"
        )
        print(
            f"Session     : {session_number}"
        )
        print(
            f"Score       : {score}"
        )
        print(
            f"Attempts    : {total_attempts}"
        )
        print(
            f"Targets     : {targets_reached}"
        )
        print(
            f"Accuracy    : {accuracy:.1f}%"
        )
        print(
            f"Time        : {final_time:.1f} sec"
        )
        print(
            f"CSV         : {CSV_PATH}"
        )
        print("========================================")

        return True


    except Exception as e:

        save_success = False

        print("\n========================================")
        print(
            "ERROR: Could not save Balloon Reach results."
        )
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


    # --------------------------------------------------------
    # IMPORTANT:
    # MediaPipe receives the ORIGINAL camera frame.
    # The frame is mirrored only for display.
    # This keeps LEFT/RIGHT anatomical landmarks correct.
    # --------------------------------------------------------

    detection_frame = frame.copy()

    height, width, _ = detection_frame.shape


    # MediaPipe processing
    rgb_frame = cv2.cvtColor(
        detection_frame,
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


    # --------------------------------------------------------
    # Mirror only the displayed frame
    # --------------------------------------------------------

    frame = cv2.flip(
        detection_frame,
        1
    )


    # ========================================================
    # FIND SELECTED WRIST
    # ========================================================

    wrist_x = None
    wrist_y = None


    if result.pose_landmarks:

        landmarks = result.pose_landmarks[0]

        if len(landmarks) > SELECTED_WRIST_INDEX:

            wrist = landmarks[
                SELECTED_WRIST_INDEX
            ]

            # Landmark coordinates belong to the original
            # unmirrored camera frame.

            original_x = int(
                wrist.x * width
            )

            original_y = int(
                wrist.y * height
            )

            # Convert X to mirrored display coordinates.
            wrist_x = (
                width - original_x
            )

            wrist_y = original_y


            # Draw selected wrist
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
            f"Age: {age}",
            (
                width // 2 - 70,
                275
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Training Arm: {training_arm}",
            (
                width // 2 - 160,
                315
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
                355
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
                400
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Move your {training_arm} hand to the balloon",
            (
                width // 2 - 290,
                455
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
                510
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
                560
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
            f"Training Arm: {training_arm}",
            (30, 210),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Difficulty: {difficulty}",
            (30, 250),
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

        if total_attempts > 0:

            accuracy = (
                targets_reached /
                total_attempts
            ) * 100

        else:

            accuracy = 0.0


        if not results_saved:

            save_success = (
                save_balloon_result(
                    GAME_TIME
                )
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

    # --------------------------------------------------------
    # QUIT
    # --------------------------------------------------------

    if key == ord("q"):

        # If Q is pressed while the game is running,
        # save the partial/current session first.
        if (
            game_started
            and not game_finished
            and not results_saved
        ):

            elapsed_before_quit = (
                time.time() -
                start_time
            )

            if elapsed_before_quit < 0:
                elapsed_before_quit = 0

            # Recalculate final accuracy.
            if total_attempts > 0:
                accuracy = (
                    targets_reached /
                    total_attempts
                ) * 100
            else:
                accuracy = 0.0

            save_success = save_balloon_result(
                elapsed_before_quit
            )

            results_saved = True

        break


    # --------------------------------------------------------
    # START GAME
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # RESTART
    # --------------------------------------------------------

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
