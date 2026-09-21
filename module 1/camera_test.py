import cv2
import mediapipe as mp
import math
import time

from collections import deque
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ============================================================
# MODEL SETUP
# ============================================================

MODEL_PATH = r"C:\Users\SKS\OneDrive\Desktop\FYP\models\pose_landmarker_full.task"

base_options = python.BaseOptions(
    model_asset_path=MODEL_PATH
)

options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_poses=1
)

pose_landmarker = vision.PoseLandmarker.create_from_options(options)


# ============================================================
# REGISTRATION DATA
# ============================================================

patient_id = ""
patient_name = ""
patient_age = ""
selected_arm = "Left Arm"


# ============================================================
# REGISTRATION SCREEN
# ============================================================

def registration_screen():

    global patient_id
    global patient_name
    global patient_age
    global selected_arm

    window_name = "Patient Registration"

    cv2.namedWindow(
        window_name,
        cv2.WINDOW_NORMAL
    )

    cv2.resizeWindow(
        window_name,
        900,
        650
    )

    # --------------------------------------------------------
    # Text input variables
    # --------------------------------------------------------

    fields = {
        "Patient ID": "",
        "Patient Name": "",
        "Age": ""
    }

    field_names = [
        "Patient ID",
        "Patient Name",
        "Age"
    ]

    active_field = 0

    selected_arm = "Left Arm"

    message = ""

    # --------------------------------------------------------
    # Mouse callback
    # --------------------------------------------------------

    def mouse_callback(event, x, y, flags, param):

        nonlocal active_field
        global selected_arm

        if event != cv2.EVENT_LBUTTONDOWN:
            return

        # Patient ID field
        if 250 <= x <= 750 and 170 <= y <= 220:
            active_field = 0

        # Patient Name field
        elif 250 <= x <= 750 and 250 <= y <= 300:
            active_field = 1

        # Age field
        elif 250 <= x <= 750 and 330 <= y <= 380:
            active_field = 2

        # Left arm
        elif 270 <= x <= 450 and 420 <= y <= 470:
            selected_arm = "Left Arm"

        # Right arm
        elif 500 <= x <= 680 and 420 <= y <= 470:
            selected_arm = "Right Arm"

    cv2.setMouseCallback(
        window_name,
        mouse_callback
    )

    # ========================================================
    # REGISTRATION LOOP
    # ========================================================

    while True:

        screen = cv2.imread(
            # Create a blank screen below instead of relying on
            # an external image.
            ""
        )

        # The above read intentionally fails, so create a blank
        # screen manually.
        screen = cv2.rectangle(
            cv2.UMat(
                650,
                900,
                cv2.CV_8UC3
            ),
            (0, 0),
            (899, 649),
            (245, 245, 245),
            -1
        )

        screen = screen.get()

        # ----------------------------------------------------
        # Header
        # ----------------------------------------------------

        cv2.rectangle(
            screen,
            (0, 0),
            (900, 100),
            (70, 50, 150),
            -1
        )

        cv2.putText(
            screen,
            "STROKE REHABILITATION SYSTEM",
            (190, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2
        )

        cv2.putText(
            screen,
            "Patient Registration",
            (320, 82),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (230, 230, 230),
            2
        )

        # ----------------------------------------------------
        # Labels
        # ----------------------------------------------------

        cv2.putText(
            screen,
            "Patient ID",
            (100, 200),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (40, 40, 40),
            2
        )

        cv2.putText(
            screen,
            "Patient Name",
            (100, 280),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (40, 40, 40),
            2
        )

        cv2.putText(
            screen,
            "Age",
            (100, 360),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (40, 40, 40),
            2
        )

        # ----------------------------------------------------
        # Input boxes
        # ----------------------------------------------------

        for i, name in enumerate(field_names):

            y1 = 170 + i * 80
            y2 = 220 + i * 80

            border_color = (
                (70, 50, 150)
                if active_field == i
                else (150, 150, 150)
            )

            cv2.rectangle(
                screen,
                (250, y1),
                (750, y2),
                (255, 255, 255),
                -1
            )

            cv2.rectangle(
                screen,
                (250, y1),
                (750, y2),
                border_color,
                2
            )

            cv2.putText(
                screen,
                fields[name],
                (265, y1 + 34),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (30, 30, 30),
                2
            )

        # ----------------------------------------------------
        # Training arm
        # ----------------------------------------------------

        cv2.putText(
            screen,
            "Training Arm",
            (100, 455),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (40, 40, 40),
            2
        )

        # Left arm box

        left_color = (
            (70, 50, 150)
            if selected_arm == "Left Arm"
            else (180, 180, 180)
        )

        cv2.rectangle(
            screen,
            (270, 420),
            (450, 470),
            left_color,
            2
        )

        cv2.putText(
            screen,
            "Left Arm",
            (310, 453),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            left_color,
            2
        )

        # Right arm box

        right_color = (
            (70, 50, 150)
            if selected_arm == "Right Arm"
            else (180, 180, 180)
        )

        cv2.rectangle(
            screen,
            (500, 420),
            (680, 470),
            right_color,
            2
        )

        cv2.putText(
            screen,
            "Right Arm",
            (530, 453),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            right_color,
            2
        )

        # ----------------------------------------------------
        # Start session button
        # ----------------------------------------------------

        cv2.rectangle(
            screen,
            (330, 520),
            (570, 580),
            (70, 50, 150),
            -1
        )

        cv2.putText(
            screen,
            "START SESSION",
            (365, 558),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        # ----------------------------------------------------
        # Message
        # ----------------------------------------------------

        if message:

            cv2.putText(
                screen,
                message,
                (250, 615),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 255),
                2
            )

        # ----------------------------------------------------
        # Display
        # ----------------------------------------------------

        cv2.imshow(
            window_name,
            screen
        )

        key = cv2.waitKey(1) & 0xFF

        # ----------------------------------------------------
        # Keyboard input
        # ----------------------------------------------------

        if key != 255:

            # ESC
            if key == 27:

                cv2.destroyWindow(
                    window_name
                )

                return False

            # Backspace
            elif key == 8:

                current_field = field_names[active_field]

                fields[current_field] = (
                    fields[current_field][:-1]
                )

            # Enter
            elif key == 13:

                if active_field < len(field_names) - 1:

                    active_field += 1

            # Normal character
            elif 32 <= key <= 126:

                current_field = field_names[active_field]

                # Age should contain numbers only
                if current_field == "Age":

                    if chr(key).isdigit():

                        fields[current_field] += chr(key)

                else:

                    fields[current_field] += chr(key)

        # ----------------------------------------------------
        # Start button click detection
        # ----------------------------------------------------

        mouse_state = cv2.getWindowImageRect(
            window_name
        )

        # We handle the button through mouse callback below.
        # The actual start action is checked using the current
        # mouse position.

        try:

            _, _, mouse_x, mouse_y = cv2.getWindowImageRect(
                window_name
            )

        except:

            mouse_x = 0
            mouse_y = 0

        # ----------------------------------------------------
        # Check mouse click using callback state
        # ----------------------------------------------------

        # This section is handled separately below.


        # ----------------------------------------------------
        # START SESSION using SPACE
        # ----------------------------------------------------

        if key == ord("s"):

            if (
                fields["Patient ID"].strip() != ""
                and
                fields["Patient Name"].strip() != ""
                and
                fields["Age"].strip() != ""
            ):

                patient_id = fields["Patient ID"].strip()

                patient_name = fields["Patient Name"].strip()

                patient_age = fields["Age"].strip()

                print("\n===================================")
                print("      PATIENT SESSION STARTED")
                print("===================================")
                print(f"Patient ID   : {patient_id}")
                print(f"Patient Name : {patient_name}")
                print(f"Age          : {patient_age}")
                print(f"Training Arm : {selected_arm}")
                print("===================================\n")

                cv2.destroyWindow(
                    window_name
                )

                return True

            else:

                message = "Please fill all patient details."


# ============================================================
# START REGISTRATION
# ============================================================

registration_success = registration_screen()


if not registration_success:

    pose_landmarker.close()

    cv2.destroyAllWindows()

    exit()


# ============================================================
# FUNCTIONS
# ============================================================

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


def calculate_distance(point1, point2):

    return math.sqrt(
        (point2[0] - point1[0]) ** 2 +
        (point2[1] - point1[1]) ** 2
    )


# ============================================================
# CAMERA
# ============================================================

cap = cv2.VideoCapture(0)

window_name = "Stroke Rehabilitation - Movement Analysis"

cv2.namedWindow(
    window_name,
    cv2.WINDOW_NORMAL
)

cv2.resizeWindow(
    window_name,
    1000,
    700
)


# ============================================================
# WRIST PATH
# ============================================================

left_wrist_path = deque(maxlen=50)
right_wrist_path = deque(maxlen=50)


# ============================================================
# ANGLE SMOOTHING
# ============================================================

left_angle_history = deque(maxlen=7)


# ============================================================
# REPETITION VARIABLES
# ============================================================

left_reps = 0
left_stage = "START"

bent_frames = 0
straight_frames = 0

REQUIRED_FRAMES = 4


# ============================================================
# CALIBRATION
# ============================================================

calibration_mode = False
calibration_complete = False

calibration_angles = []

CALIBRATION_REPS = 5

calibration_min_angle = None
calibration_max_angle = None

personal_rom = None

calibration_stage = "START"

calibration_bent_frames = 0
calibration_straight_frames = 0

calibration_rep_count = 0

CALIBRATION_REQUIRED_FRAMES = 4


# ============================================================
# MOVEMENT METRICS
# ============================================================

movement_start_time = None
movement_times = []

current_rep_min_angle = None
current_rep_max_angle = None

rep_rom_values = []

previous_wrist = None
previous_time = None

wrist_distances = []

average_speed = 0


# ============================================================
# TRUNK COMPENSATION VARIABLES
# ============================================================

baseline_shoulder = None
baseline_hip = None

baseline_torso_length = None

trunk_displacements = []

current_rep_max_trunk_displacement = 0

current_trunk_displacement = 0


# ============================================================
# TIMESTAMP
# ============================================================

timestamp = 0


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    ret, frame = cap.read()

    if not ret:

        print("Could not access camera")

        break

    frame = cv2.flip(
        frame,
        1
    )

    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    timestamp += 1

    results = pose_landmarker.detect_for_video(
        mp_image,
        timestamp
    )


    # ========================================================
    # PERSON DETECTED
    # ========================================================

    if results.pose_landmarks:

        landmarks = results.pose_landmarks[0]

        h, w, _ = frame.shape


        # ----------------------------------------------------
        # UPPER BODY LANDMARKS
        # ----------------------------------------------------

        left_shoulder = landmarks[11]
        left_elbow = landmarks[13]
        left_wrist = landmarks[15]

        right_shoulder = landmarks[12]
        right_elbow = landmarks[14]
        right_wrist = landmarks[16]

        left_hip = landmarks[23]
        right_hip = landmarks[24]


        # ----------------------------------------------------
        # PIXEL COORDINATES
        # ----------------------------------------------------

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

        lh = (
            int(left_hip.x * w),
            int(left_hip.y * h)
        )

        rh = (
            int(right_hip.x * w),
            int(right_hip.y * h)
        )


        # ----------------------------------------------------
        # ELBOW ANGLES
        # ----------------------------------------------------

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

        left_angle_history.append(
            left_angle
        )

        smooth_left_angle = (
            sum(left_angle_history)
            /
            len(left_angle_history)
        )


        # ====================================================
        # TRUNK CENTER
        # ====================================================

        shoulder_center = (
            (ls[0] + rs[0]) / 2,
            (ls[1] + rs[1]) / 2
        )

        hip_center = (
            (lh[0] + rh[0]) / 2,
            (lh[1] + rh[1]) / 2
        )


        torso_length = calculate_distance(
            shoulder_center,
            hip_center
        )


        # ====================================================
        # CALIBRATION MODE
        # ====================================================

        if calibration_mode:

            if baseline_shoulder is None:

                baseline_shoulder = shoulder_center

                baseline_hip = hip_center

                baseline_torso_length = torso_length


            if smooth_left_angle < 90:

                calibration_bent_frames += 1

                calibration_straight_frames = 0

                if calibration_bent_frames >= CALIBRATION_REQUIRED_FRAMES:

                    if (
                        calibration_stage == "STRAIGHT"
                        or
                        calibration_stage == "START"
                    ):

                        calibration_stage = "BENT"

                        calibration_angles.append(
                            smooth_left_angle
                        )


            elif smooth_left_angle > 140:

                calibration_straight_frames += 1

                calibration_bent_frames = 0

                if calibration_straight_frames >= CALIBRATION_REQUIRED_FRAMES:

                    if calibration_stage == "BENT":

                        calibration_stage = "STRAIGHT"

                        calibration_rep_count += 1

                        calibration_angles.append(
                            smooth_left_angle
                        )

                        if calibration_rep_count >= CALIBRATION_REPS:

                            calibration_mode = False

                            calibration_complete = True

                            calibration_min_angle = min(
                                calibration_angles
                            )

                            calibration_max_angle = max(
                                calibration_angles
                            )

                            personal_rom = (
                                calibration_max_angle
                                -
                                calibration_min_angle
                            )

                            print("\nCalibration completed!")

                            print(
                                f"Minimum angle: "
                                f"{calibration_min_angle:.2f}"
                            )

                            print(
                                f"Maximum angle: "
                                f"{calibration_max_angle:.2f}"
                            )

                            print(
                                f"Personal ROM: "
                                f"{personal_rom:.2f} degrees"
                            )

            else:

                calibration_bent_frames = 0

                calibration_straight_frames = 0


        # ====================================================
        # NORMAL EXERCISE MODE
        # ====================================================

        elif calibration_complete:

            bent_threshold = (
                calibration_min_angle + 10
            )

            straight_threshold = (
                calibration_max_angle - 10
            )


            # =================================================
            # TRACK CURRENT REP ANGLE
            # =================================================

            if left_stage == "BENT":

                if current_rep_min_angle is None:

                    current_rep_min_angle = smooth_left_angle

                else:

                    current_rep_min_angle = min(
                        current_rep_min_angle,
                        smooth_left_angle
                    )


                if current_rep_max_angle is None:

                    current_rep_max_angle = smooth_left_angle

                else:

                    current_rep_max_angle = max(
                        current_rep_max_angle,
                        smooth_left_angle
                    )


            # =================================================
            # BENT POSITION
            # =================================================

            if smooth_left_angle < bent_threshold:

                bent_frames += 1

                straight_frames = 0

                if bent_frames >= REQUIRED_FRAMES:

                    if left_stage != "BENT":

                        left_stage = "BENT"

                        movement_start_time = time.time()

                        current_rep_min_angle = smooth_left_angle

                        current_rep_max_angle = smooth_left_angle

                        current_rep_max_trunk_displacement = 0

                        wrist_distances = []

                    else:

                        left_stage = "BENT"


            # =================================================
            # STRAIGHT POSITION
            # =================================================

            elif smooth_left_angle > straight_threshold:

                straight_frames += 1

                bent_frames = 0

                if (
                    straight_frames >= REQUIRED_FRAMES
                    and
                    left_stage == "BENT"
                ):

                    left_stage = "STRAIGHT"

                    left_reps += 1


                    # ----------------------------------------
                    # MOVEMENT TIME
                    # ----------------------------------------

                    if movement_start_time is not None:

                        movement_time = (
                            time.time()
                            -
                            movement_start_time
                        )

                        movement_times.append(
                            movement_time
                        )

                        movement_start_time = None


                    # ----------------------------------------
                    # ROM
                    # ----------------------------------------

                    if (
                        current_rep_min_angle is not None
                        and
                        current_rep_max_angle is not None
                    ):

                        rep_rom = (
                            current_rep_max_angle
                            -
                            current_rep_min_angle
                        )

                        rep_rom_values.append(
                            rep_rom
                        )


                    # ----------------------------------------
                    # TRUNK MOVEMENT
                    # ----------------------------------------

                    trunk_displacements.append(
                        current_rep_max_trunk_displacement
                    )


                    # Reset current rep

                    current_rep_min_angle = None

                    current_rep_max_angle = None

                    current_rep_max_trunk_displacement = 0


            # =================================================
            # MIDDLE POSITION
            # =================================================

            else:

                bent_frames = 0

                straight_frames = 0


            # =================================================
            # TRUNK DISPLACEMENT
            # =================================================

            if baseline_shoulder is not None:

                trunk_displacement = calculate_distance(
                    baseline_shoulder,
                    shoulder_center
                )

                current_trunk_displacement = (
                    trunk_displacement
                )

                if baseline_torso_length > 0:

                    normalized_trunk = (
                        trunk_displacement
                        /
                        baseline_torso_length
                    )

                if left_stage == "BENT":

                    current_rep_max_trunk_displacement = max(
                        current_rep_max_trunk_displacement,
                        trunk_displacement
                    )


            # =================================================
            # WRIST SPEED
            # =================================================

            current_time = time.time()

            if previous_wrist is not None:

                distance = calculate_distance(
                    previous_wrist,
                    lw
                )

                time_difference = (
                    current_time
                    -
                    previous_time
                )

                if time_difference > 0:

                    speed = (
                        distance
                        /
                        time_difference
                    )

                    wrist_distances.append(
                        speed
                    )


            previous_wrist = lw

            previous_time = current_time


            # =================================================
            # AVERAGE WRIST SPEED
            # =================================================

            if len(wrist_distances) > 0:

                average_speed = (
                    sum(wrist_distances)
                    /
                    len(wrist_distances)
                )


        # ====================================================
        # WRIST PATH
        # ====================================================

        left_wrist_path.append(lw)

        right_wrist_path.append(rw)


        # ====================================================
        # DRAW ARM SKELETON
        # ====================================================

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


        # ====================================================
        # DRAW TRUNK
        # ====================================================

        cv2.line(
            frame,
            ls,
            rs,
            (255, 255, 0),
            2
        )

        cv2.line(
            frame,
            lh,
            rh,
            (255, 255, 0),
            2
        )

        cv2.line(
            frame,
            (
                int(shoulder_center[0]),
                int(shoulder_center[1])
            ),
            (
                int(hip_center[0]),
                int(hip_center[1])
            ),
            (255, 255, 0),
            2
        )


        # ====================================================
        # LANDMARK POINTS
        # ====================================================

        for point in [
            ls,
            le,
            lw,
            rs,
            re,
            rw,
            lh,
            rh
        ]:

            cv2.circle(
                frame,
                point,
                7,
                (0, 255, 0),
                -1
            )


        # ====================================================
        # WRIST TRAJECTORY
        # ====================================================

        for i in range(
            1,
            len(left_wrist_path)
        ):

            cv2.line(
                frame,
                left_wrist_path[i - 1],
                left_wrist_path[i],
                (255, 0, 0),
                2
            )


        # ====================================================
        # BASIC INFORMATION
        # ====================================================

        cv2.putText(
            frame,
            f"Patient: {patient_name}",
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Training Arm: {selected_arm}",
            (30, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Left Elbow: {int(smooth_left_angle)} deg",
            (30, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Right Elbow: {int(right_angle)} deg",
            (30, 145),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )


        # ====================================================
        # CALIBRATION DISPLAY
        # ====================================================

        if calibration_mode:

            cv2.putText(
                frame,
                "CALIBRATION MODE",
                (30, 190),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 255),
                3
            )

            cv2.putText(
                frame,
                f"Movements: {calibration_rep_count}/5",
                (30, 230),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2
            )

            cv2.putText(
                frame,
                "Perform comfortable elbow bends",
                (30, 270),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )


        # ====================================================
        # NORMAL MODE DISPLAY
        # ====================================================

        elif calibration_complete:

            cv2.putText(
                frame,
                f"Repetitions: {left_reps}",
                (30, 190),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 255),
                3
            )

            cv2.putText(
                frame,
                f"Stage: {left_stage}",
                (30, 225),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"Personal ROM: {int(personal_rom)} deg",
                (30, 260),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 200, 0),
                2
            )


            if len(rep_rom_values) > 0:

                cv2.putText(
                    frame,
                    f"Last Rep ROM: "
                    f"{int(rep_rom_values[-1])} deg",
                    (30, 295),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )


            if len(movement_times) > 0:

                cv2.putText(
                    frame,
                    f"Last Movement: "
                    f"{movement_times[-1]:.2f} sec",
                    (30, 330),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2
                )


            cv2.putText(
                frame,
                f"Wrist Speed: "
                f"{average_speed:.1f} px/sec",
                (30, 365),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 0, 0),
                2
            )


            cv2.putText(
                frame,
                f"Trunk Movement: "
                f"{current_trunk_displacement:.1f} px",
                (30, 400),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 100, 255),
                2
            )


            if baseline_torso_length is not None:

                normalized_display = (
                    current_trunk_displacement
                    /
                    baseline_torso_length
                )

                cv2.putText(
                    frame,
                    f"Trunk Ratio: "
                    f"{normalized_display:.2f}",
                    (30, 435),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 100, 255),
                    2
                )


        # ====================================================
        # BEFORE CALIBRATION
        # ====================================================

        else:

            cv2.putText(
                frame,
                "Press C to start calibration",
                (30, 190),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2
            )

            cv2.putText(
                frame,
                "Press R to reset",
                (30, 230),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )


        # ====================================================
        # WRIST COORDINATES
        # ====================================================

        cv2.putText(
            frame,
            f"Left Wrist: X={lw[0]} Y={lw[1]}",
            (30, 475),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 0),
            2
        )


    # ========================================================
    # DISPLAY
    # ========================================================

    cv2.imshow(
        window_name,
        frame
    )


    # ========================================================
    # KEYBOARD
    # ========================================================

    key = cv2.waitKey(1) & 0xFF


    # ========================================================
    # START CALIBRATION
    # ========================================================

    if key == ord("c"):

        calibration_mode = True

        calibration_complete = False

        calibration_angles = []

        calibration_rep_count = 0

        calibration_stage = "START"

        calibration_bent_frames = 0

        calibration_straight_frames = 0

        left_reps = 0

        left_stage = "START"

        baseline_shoulder = None
        baseline_hip = None
        baseline_torso_length = None

        movement_start_time = None

        movement_times = []

        current_rep_min_angle = None

        current_rep_max_angle = None

        rep_rom_values = []

        previous_wrist = None

        previous_time = None

        wrist_distances = []

        average_speed = 0

        trunk_displacements = []

        current_rep_max_trunk_displacement = 0

        current_trunk_displacement = 0

        print("\nCalibration started.")

        print("Perform 5 comfortable elbow bends.")


    # ========================================================
    # RESET
    # ========================================================

    elif key == ord("r"):

        calibration_mode = False

        calibration_complete = False

        calibration_angles = []

        calibration_rep_count = 0

        calibration_min_angle = None

        calibration_max_angle = None

        personal_rom = None

        left_reps = 0

        left_stage = "START"

        bent_frames = 0

        straight_frames = 0

        baseline_shoulder = None
        baseline_hip = None
        baseline_torso_length = None

        movement_start_time = None

        movement_times = []

        current_rep_min_angle = None

        current_rep_max_angle = None

        rep_rom_values = []

        previous_wrist = None

        previous_time = None

        wrist_distances = []

        average_speed = 0

        trunk_displacements = []

        current_rep_max_trunk_displacement = 0

        current_trunk_displacement = 0

        print("\nSystem reset.")


    # ========================================================
    # QUIT
    # ========================================================

    elif key == ord("q"):

        break


# ============================================================
# CLEANUP
# ============================================================

cap.release()

pose_landmarker.close()

cv2.destroyAllWindows()