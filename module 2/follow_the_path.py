"""
=====================================================================
 MOVENTRA - AR/VR Rehabilitation System for Stroke Patients
 Module 2: Gamified Rehabilitation
 Game 2: FOLLOW THE PATH

 A software-only, non-immersive, camera-based rehabilitation game.
 The patient moves their LEFT hand/wrist in front of the webcam to
 follow a visible path drawn on screen. This helps stroke patients
 practice smooth, controlled arm movement.

 Tech stack:
   - Python 3.12
   - OpenCV (camera feed + drawing)
   - MediaPipe Pose Landmarker (Tasks API) for markerless body tracking

 Run with:
   python "module 2\\follow_the_path.py"

 Controls:
   1 / 2 / 3  -> Select Easy / Medium / Hard (at the menu screen)
   SPACE      -> Start the game (at the start screen)
   R          -> Restart (after session complete)
   Q          -> Quit (any screen)
=====================================================================
"""

import cv2
import time
import math
import os
import sys
import numpy as np          # already installed as a dependency of OpenCV/MediaPipe
import tkinter as tk        # standard library - used only to read the screen resolution

# ---------------------------------------------------------------
# MediaPipe Tasks API imports (same style as Balloon Reach)
# ---------------------------------------------------------------
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ---------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------
MODEL_PATH = "models/pose_landmarker_full.task"   # relative model path
CAM_INDEX = 0                                      # default webcam
SESSION_SECONDS = 60                                # fixed session length
WINDOW_NAME = "MOVENTRA - Follow the Path"          # single source of truth for the window title

# Index of the LEFT WRIST landmark in MediaPipe's 33-point Pose model.
# (11=left_shoulder, 13=left_elbow, 15=left_wrist)
# NOTE: MediaPipe assigns this label based on the RAW (unflipped) camera
# image, so it already corresponds to the patient's true anatomical left
# wrist -- as long as detection runs on the unflipped frame (see main loop).
LEFT_WRIST_INDEX = 15

# Colors (BGR format, since OpenCV uses BGR not RGB)
COLOR_BG_TEXT = (255, 255, 255)
COLOR_PATH = (200, 200, 0)          # path line color (cyan-ish)
COLOR_PATH_DONE = (0, 200, 0)       # completed part of path (green)
COLOR_CHECKPOINT = (0, 165, 255)    # upcoming checkpoint (orange)
COLOR_CHECKPOINT_DONE = (0, 255, 0) # reached checkpoint (green)
COLOR_WRIST_ON = (0, 255, 0)        # wrist marker when ON the path
COLOR_WRIST_OFF = (0, 0, 255)       # wrist marker when OFF the path
COLOR_WARNING = (0, 0, 255)

# ---------------------------------------------------------------
# GAME STATES
# ---------------------------------------------------------------
STATE_MENU = "MENU"                     # choose difficulty
STATE_START_SCREEN = "START_SCREEN"     # camera shown, press SPACE
STATE_PLAYING = "PLAYING"               # active session
STATE_COMPLETE = "SESSION_COMPLETE"     # results screen


# =====================================================================
# PATH GENERATION
# =====================================================================
def generate_path(difficulty, width, height):
    """
    Generates a smooth sine-wave shaped path across the screen.
    Returns:
        path_points   -> list of (x, y) points used to DRAW the curve (smooth line)
        checkpoints   -> list of (x, y) points the patient must reach, in order
        tolerance_px  -> how close (in pixels) the wrist must be to count as "on path"
    Difficulty changes the wave complexity, the number of checkpoints,
    and how strict (tolerance) the path following needs to be.
    """
    # Difficulty settings: (amplitude, frequency, num_checkpoints, tolerance)
    settings = {
        "EASY":   {"amp": 60,  "freq": 1, "checkpoints": 6,  "tolerance": 60},
        "MEDIUM": {"amp": 90,  "freq": 2, "checkpoints": 8,  "tolerance": 45},
        "HARD":   {"amp": 120, "freq": 3, "checkpoints": 10, "tolerance": 30},
    }
    cfg = settings[difficulty]

    # Usable horizontal margin so the path doesn't touch screen edges
    left_margin = 80
    right_margin = width - 80
    mid_y = height // 2

    # Generate a smooth curve using a sine wave (many points = smooth line)
    path_points = []
    num_points = 300
    for i in range(num_points):
        t = i / (num_points - 1)
        x = int(left_margin + t * (right_margin - left_margin))
        y = int(mid_y + cfg["amp"] * math.sin(2 * math.pi * cfg["freq"] * t))
        path_points.append((x, y))

    # Pick evenly spaced checkpoints along the path for the patient to reach
    checkpoints = []
    num_cp = cfg["checkpoints"]
    for i in range(num_cp):
        idx = int(i * (num_points - 1) / (num_cp - 1))
        checkpoints.append(path_points[idx])

    return path_points, checkpoints, cfg["tolerance"]


def distance_point_to_polyline(point, polyline):
    """
    Returns the shortest distance from a point (wrist position) to the
    path polyline (a list of connected points). Used to measure how
    accurately the patient is following the path.
    """
    px, py = point
    min_dist = float("inf")
    for i in range(len(polyline) - 1):
        x1, y1 = polyline[i]
        x2, y2 = polyline[i + 1]
        # Distance from point to segment (x1,y1)-(x2,y2)
        seg_len_sq = (x2 - x1) ** 2 + (y2 - y1) ** 2
        if seg_len_sq == 0:
            dist = math.hypot(px - x1, py - y1)
        else:
            t = max(0, min(1, ((px - x1) * (x2 - x1) + (py - y1) * (y2 - y1)) / seg_len_sq))
            proj_x = x1 + t * (x2 - x1)
            proj_y = y1 + t * (y2 - y1)
            dist = math.hypot(px - proj_x, py - proj_y)
        if dist < min_dist:
            min_dist = dist
    return min_dist


# =====================================================================
# FULLSCREEN DISPLAY HELPER
# =====================================================================
def resize_with_letterbox(frame, target_w, target_h):
    """
    Resizes 'frame' to fit inside a (target_w x target_h) canvas WITHOUT
    stretching or distorting the image. The aspect ratio is preserved and
    any leftover space is filled with black bars (letterboxing), then the
    frame is centered on the canvas. This is what makes the fullscreen
    window look correct instead of stretching the patient's body.
    """
    h, w = frame.shape[:2]
    scale = min(target_w / w, target_h / h)
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2
    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
    return canvas


# =====================================================================
# DRAWING HELPER FUNCTIONS
# =====================================================================
def draw_text_center(frame, text, y, scale=1.0, color=COLOR_BG_TEXT, thickness=2):
    """Draws horizontally centered text on the frame at a given y position."""
    (text_w, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    x = (frame.shape[1] - text_w) // 2
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def draw_menu_screen(frame):
    """Difficulty selection menu shown at program start."""
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

    draw_text_center(frame, "FOLLOW THE PATH", 100, 1.4, (0, 255, 255), 3)
    draw_text_center(frame, "MOVENTRA - Gamified Rehabilitation", 140, 0.7, (200, 200, 200), 1)
    draw_text_center(frame, "Select Difficulty", 220, 1.0, COLOR_BG_TEXT, 2)
    draw_text_center(frame, "1 - Easy", 280, 0.9, (0, 255, 0), 2)
    draw_text_center(frame, "2 - Medium", 320, 0.9, (0, 165, 255), 2)
    draw_text_center(frame, "3 - Hard", 360, 0.9, (0, 0, 255), 2)
    draw_text_center(frame, "Press Q to Quit", frame.shape[0] - 40, 0.7, (180, 180, 180), 1)


def draw_start_screen(frame, difficulty):
    """Camera feed + instructions, waiting for SPACE to start."""
    draw_text_center(frame, "FOLLOW THE PATH", 60, 1.1, (0, 255, 255), 2)
    draw_text_center(frame, f"Difficulty: {difficulty}", 100, 0.8, COLOR_BG_TEXT, 2)
    draw_text_center(frame, "Use your LEFT hand to follow the path", 140, 0.7, COLOR_BG_TEXT, 1)
    draw_text_center(frame, "Press SPACE to Start", frame.shape[0] - 60, 0.9, (0, 255, 0), 2)
    draw_text_center(frame, "Press Q to Quit", frame.shape[0] - 25, 0.6, (180, 180, 180), 1)


def draw_path(frame, path_points, checkpoints, current_cp_index):
    """
    Draws the path curve, coloring the already-completed section green
    and the remaining section cyan. Also draws checkpoint circles.
    """
    # Draw the smooth path line, segment by segment
    for i in range(len(path_points) - 1):
        color = COLOR_PATH_DONE if i < current_cp_index * (len(path_points) // max(1, len(checkpoints))) else COLOR_PATH
        cv2.line(frame, path_points[i], path_points[i + 1], color, 4)

    # Draw checkpoints (numbered circles)
    for idx, cp in enumerate(checkpoints):
        if idx < current_cp_index:
            color = COLOR_CHECKPOINT_DONE
        elif idx == current_cp_index:
            color = COLOR_CHECKPOINT
        else:
            color = (120, 120, 120)
        radius = 14 if idx == current_cp_index else 8
        cv2.circle(frame, cp, radius, color, -1)
        cv2.circle(frame, cp, radius, (255, 255, 255), 2)


def draw_hud(frame, score, accuracy, time_left, checkpoints_reached, total_checkpoints, wrist_on_path):
    """Draws the heads-up display: score, accuracy, timer, checkpoints."""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 70), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    cv2.putText(frame, f"Score: {score}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"Accuracy: {accuracy:.0f}%", (20, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

    cp_text = f"Checkpoints: {checkpoints_reached}/{total_checkpoints}"
    (tw, _), _ = cv2.getTextSize(cp_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.putText(frame, cp_text, (w // 2 - tw // 2, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_BG_TEXT, 2)

    status_text = "ON PATH" if wrist_on_path else "OFF PATH - move back to the line"
    status_color = COLOR_WRIST_ON if wrist_on_path else COLOR_WRIST_OFF
    (tw2, _), _ = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.putText(frame, status_text, (w // 2 - tw2 // 2, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

    timer_text = f"Time: {int(time_left)}s"
    (tw3, _), _ = cv2.getTextSize(timer_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.putText(frame, timer_text, (w - tw3 - 20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)


def draw_wrist_not_detected_warning(frame):
    """Shown when the left wrist cannot be detected in the camera frame."""
    draw_text_center(frame, "LEFT WRIST NOT DETECTED", 110, 0.8, COLOR_WARNING, 2)
    draw_text_center(frame, "Please make sure your left arm is visible", 140, 0.6, COLOR_WARNING, 1)


def draw_session_complete(frame, results):
    """Final results screen shown after the 60-second session ends."""
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.9, frame, 0.1, 0, frame)

    draw_text_center(frame, "SESSION COMPLETE", 80, 1.3, (0, 255, 255), 3)

    lines = [
        f"Difficulty: {results['difficulty']}",
        f"Score: {results['score']}",
        f"Path-Following Accuracy: {results['accuracy']:.1f}%",
        f"Time Taken: {results['time_taken']:.1f} sec",
        f"Distance Travelled: {results['distance']:.0f} px",
        f"Checkpoints Reached: {results['checkpoints_reached']}/{results['total_checkpoints']}",
    ]
    y = 150
    for line in lines:
        draw_text_center(frame, line, y, 0.75, COLOR_BG_TEXT, 2)
        y += 40

    draw_text_center(frame, "Press R to Restart   |   Press Q to Quit", frame.shape[0] - 40, 0.7, (0, 255, 0), 2)


# =====================================================================
# MAIN PROGRAM
# =====================================================================
def main():
    # -----------------------------------------------------------
    # Check the pose model file exists before doing anything else
    # -----------------------------------------------------------
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Pose model not found at '{MODEL_PATH}'.")
        print("Make sure the 'models' folder with 'pose_landmarker_full.task' is next to this script.")
        sys.exit(1)

    # -----------------------------------------------------------
    # Set up MediaPipe Pose Landmarker (Tasks API, VIDEO mode)
    # -----------------------------------------------------------
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    landmarker = vision.PoseLandmarker.create_from_options(options)

    # -----------------------------------------------------------
    # Open the webcam
    # -----------------------------------------------------------
    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print("ERROR: Could not open the webcam. Check CAM_INDEX or camera permissions.")
        sys.exit(1)

    # -----------------------------------------------------------
    # Get the laptop screen resolution (standard library only, no
    # extra package needed) so we know how big to make the window.
    # -----------------------------------------------------------
    root = tk.Tk()
    root.withdraw()
    screen_w = root.winfo_screenwidth()
    screen_h = root.winfo_screenheight()
    root.destroy()

    # -----------------------------------------------------------
    # Create the OpenCV window ONCE, before the main loop, and make
    # it truly fullscreen. NOTE: namedWindow must be created with the
    # WINDOW_NORMAL flag for setWindowProperty(..., WINDOW_FULLSCREEN)
    # to take effect correctly on most backends.
    # The camera frame itself is resized/letterboxed every frame (see
    # resize_with_letterbox) so the fullscreen window is actually filled
    # instead of showing a small native-resolution image in a corner.
    # -----------------------------------------------------------
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # -----------------------------------------------------------
    # Game state variables
    # -----------------------------------------------------------
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

    session_start_time = 0.0
    frame_timestamp_ms = 0

    results = {}

    print("MOVENTRA - Follow the Path started. Press Q at any time to quit.")

    while True:
        ret, raw_frame = cap.read()
        if not ret:
            print("ERROR: Failed to read frame from webcam.")
            break

        h, w = raw_frame.shape[:2]

        # -----------------------------------------------------------
        # Run pose detection on the UNFLIPPED (raw) frame.
        #
        # MediaPipe's "left_wrist" (landmark 15) label is assigned based
        # on the raw camera image, so it corresponds to the patient's
        # true anatomical left wrist ONLY if detection runs before any
        # horizontal flip is applied. Previously the frame was flipped
        # first and then fed to the model, which silently swapped the
        # left/right labels -- that was the root cause of the game
        # tracking the RIGHT hand instead of the LEFT hand.
        # -----------------------------------------------------------
        wrist_pos = None
        wrist_detected = False

        if state in (STATE_START_SCREEN, STATE_PLAYING):
            rgb_frame = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            frame_timestamp_ms += 33  # approx. 30 FPS timestamp increment
            pose_result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

            if pose_result.pose_landmarks:
                landmarks = pose_result.pose_landmarks[0]
                wrist_lm = landmarks[LEFT_WRIST_INDEX]
                # Only trust the landmark if MediaPipe is reasonably confident it's visible
                if wrist_lm.visibility is None or wrist_lm.visibility > 0.3:
                    raw_x = int(wrist_lm.x * w)
                    raw_y = int(wrist_lm.y * h)
                    # Mirror the x-coordinate to match the horizontally
                    # flipped frame we display below, so the on-screen
                    # marker still lands on the correct (left) hand.
                    wrist_x = w - raw_x
                    wrist_y = raw_y
                    wrist_pos = (wrist_x, wrist_y)
                    wrist_detected = True

        # Now flip horizontally so the DISPLAY acts like a mirror
        # (natural for the patient). Detection already happened above
        # on the unflipped frame, so left/right labeling stays correct.
        frame = cv2.flip(raw_frame, 1)

        # =====================================================
        # STATE: MENU - difficulty selection
        # =====================================================
        if state == STATE_MENU:
            draw_menu_screen(frame)

        # =====================================================
        # STATE: START_SCREEN - waiting for SPACE
        # =====================================================
        elif state == STATE_START_SCREEN:
            draw_start_screen(frame, difficulty)
            if wrist_detected:
                cv2.circle(frame, wrist_pos, 12, COLOR_WRIST_ON, -1)
            else:
                draw_wrist_not_detected_warning(frame)

        # =====================================================
        # STATE: PLAYING - active rehabilitation session
        # =====================================================
        elif state == STATE_PLAYING:
            elapsed = time.time() - session_start_time
            time_left = max(0, SESSION_SECONDS - elapsed)

            # Draw the path and checkpoints first (background layer)
            draw_path(frame, path_points, checkpoints, current_cp_index)

            wrist_on_path = False
            if wrist_detected:
                total_frames += 1

                # Track distance travelled (sum of movement between frames)
                if prev_wrist_pos is not None:
                    step_dist = math.hypot(wrist_pos[0] - prev_wrist_pos[0], wrist_pos[1] - prev_wrist_pos[1])
                    distance_travelled += step_dist
                prev_wrist_pos = wrist_pos

                # Check how far the wrist is from the path line
                dist_to_path = distance_point_to_polyline(wrist_pos, path_points)
                if dist_to_path <= tolerance_px:
                    wrist_on_path = True
                    on_path_frames += 1
                    score += 1  # small continuous reward for staying on path

                # Check if the wrist has reached the current checkpoint
                if current_cp_index < len(checkpoints):
                    cp_x, cp_y = checkpoints[current_cp_index]
                    dist_to_cp = math.hypot(wrist_pos[0] - cp_x, wrist_pos[1] - cp_y)
                    if dist_to_cp <= tolerance_px:
                        current_cp_index += 1
                        score += 50  # bonus points for reaching a checkpoint

                # Draw the wrist marker (green if on path, red if off path)
                marker_color = COLOR_WRIST_ON if wrist_on_path else COLOR_WRIST_OFF
                cv2.circle(frame, wrist_pos, 15, marker_color, -1)
                cv2.circle(frame, wrist_pos, 15, (255, 255, 255), 2)
            else:
                draw_wrist_not_detected_warning(frame)
                prev_wrist_pos = None  # avoid a big "jump" once detection resumes

            # Compute live accuracy safely (avoid divide-by-zero)
            accuracy = (on_path_frames / total_frames * 100) if total_frames > 0 else 0.0

            draw_hud(frame, score, accuracy, time_left, current_cp_index, len(checkpoints), wrist_on_path)

            # End the session when time runs out OR all checkpoints are reached
            if time_left <= 0 or current_cp_index >= len(checkpoints):
                results = {
                    "difficulty": difficulty,
                    "score": score,
                    "accuracy": accuracy,
                    "time_taken": elapsed,
                    "distance": distance_travelled,
                    "checkpoints_reached": current_cp_index,
                    "total_checkpoints": len(checkpoints),
                }
                state = STATE_COMPLETE

        # =====================================================
        # STATE: SESSION_COMPLETE - show final results
        # =====================================================
        elif state == STATE_COMPLETE:
            draw_session_complete(frame, results)

        # -----------------------------------------------------------
        # Letterbox-resize the frame to fill the fullscreen window
        # without stretching/distorting the patient's body, then show it.
        # -----------------------------------------------------------
        display_frame = resize_with_letterbox(frame, screen_w, screen_h)
        cv2.imshow("MOVENTRA - Follow the Path", display_frame)

        # -----------------------------------------------------------
        # Handle keyboard input (single waitKey call per frame)
        # -----------------------------------------------------------
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        if state == STATE_MENU:
            if key in (ord('1'), ord('2'), ord('3')):
                difficulty = {ord('1'): "EASY", ord('2'): "MEDIUM", ord('3'): "HARD"}[key]
                path_points, checkpoints, tolerance_px = generate_path(difficulty, w, h)
                state = STATE_START_SCREEN

        elif state == STATE_START_SCREEN:
            if key == ord(' '):
                # Reset all session metrics right before starting
                current_cp_index = 0
                score = 0
                on_path_frames = 0
                total_frames = 0
                distance_travelled = 0.0
                prev_wrist_pos = None
                session_start_time = time.time()
                state = STATE_PLAYING

        elif state == STATE_COMPLETE:
            if key == ord('r'):
                # Go back to the menu so the patient/therapist can pick a difficulty again
                state = STATE_MENU
                difficulty = None

    # -----------------------------------------------------------
    # Cleanup
    # -----------------------------------------------------------
    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()
    print("MOVENTRA - Follow the Path closed.")


if __name__ == "__main__":
    main()