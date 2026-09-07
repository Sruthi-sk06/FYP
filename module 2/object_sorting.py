"""
=====================================================================
 MOVENTRA - AR/VR Rehabilitation System for Stroke Patients
 Module 2: Gamified Rehabilitation
 Game 3: OBJECT SORTING

 A software-only, non-immersive, camera-based rehabilitation game.
 Several colored objects are shown on screen and the patient is asked
 to reach the correctly-colored object using their LEFT hand/wrist.
 This encourages controlled, targeted upper-limb reaching movement.

 Tech stack:
   - Python 3.12
   - OpenCV (camera feed + drawing)
   - MediaPipe Pose Landmarker (Tasks API) for markerless body tracking

 Run with:
   python "module 2\\object_sorting.py"

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
import random
import numpy as np          # already installed as a dependency of OpenCV/MediaPipe
import tkinter as tk        # standard library - used only to read the screen resolution

# ---------------------------------------------------------------
# MediaPipe Tasks API imports (same style as Balloon Reach / Follow the Path)
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
WINDOW_NAME = "MOVENTRA - Object Sorting"           # single source of truth for the window title
FEEDBACK_SECONDS = 0.6                              # how long success/error flash stays on screen

# Index of the LEFT WRIST landmark in MediaPipe's 33-point Pose model.
# (11=left_shoulder, 13=left_elbow, 15=left_wrist)
# NOTE: MediaPipe assigns this label based on the RAW (unflipped) camera
# image, so it already corresponds to the patient's true anatomical left
# wrist -- as long as detection runs on the unflipped frame (see main loop).
LEFT_WRIST_INDEX = 15

# Colors (BGR format, since OpenCV uses BGR not RGB)
COLOR_BG_TEXT = (255, 255, 255)
COLOR_WRIST_NEUTRAL = (255, 255, 255)  # wrist marker when not touching anything
COLOR_CORRECT = (0, 255, 0)
COLOR_WRONG = (0, 0, 255)
COLOR_WARNING = (0, 0, 255)

# Palette of colored objects the patient can be asked to find.
# 8 distinct colors is enough to give every object a unique color even
# on Hard difficulty (max 7 objects on screen at once).
COLOR_PALETTE = {
    "RED":    (0, 0, 255),
    "GREEN":  (0, 200, 0),
    "BLUE":   (255, 120, 0),
    "YELLOW": (0, 220, 220),
    "PURPLE": (200, 0, 200),
    "ORANGE": (0, 140, 255),
    "CYAN":   (255, 255, 0),
    "PINK":   (200, 150, 255),
}

# Difficulty settings: object count, object radius, extra "forgiveness"
# pixels added to the radius for hit detection, and points per correct hit.
DIFFICULTY_SETTINGS = {
    "EASY":   {"count": 3, "radius": 70, "hit_tolerance": 35, "points": 50},
    "MEDIUM": {"count": 5, "radius": 50, "hit_tolerance": 20, "points": 75},
    "HARD":   {"count": 7, "radius": 35, "hit_tolerance": 8,  "points": 100},
}

# ---------------------------------------------------------------
# GAME STATES
# ---------------------------------------------------------------
STATE_MENU = "MENU"                     # choose difficulty
STATE_START_SCREEN = "START_SCREEN"     # camera shown, press SPACE
STATE_PLAYING = "PLAYING"               # active session
STATE_COMPLETE = "SESSION_COMPLETE"     # results screen


# =====================================================================
# OBJECT / ROUND GENERATION
# =====================================================================
def generate_objects(difficulty, width, height):
    """
    Creates a fresh set of colored objects laid out across the usable
    part of the screen (leaving room for the HUD at the top and the
    instruction text near the bottom), and picks one of them at random
    to be the current target.

    Returns:
        objects       -> list of dicts: {"pos": (x, y), "color_name": str,
                                          "color_bgr": (b, g, r)}
        target_index  -> index into 'objects' of the correct object
        radius        -> base radius (in pixels) used to draw each object
        hit_tolerance -> extra pixels added to radius for hit detection
    """
    cfg = DIFFICULTY_SETTINGS[difficulty]
    count = cfg["count"]
    radius = cfg["radius"]
    hit_tolerance = cfg["hit_tolerance"]

    # Pick 'count' unique colors from the palette so every object on
    # screen is visually distinct (avoids ambiguous targets).
    color_names = random.sample(list(COLOR_PALETTE.keys()), count)

    # Lay objects out in a single row across the usable width, with a
    # small alternating vertical offset so the patient has to reach up
    # and down a little rather than moving in a perfectly flat line.
    left_margin = radius + 60
    right_margin = width - radius - 60
    usable_width = max(1, right_margin - left_margin)
    mid_y = height // 2

    objects = []
    for i in range(count):
        if count == 1:
            x = width // 2
        else:
            x = int(left_margin + i * (usable_width / (count - 1)))
        y_offset = 60 if i % 2 == 0 else -60
        y = mid_y + y_offset
        objects.append({
            "pos": (x, y),
            "color_name": color_names[i],
            "color_bgr": COLOR_PALETTE[color_names[i]],
        })

    target_index = random.randrange(count)
    return objects, target_index, radius, hit_tolerance


def find_touched_object(wrist_pos, objects, radius, hit_tolerance):
    """
    Returns the index of the object the wrist is currently overlapping,
    or None if the wrist is not touching any object.
    """
    hit_radius = radius + hit_tolerance
    for idx, obj in enumerate(objects):
        ox, oy = obj["pos"]
        dist = math.hypot(wrist_pos[0] - ox, wrist_pos[1] - oy)
        if dist <= hit_radius:
            return idx
    return None


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

    draw_text_center(frame, "OBJECT SORTING", 100, 1.4, (0, 255, 255), 3)
    draw_text_center(frame, "MOVENTRA - Gamified Rehabilitation", 140, 0.7, (200, 200, 200), 1)
    draw_text_center(frame, "Select Difficulty", 220, 1.0, COLOR_BG_TEXT, 2)
    draw_text_center(frame, "1 - Easy", 280, 0.9, (0, 255, 0), 2)
    draw_text_center(frame, "2 - Medium", 320, 0.9, (0, 165, 255), 2)
    draw_text_center(frame, "3 - Hard", 360, 0.9, (0, 0, 255), 2)
    draw_text_center(frame, "Press Q to Quit", frame.shape[0] - 40, 0.7, (180, 180, 180), 1)


def draw_start_screen(frame, difficulty):
    """Camera feed + instructions, waiting for SPACE to start."""
    draw_text_center(frame, "OBJECT SORTING", 60, 1.1, (0, 255, 255), 2)
    draw_text_center(frame, f"Difficulty: {difficulty}", 100, 0.8, COLOR_BG_TEXT, 2)
    draw_text_center(frame, "Reach the object that matches the target color", 140, 0.7, COLOR_BG_TEXT, 1)
    draw_text_center(frame, "Use your LEFT hand", frame.shape[0] - 100, 0.8, (0, 255, 0), 2)
    draw_text_center(frame, "Press SPACE to Start", frame.shape[0] - 60, 0.9, (0, 255, 0), 2)
    draw_text_center(frame, "Press Q to Quit", frame.shape[0] - 25, 0.6, (180, 180, 180), 1)


def draw_objects(frame, objects, target_index, radius, feedback_active, feedback_type, touched_index):
    """
    Draws every object on screen. The object currently under a
    success/error flash is highlighted with a colored ring so the
    patient gets clear visual feedback.
    """
    for idx, obj in enumerate(objects):
        pos = obj["pos"]
        color = obj["color_bgr"]
        cv2.circle(frame, pos, radius, color, -1)
        cv2.circle(frame, pos, radius, (255, 255, 255), 2)

        # Highlight ring for feedback (green = correct, red = wrong)
        if feedback_active and idx == touched_index:
            ring_color = COLOR_CORRECT if feedback_type == "correct" else COLOR_WRONG
            cv2.circle(frame, pos, radius + 12, ring_color, 4)


def draw_target_instruction(frame, target_color_name):
    """Shows the current target instruction, e.g. 'Find the RED Object'."""
    h, w = frame.shape[:2]
    text = f"Find the {target_color_name} Object"
    color = COLOR_PALETTE[target_color_name]
    (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
    box_y = h - 80
    cv2.rectangle(frame, (w // 2 - text_w // 2 - 20, box_y - text_h - 15),
                  (w // 2 + text_w // 2 + 20, box_y + 15), (20, 20, 20), -1)
    cv2.putText(frame, text, (w // 2 - text_w // 2, box_y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)


def draw_feedback_banner(frame, feedback_type):
    """Big centered SUCCESS / TRY AGAIN banner shown briefly after a hit."""
    if feedback_type == "correct":
        draw_text_center(frame, "CORRECT!", 100, 1.3, COLOR_CORRECT, 3)
    else:
        draw_text_center(frame, "WRONG OBJECT - TRY AGAIN", 100, 1.0, COLOR_WRONG, 3)


def draw_hud(frame, score, accuracy, time_left, targets_completed):
    """Draws the heads-up display: score, accuracy, timer, targets completed."""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 70), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    cv2.putText(frame, f"Score: {score}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"Accuracy: {accuracy:.0f}%", (20, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

    targets_text = f"Targets Completed: {targets_completed}"
    (tw, _), _ = cv2.getTextSize(targets_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.putText(frame, targets_text, (w // 2 - tw // 2, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_BG_TEXT, 2)

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
        f"Accuracy: {results['accuracy']:.1f}%",
        f"Correct Selections: {results['correct']}",
        f"Wrong Selections: {results['wrong']}",
        f"Targets Completed: {results['targets_completed']}",
        f"Time Taken: {results['time_taken']:.1f} sec",
        f"Distance Travelled: {results['distance']:.0f} px",
    ]
    y = 140
    for line in lines:
        draw_text_center(frame, line, y, 0.75, COLOR_BG_TEXT, 2)
        y += 38

    draw_text_center(frame, "Press R to PLAY AGAIN   |   Press Q to QUIT", frame.shape[0] - 40, 0.7, (0, 255, 0), 2)


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

    # Feedback flash state (shown briefly after touching an object)
    feedback_active = False
    feedback_type = None      # "correct" or "wrong"
    feedback_until = 0.0
    feedback_touched_index = None
    last_touched_index = None  # used to debounce repeated hits while lingering on an object

    session_start_time = 0.0
    frame_timestamp_ms = 0

    results = {}

    print("MOVENTRA - Object Sorting started. Press Q at any time to quit.")

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
        # horizontal flip is applied. If the frame were flipped first,
        # the model would silently swap the left/right labels and the
        # game would end up tracking the RIGHT hand instead.
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
                cv2.circle(frame, wrist_pos, 12, COLOR_WRIST_NEUTRAL, -1)
            else:
                draw_wrist_not_detected_warning(frame)

        # =====================================================
        # STATE: PLAYING - active rehabilitation session
        # =====================================================
        elif state == STATE_PLAYING:
            elapsed = time.time() - session_start_time
            time_left = max(0, SESSION_SECONDS - elapsed)

            # Check whether the current feedback flash has expired
            if feedback_active and time.time() >= feedback_until:
                if feedback_type == "correct":
                    # Start a brand-new round: new objects, new target
                    objects, target_index, obj_radius, hit_tolerance = generate_objects(difficulty, w, h)
                    last_touched_index = None
                feedback_active = False
                feedback_type = None
                feedback_touched_index = None

            # Draw the objects for the current round
            draw_objects(frame, objects, target_index, obj_radius,
                         feedback_active, feedback_type, feedback_touched_index)
            draw_target_instruction(frame, objects[target_index]["color_name"])

            touched_index = None
            if wrist_detected:
                # Track distance travelled (sum of movement between frames)
                if prev_wrist_pos is not None:
                    step_dist = math.hypot(wrist_pos[0] - prev_wrist_pos[0], wrist_pos[1] - prev_wrist_pos[1])
                    distance_travelled += step_dist
                prev_wrist_pos = wrist_pos

                # Only evaluate new touches while no feedback is showing,
                # so a single reach only registers as one hit.
                if not feedback_active:
                    touched_index = find_touched_object(wrist_pos, objects, obj_radius, hit_tolerance)

                    # Only fire a hit event the moment the wrist ENTERS an
                    # object (debounced), not on every frame it lingers there.
                    if touched_index is not None and touched_index != last_touched_index:
                        if touched_index == target_index:
                            correct_count += 1
                            targets_completed += 1
                            score += DIFFICULTY_SETTINGS[difficulty]["points"]
                            feedback_type = "correct"
                        else:
                            wrong_count += 1
                            feedback_type = "wrong"

                        feedback_active = True
                        feedback_until = time.time() + FEEDBACK_SECONDS
                        feedback_touched_index = touched_index

                    last_touched_index = touched_index

                # Draw the wrist marker on top of everything else
                marker_color = COLOR_WRIST_NEUTRAL
                if feedback_active:
                    marker_color = COLOR_CORRECT if feedback_type == "correct" else COLOR_WRONG
                cv2.circle(frame, wrist_pos, 15, marker_color, -1)
                cv2.circle(frame, wrist_pos, 15, (255, 255, 255), 2)
            else:
                draw_wrist_not_detected_warning(frame)
                prev_wrist_pos = None  # avoid a big "jump" once detection resumes
                last_touched_index = None

            if feedback_active:
                draw_feedback_banner(frame, feedback_type)

            # Compute live accuracy safely (avoid divide-by-zero)
            total_attempts = correct_count + wrong_count
            accuracy = (correct_count / total_attempts * 100) if total_attempts > 0 else 0.0

            draw_hud(frame, score, accuracy, time_left, targets_completed)

            # End the session when time runs out
            if time_left <= 0:
                results = {
                    "difficulty": difficulty,
                    "score": score,
                    "accuracy": accuracy,
                    "correct": correct_count,
                    "wrong": wrong_count,
                    "targets_completed": targets_completed,
                    "time_taken": elapsed,
                    "distance": distance_travelled,
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
        cv2.imshow(WINDOW_NAME, display_frame)

        # -----------------------------------------------------------
        # Handle keyboard input (single waitKey call per frame)
        # -----------------------------------------------------------
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        if state == STATE_MENU:
            if key in (ord('1'), ord('2'), ord('3')):
                difficulty = {ord('1'): "EASY", ord('2'): "MEDIUM", ord('3'): "HARD"}[key]
                state = STATE_START_SCREEN

        elif state == STATE_START_SCREEN:
            if key == ord(' '):
                # Reset all session metrics right before starting
                objects, target_index, obj_radius, hit_tolerance = generate_objects(difficulty, w, h)
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
    print("MOVENTRA - Object Sorting closed.")


if __name__ == "__main__":
    main()