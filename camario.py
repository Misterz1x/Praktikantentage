import cv2
import threading
import time
from pyboy import PyBoy
from pyboy.utils import WindowEvent
from ultralytics import YOLO

# Load YOLOv8 pose model
model = YOLO("yolov8n-pose.pt")

# Start emulator
pyboy = PyBoy('Super Mario Bros. Deluxe (USA, Europe) (Rev 1).gbc', window="SDL2")
pyboy.set_emulation_speed(1.0)

# Track button state
buttons_state = {'left': False, 'right': False, 'jump': False}
pose_result = None
running = True

# Open webcam
cap = cv2.VideoCapture(0)

# Cooldown control for jump
jump_ready = True
jump_last_time = 0
JUMP_COOLDOWN = 1  # Seconds between jumps
JUMP_TAP_DURATION = 0.5  # How long the jump "button" is held

def is_arm_horizontal(elbow, wrist):
    dx = wrist[0] - elbow[0]
    dy = abs(wrist[1] - elbow[1])
    return abs(dx) > 50 and dy < 50

def is_arm_up(elbow, wrist):
    return wrist[1] < elbow[1] - 60

#def are_you_crouched(knee, hip):
#    return knee[1] > hip[1] + 50

# Background pose detection thread
def pose_thread():
    global pose_result, running
    while running:
        ret, frame = cap.read()
        if not ret:
            continue
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model(rgb_frame, verbose=False)
        pose_result = (results, frame)
        time.sleep(0.01)

threading.Thread(target=pose_thread, daemon=True).start()

# Main emulator loop
while pyboy.tick():
    if pose_result is None:
        continue

    results, frame = pose_result
    if results and results[0].keypoints is not None and len(results[0].keypoints) > 0:
        keypoints_data = results[0].keypoints[0].data
        if keypoints_data is not None:
            kps = keypoints_data[0].cpu().numpy()
            left_elbow = kps[7][:2]
            right_elbow = kps[8][:2]
            left_wrist = kps[9][:2]
            right_wrist = kps[10][:2]
            left_knee = kps[13][:2]
            right_knee = kps[14][:2]
            left_hip = kps[11][:2]
            right_hip = kps[12][:2]

            move_left = is_arm_horizontal(left_elbow, left_wrist)
            move_right = is_arm_horizontal(right_elbow, right_wrist)
            arm_up = is_arm_up(right_elbow, right_wrist) or is_arm_up(left_elbow, left_wrist)
            #crouch = are_you_crouched(left_knee, left_hip) or are_you_crouched(right_knee, right_hip)

            # --- LEFT ---
            if move_left and not buttons_state['left']:
                pyboy.send_input(WindowEvent.PRESS_ARROW_LEFT)
                buttons_state['left'] = True
            elif not move_left and buttons_state['left']:
                pyboy.send_input(WindowEvent.RELEASE_ARROW_LEFT)
                buttons_state['left'] = False

            # --- RIGHT ---
            if move_right and not buttons_state['right']:
                pyboy.send_input(WindowEvent.PRESS_ARROW_RIGHT)
                buttons_state['right'] = True
            elif not move_right and buttons_state['right']:
                pyboy.send_input(WindowEvent.RELEASE_ARROW_RIGHT)
                buttons_state['right'] = False

            # --- CROUCH ---
            #if crouch and not buttons_state['down']:
            #    pyboy.send_input(WindowEvent.PRESS_ARROW_DOWN)
            #    buttons_state['down'] = True
            #elif not crouch and buttons_state['down']:
            #    pyboy.send_input(WindowEvent.RELEASE_ARROW_DOWN)
            #    buttons_state['down'] = False

            # --- JUMP TAP ---
            current_time = time.time()
            if arm_up and jump_ready:
                pyboy.send_input(WindowEvent.PRESS_BUTTON_A)
                buttons_state['jump'] = True
                jump_ready = False
                jump_last_time = current_time

                # Schedule auto-release after tap duration
                def release_jump():
                    time.sleep(JUMP_TAP_DURATION)
                    pyboy.send_input(WindowEvent.RELEASE_BUTTON_A)
                    buttons_state['jump'] = False
                threading.Thread(target=release_jump, daemon=True).start()

            # Reset cooldown after enough time has passed
            if not jump_ready and (current_time - jump_last_time) > JUMP_COOLDOWN:
                jump_ready = True

    # Display webcam + keypoints
    annotated = results[0].plot()
    cv2.imshow("Pose Input", annotated)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
running = False
cap.release()
cv2.destroyAllWindows()
pyboy.stop()
