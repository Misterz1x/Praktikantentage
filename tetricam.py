import cv2
import threading
import time
from pyboy import PyBoy
from pyboy.utils import WindowEvent
from ultralytics import YOLO

# Load YOLOv8 pose model
model = YOLO("yolov8n-pose.pt")

# Start emulator
pyboy = PyBoy('tetris.gb', window="SDL2")
pyboy.set_emulation_speed(1.0)

# Track button state
buttons_state = {'left': False, 'right': False, 'down': False, 'turn': False, 'start': False}
pose_result = None
running = True

# Open webcam
cap = cv2.VideoCapture(0)

# Cooldown control for jump
#jump_ready = True
#jump_last_time = 0
#JUMP_COOLDOWN = 0.8  # Seconds between jumps
#JUMP_TAP_DURATION = 0.5  # How long the jump "button" is held
turn_ready = True
turn_last_time = 0
TURN_TAP_DURATION = 0.5  # How long the turn "button" is held
TURN_COOLDOWN = 0.5  # Seconds between turns

def is_left_hand_left(elbow, wrist):
    dx = wrist[0] - elbow[0]
    dy = abs(wrist[1] - elbow[1])
    return dx > 50 and dy < 50  # You can tune these values

def is_right_hand_right(elbow, wrist):
    dx = elbow[0] - wrist[0]
    dy = abs(wrist[1] - elbow[1])
    return dx > 50 and dy < 50  # Same tuning here

def is_arm_up(elbow, wrist):
    return wrist[1] < elbow[1] - 45

def is_arm_down(elbow, wrist):
    return wrist[1] > elbow[1] + 60


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

            move_left = is_left_hand_left(left_elbow, left_wrist)
            move_right = is_right_hand_right(right_elbow, right_wrist)
            right_arm_up = is_arm_up(right_elbow, right_wrist)
            left_arm_up = is_arm_up(left_elbow, left_wrist)
            arm_down = is_arm_down(right_elbow, right_wrist) or is_arm_down(left_elbow, left_wrist)
            start = is_arm_up(left_elbow, left_wrist) and is_arm_up(right_elbow, right_wrist)
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

            # --- DOWN ---
            if arm_down and not buttons_state['down']:
                pyboy.send_input(WindowEvent.PRESS_ARROW_DOWN)
                buttons_state['down'] = True
            elif not arm_down and buttons_state['down']:
                pyboy.send_input(WindowEvent.RELEASE_ARROW_DOWN)
                buttons_state['down'] = False

            # --- START ---
            if start and not buttons_state['start']:
                pyboy.send_input(WindowEvent.PRESS_BUTTON_START)
                buttons_state['start'] = True
            elif not start and buttons_state['start']:
                pyboy.send_input(WindowEvent.RELEASE_BUTTON_START)
                buttons_state['start'] = False
                
            # --- CROUCH ---
            #if crouch and not buttons_state['down']:
            #    pyboy.send_input(WindowEvent.PRESS_ARROW_DOWN)
            #    buttons_state['down'] = True
            #elif not crouch and buttons_state['down']:
            #    pyboy.send_input(WindowEvent.RELEASE_ARROW_DOWN)
            #    buttons_state['down'] = False

            # --- JUMP TAP ---
            current_time = time.time()
            if right_arm_up and turn_ready and not buttons_state['turn']:
                pyboy.send_input(WindowEvent.PRESS_BUTTON_A)
                buttons_state['turn'] = True

                # Schedule auto-release after tap duration
                def release_turn():
                    time.sleep(TURN_TAP_DURATION)
                    pyboy.send_input(WindowEvent.RELEASE_BUTTON_A)
                    buttons_state['turn'] = False
                threading.Thread(target=release_turn, daemon=True).start()

            if left_arm_up and turn_ready and not buttons_state['turn']:
                pyboy.send_input(WindowEvent.PRESS_BUTTON_B)
                buttons_state['turn'] = True

                # Schedule auto-release after tap duration
                def release_turn():
                    time.sleep(TURN_TAP_DURATION)
                    pyboy.send_input(WindowEvent.RELEASE_BUTTON_B)
                    buttons_state['turn'] = False
                threading.Thread(target=release_turn, daemon=True).start()

            # Reset cooldown after enough time has passed
            if not turn_ready and (current_time - turn_last_time) > TURN_COOLDOWN:
                turn_ready = True

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
