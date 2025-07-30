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



# Cooldown control for jump
jump_ready = True
jump_last_time = 0
JUMP_COOLDOWN = 0.8  # Seconds between jumps
JUMP_TAP_DURATION = 0.5  # How long the jump "button" is held

# Funktionen zum Selberbasteln
def is_left_really_left(keyp1, keyp2):
    dx = keyp2[0] - keyp1[0]
    dy = abs(keyp2[1] - keyp1[1])
    return dx > 50 and dy < 40  # Ihr könnt diese Werte verändern

def is_right_really_right(keyp1, keyp2):
    dx = keyp1[0] - keyp2[0]
    dy = abs(keyp2[1] - keyp1[1])
    return dx > 50 and dy < 40  # Ihr könnt diese Werte verändern

def is_up_really_up(keyp1, wrist):
    return wrist[1] < keyp1[1] - 40

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
    # Check if the webcam is open
    if pose_result is None:
        continue

    # Process pose detection results with keypoints  
    results, frame = pose_result
    if results and results[0].keypoints is not None and len(results[0].keypoints) > 0:
        keypoints_data = results[0].keypoints[0].data
        if keypoints_data is not None:
            kps = keypoints_data[0].cpu().numpy()
            nose = kps[0][:2]
            left_eye = kps[1][:2]
            right_eye = kps[2][:2]
            left_ear = kps[3][:2]
            right_ear = kps[4][:2]
            left_shoulder = kps[5][:2]
            right_shoulder = kps[6][:2]
            left_elbow = kps[7][:2]
            right_elbow = kps[8][:2]
            left_wrist = kps[9][:2]
            right_wrist = kps[10][:2]
            left_knee = kps[13][:2]
            right_knee = kps[14][:2]
            left_hip = kps[11][:2]
            right_hip = kps[12][:2]
            left_knee = kps[13][:2]
            right_knee = kps[14][:2]
            left_ankle = kps[15][:2]
            right_ankle = kps[16][:2]



            # Hier könnt ihr eure eigenen Parameter übergeben!
            
            
            move_left = is_left_really_left(left_elbow, left_wrist)
            
            
            move_right = is_right_really_right(right_elbow, right_wrist)
            
            
            jump_up = is_up_really_up(right_elbow, right_wrist) or is_up_really_up(left_elbow, left_wrist)
            


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

         
            # --- JUMP TAP ---
            current_time = time.time()
            if jump_up and jump_ready:
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

  

# Cleanup
running = False
cap.release()
cv2.destroyAllWindows()
pyboy.stop()
