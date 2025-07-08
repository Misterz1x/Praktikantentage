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

def is_arm_horizontal(shoulder, wrist):
    dx = wrist[0] - shoulder[0]
    dy = abs(wrist[1] - shoulder[1])
    return abs(dx) > 50 and dy < 50

def is_arm_up(shoulder, wrist):
    return wrist[1] < shoulder[1] - 50

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
        time.sleep(0.01)  # small delay to reduce CPU load

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
            left_shoulder = kps[5][:2]
            right_shoulder = kps[6][:2]
            left_wrist = kps[9][:2]
            right_wrist = kps[10][:2]

            move_left = is_arm_horizontal(left_shoulder, left_wrist)
            move_right = is_arm_horizontal(right_shoulder, right_wrist)
            jump = is_arm_up(right_shoulder, right_wrist) or is_arm_up(left_shoulder, left_wrist)

            if move_left and not buttons_state['left']:
                pyboy.send_input(WindowEvent.PRESS_ARROW_LEFT)
                buttons_state['left'] = True
            elif not move_left and buttons_state['left']:
                pyboy.send_input(WindowEvent.RELEASE_ARROW_LEFT)
                buttons_state['left'] = False

            if move_right and not buttons_state['right']:
                pyboy.send_input(WindowEvent.PRESS_ARROW_RIGHT)
                buttons_state['right'] = True
            elif not move_right and buttons_state['right']:
                pyboy.send_input(WindowEvent.RELEASE_ARROW_RIGHT)
                buttons_state['right'] = False

            if jump and not buttons_state['jump']:
                pyboy.send_input(WindowEvent.PRESS_BUTTON_A)
                buttons_state['jump'] = True
            elif not jump and buttons_state['jump']:
                pyboy.send_input(WindowEvent.RELEASE_BUTTON_A)
                buttons_state['jump'] = False

    annotated = results[0].plot()
    cv2.imshow("Pose Input", annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
running = False
cap.release()
cv2.destroyAllWindows()
pyboy.stop()
