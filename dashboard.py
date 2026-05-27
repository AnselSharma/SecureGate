import streamlit as st
import cv2
import supervision as sv
import time
import pygame

from datetime import datetime
from ultralytics import YOLO

# -----------------------------------
# PAGE CONFIG
# -----------------------------------

st.set_page_config(
    page_title="SecureGate AI",
    layout="wide"
)

st.title("🔐 SecureGate AI Surveillance Dashboard")

# -----------------------------------
# INITIALIZE ALARM
# -----------------------------------

pygame.mixer.init()

alarm_sound = pygame.mixer.Sound(
    "alarm.wav"
)

# -----------------------------------
# METRICS SECTION
# -----------------------------------

col1, col2, col3 = st.columns(3)

entry_metric = col1.empty()
exit_metric = col2.empty()
alert_metric = col3.empty()

# Alert placeholder
alert_placeholder = st.empty()

# -----------------------------------
# CENTERED VIDEO PLACEHOLDER
# -----------------------------------

col_left, col_center, col_right = st.columns([1, 2, 1])

with col_center:
    frame_placeholder = st.empty()

# -----------------------------------
# LOAD YOLO MODEL
# -----------------------------------

model = YOLO("yolov8n.pt")

# -----------------------------------
# OPEN WEBCAM
# -----------------------------------

cap = cv2.VideoCapture(0)

# -----------------------------------
# TRACKER
# -----------------------------------

tracker = sv.ByteTrack()

# -----------------------------------
# CONFIG
# -----------------------------------

LINE_Y = 300

TAILGATING_THRESHOLD = 2
TIME_WINDOW = 3

# -----------------------------------
# MEMORY
# -----------------------------------

previous_positions = {}

entered_ids = set()
exited_ids = set()

crossing_times = []

alarm_playing = False

# -----------------------------------
# LOGGING FUNCTION
# -----------------------------------

def log_event(message):

    timestamp = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    log_message = (
        f"[{timestamp}] {message}\n"
    )

    with open(
        "logs/events.txt",
        "a"
    ) as file:

        file.write(log_message)

# -----------------------------------
# SCREENSHOT FUNCTION
# -----------------------------------

def save_screenshot(frame):

    timestamp = datetime.now().strftime(
        "%Y-%m-%d_%H-%M-%S"
    )

    filename = (
        f"screenshots/alert_{timestamp}.jpg"
    )

    cv2.imwrite(filename, frame)

# -----------------------------------
# MAIN LOOP
# -----------------------------------

while True:

    # Read webcam frame
    ret, frame = cap.read()

    if not ret:
        st.error("Failed to access webcam")
        break

    # -----------------------------------
    # YOLO DETECTION
    # -----------------------------------

    results = model(frame)[0]

    # Convert detections
    detections = sv.Detections.from_ultralytics(
        results
    )

    # Keep only persons
    detections = detections[
        detections.class_id == 0
    ]

    # -----------------------------------
    # TRACKING
    # -----------------------------------

    detections = tracker.update_with_detections(
        detections
    )

    # -----------------------------------
    # DRAW GATE LINE
    # -----------------------------------

    cv2.line(
        frame,
        (0, LINE_Y),
        (frame.shape[1], LINE_Y),
        (0, 0, 255),
        3
    )

    # -----------------------------------
    # PROCESS DETECTIONS
    # -----------------------------------

    for i in range(len(detections)):

        # Tracker ID
        tracker_id = detections.tracker_id[i]

        # Bounding box
        x1, y1, x2, y2 = map(
            int,
            detections.xyxy[i]
        )

        # Center point
        center_y = (y1 + y2) // 2

        # Previous position
        previous_y = previous_positions.get(
            tracker_id,
            center_y
        )

        # -----------------------------------
        # ENTRY DETECTION
        # -----------------------------------

        if previous_y < LINE_Y and center_y > LINE_Y:

            if tracker_id not in entered_ids:

                entered_ids.add(tracker_id)

                crossing_times.append(
                    time.time()
                )

        # -----------------------------------
        # EXIT DETECTION
        # -----------------------------------

        elif previous_y > LINE_Y and center_y < LINE_Y:

            if tracker_id not in exited_ids:

                exited_ids.add(tracker_id)

        # Update memory
        previous_positions[tracker_id] = center_y

        # -----------------------------------
        # DRAW PERSON BOX
        # -----------------------------------

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        # -----------------------------------
        # DRAW TRACKER ID
        # -----------------------------------

        cv2.putText(
            frame,
            f"ID {tracker_id}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    # -----------------------------------
    # REMOVE OLD EVENTS
    # -----------------------------------

    current_time = time.time()

    crossing_times = [
        t for t in crossing_times
        if current_time - t <= TIME_WINDOW
    ]

    # -----------------------------------
    # TAILGATING ALERT
    # -----------------------------------

    alert_count = 0

    if len(crossing_times) >= TAILGATING_THRESHOLD:

        alert_count = 1

        cv2.putText(
            frame,
            "TAILGATING ALERT!",
            (120, 150),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 0, 255),
            4
        )

        alert_placeholder.error(
            "⚠ Tailgating Detected!"
        )

        # Play alarm once
        if not alarm_playing:

            alarm_sound.play(-1)

            alarm_playing = True

            log_event(
                "Tailgating detected"
            )

            save_screenshot(frame)

    else:

        alert_placeholder.success(
            "✅ System Secure"
        )

        # Stop alarm
        if alarm_playing:

            alarm_sound.stop()

            alarm_playing = False

    # -----------------------------------
    # UPDATE METRICS
    # -----------------------------------

    entry_metric.metric(
        "Total Entries",
        len(entered_ids)
    )

    exit_metric.metric(
        "Total Exits",
        len(exited_ids)
    )

    alert_metric.metric(
        "Security Alerts",
        alert_count
    )

    # -----------------------------------
    # CONVERT COLOR FORMAT
    # -----------------------------------

    frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    # -----------------------------------
    # DISPLAY VIDEO
    # -----------------------------------

    frame_placeholder.image(
        frame,
        channels="RGB",
        width=700
    )

# -----------------------------------
# CLEANUP
# -----------------------------------

cap.release()