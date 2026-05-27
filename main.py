import cv2
import supervision as sv

from detector import detect_people
from tracker_module import track_objects

from gate_logic import (
    process_crossing,
    check_tailgating,
    entered_ids,
    exited_ids
)

from config import (
    LINE_Y,
    TAILGATING_THRESHOLD
)

# Open webcam
cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # Detect humans
    results = detect_people(frame)

    # Convert detections
    detections = sv.Detections.from_ultralytics(results)

    # Keep only persons
    detections = detections[detections.class_id == 0]

    # Track detections
    detections = track_objects(detections)

    # Draw virtual gate line
    cv2.line(
        frame,
        (0, LINE_Y),
        (frame.shape[1], LINE_Y),
        (0, 0, 255),
        3
    )

    # Loop through detections
    for i in range(len(detections)):

        tracker_id = detections.tracker_id[i]

        x1, y1, x2, y2 = map(
            int,
            detections.xyxy[i]
        )

        center_y = (y1 + y2) // 2

        # Process entry/exit
        process_crossing(
            tracker_id,
            center_y
        )

        # Draw rectangle
        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        # Draw ID
        cv2.putText(
            frame,
            f"ID {tracker_id}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    # Check tailgating
    tailgating_count = check_tailgating()

    if tailgating_count >= TAILGATING_THRESHOLD:

        cv2.putText(
            frame,
            "TAILGATING ALERT!",
            (150, 150),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 0, 255),
            4
        )

    # Entry count
    cv2.putText(
        frame,
        f"Entries: {len(entered_ids)}",
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        3
    )

    # Exit count
    cv2.putText(
        frame,
        f"Exits: {len(exited_ids)}",
        (20, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 0, 0),
        3
    )

    # Show frame
    cv2.imshow("SecureGate AI", frame)

    # Quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()