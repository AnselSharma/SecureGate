import time
from config import LINE_Y, TIME_WINDOW

# Memory storage
previous_positions = {}

entered_ids = set()
exited_ids = set()

crossing_times = []

def process_crossing(tracker_id, center_y):

    global previous_positions
    global entered_ids
    global exited_ids
    global crossing_times

    previous_y = previous_positions.get(
        tracker_id,
        center_y
    )

    # ENTRY
    if previous_y < LINE_Y and center_y > LINE_Y:

        if tracker_id not in entered_ids:

            entered_ids.add(tracker_id)

            crossing_times.append(time.time())

    # EXIT
    elif previous_y > LINE_Y and center_y < LINE_Y:

        if tracker_id not in exited_ids:

            exited_ids.add(tracker_id)

    # Update memory
    previous_positions[tracker_id] = center_y

def check_tailgating():

    global crossing_times

    current_time = time.time()

    crossing_times = [
        t for t in crossing_times
        if current_time - t <= TIME_WINDOW
    ]

    return len(crossing_times)