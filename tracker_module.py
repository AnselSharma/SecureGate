import supervision as sv

# Create tracker
tracker = sv.ByteTrack()

def track_objects(detections):

    tracked_detections = tracker.update_with_detections(
        detections
    )

    return tracked_detections