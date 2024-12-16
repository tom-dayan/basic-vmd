import cv2
from datetime import datetime
import multiprocessing as mp
from typing import Tuple, Optional

def renderer(renderer_queue: mp.Queue) -> None:
    """
    Receives frames and detections, annotates and displays the video.

    Args:
        renderer_queue (mp.Queue): Queue to receive frames and detections from the detector.
    """
    while True:
        data: Optional[Tuple] = renderer_queue.get()
        if data is None:
            break

        frame, detections = data

        # Draw detections on the frame
        for (x, y, w, h) in detections:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Add timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, timestamp, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # Display the frame
        cv2.imshow("Video Stream", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
