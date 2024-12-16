import cv2
from datetime import datetime
import multiprocessing as mp
from typing import Tuple, Optional, Any

# Configurable variables
BB_COLOR = (0, 255, 0)  # Bounding Box color (Green)
CLOCK_COLOR = (255, 0, 0)  # Clock color (Blue)
CLOCK_POSITION = (10, 30)  # Clock position (x, y)

def renderer(renderer_queue: mp.Queue, stop_signal: Any) -> None:
    """
    Receives frames and detections, annotates, and displays the video.

    Args:
        renderer_queue (mp.Queue): Queue to receive frames and detections from the detector.
        stop_signal (Any): Multiprocessing event used to signal stopping the renderer.
    """
    while not stop_signal.is_set():
        try:
            data: Optional[Tuple] = renderer_queue.get(timeout=0.5)
            if data is None:
                break

            frame, detections = data

            # Draw detections on the frame
            for (x, y, w, h) in detections:
                cv2.rectangle(frame, (x, y), (x + w, y + h), BB_COLOR, 2)

            # Add timestamp
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(
                frame,
                timestamp,
                CLOCK_POSITION,
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                CLOCK_COLOR,
                2,
            )

            # Display the frame
            cv2.imshow("Video Stream", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                stop_signal.set()  # Signal to stop other processes
                break
        except mp.queues.Empty:
            if stop_signal.is_set():
                break

    cv2.destroyAllWindows()