import cv2
from datetime import datetime
import multiprocessing as mp
from typing import Tuple, Optional, Any, List

"""
Handles rendering of video frames with optional annotation and blurring.

Features:
- Draws bounding boxes around detected regions.
- Overlays a timestamp on each frame.
- Applies Gaussian blurring to detected regions if enabled.
"""

# Configurable variables
BB_COLOR = (0, 255, 0)  # Bounding Box color (Green)
CLOCK_COLOR = (255, 0, 0)  # Clock color (Blue)
CLOCK_POSITION = (10, 30)  # Clock position (x, y)
BLUR_KERNEL = (15, 15)     # Kernel size for Gaussian blur

def apply_blur(frame: cv2.Mat, detections: List[cv2.rectangle]):
    """
    Blurs regions inside bounding boxes.

    Args:
        frame: Frame to process.
        detections: List of bounding boxes (x, y, w, h).

    Returns:
        Frame with blurred regions.
    """
    for (x, y, w, h) in detections:
        # Extract the region of interest (ROI)
        roi = frame[y:y+h, x:x+w]
        # Apply Gaussian blur to the ROI
        blurred_roi = cv2.GaussianBlur(roi, BLUR_KERNEL, 0)
        # Replace the original ROI with the blurred ROI
        frame[y:y+h, x:x+w] = blurred_roi
    return frame

def renderer(renderer_queue: mp.Queue, stop_signal: Any, enable_blur: bool) -> None:
    """
    Processes and renders video frames with optional annotations and blur.

    Args:
        renderer_queue: Queue receiving frames and detections.
        stop_signal: Event signaling termination.
        enable_blur: Enables or disables blurring of detected regions.
    """
    while not stop_signal.is_set():
        try:
            data: Optional[Tuple] = renderer_queue.get(timeout=0.5)
            if data is None:
                break

            frame, detections = data

            # Apply blur if enabled
            if enable_blur:
                frame = apply_blur(frame, detections)

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
