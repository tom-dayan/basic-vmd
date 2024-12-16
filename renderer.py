import cv2
from datetime import datetime
import multiprocessing as mp
from typing import Tuple, Optional, Any

# Configurable variables
BB_COLOR = (0, 255, 0)  # Bounding Box color (Green)
CLOCK_COLOR = (255, 0, 0)  # Clock color (Blue)
CLOCK_POSITION = (10, 30)  # Clock position (x, y)
BLUR_KERNEL = (15, 15)     # Kernel size for Gaussian blur

def apply_blur(frame, detections):
    """
    Applies Gaussian blur to the regions defined by bounding boxes.

    Args:
        frame (ndarray): The frame to process.
        detections (list of tuples): List of bounding boxes (x, y, w, h).

    Returns:
        ndarray: The processed frame with blurred regions.
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
    Receives frames and detections, annotates, applies blurring, and displays the video.

    Args:
        renderer_queue (mp.Queue): Queue to receive frames and detections from the detector.
        stop_signal (Any): Multiprocessing event used to signal stopping the renderer.
        enable_blur (bool): Whether to enable blurring of detections.
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
