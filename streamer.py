import cv2
import multiprocessing as mp
import time
from typing import Any

"""
Streams frames from a video file.

Reads frames and sends them to the Detector via a multiprocessing queue.
Supports graceful termination using a stop signal.
"""

def streamer(video_path: str, detector_queue: mp.Queue, stop_signal: Any) -> None:
    """
    Streams video frames to the detector queue.

    Args:
        video_path: Path to the video file.
        detector_queue: Queue for sending frames to the detector.
        stop_signal: Event signaling termination.
    """
    cap = cv2.VideoCapture(video_path)
    while not stop_signal.is_set():
        ret, frame = cap.read()
        if not ret:  # End of video
            break
        try:
            detector_queue.put(frame, timeout=0.5)  # Reduced timeout
        except mp.queues.Full:
            if stop_signal.is_set():
                break
            continue
        time.sleep(0.01)  # Small sleep to prevent tight-loop hogging

    cap.release()
    detector_queue.put(None)  # Signal end of stream