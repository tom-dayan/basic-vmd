import cv2
import multiprocessing as mp
import time
from typing import Any

def streamer(video_path: str, detector_queue: mp.Queue, stop_signal: Any) -> None:
    """
    Reads frames from a video file and sends them to the detector queue.

    Args:
        video_path (str): Path to the video file.
        detector_queue (mp.Queue): Queue to send frames to the detector.
        stop_signal (mp.Event): Event to signal stopping the streamer.
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