import cv2
import multiprocessing as mp
from basic_vmd import detector as basic_vmd_detector
from typing import Any

"""
Processes video frames to detect motion.

Receives frames from the Streamer, performs motion detection, 
and sends results (frames + detections) to the Renderer.
"""

def detector(detector_queue: mp.Queue, renderer_queue: mp.Queue, stop_signal: Any) -> None:
    """
    Detects motion in video frames and sends results to the renderer.

    Args:
        detector_queue: Queue receiving frames from the Streamer.
        renderer_queue: Queue sending frames and detections to the Renderer.
        stop_signal: Event signaling termination.
    """
    prev_frame = None
    counter = 0

    while not stop_signal.is_set():
        try:
            frame = detector_queue.get(timeout=0.5)  # Reduced timeout
        except mp.queues.Empty:
            if stop_signal.is_set():
                break
            continue

        if frame is None:  # End of stream signal
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if counter == 0:
            prev_frame = gray_frame
            counter += 1
        else:
            detections = basic_vmd_detector(gray_frame, prev_frame)
            prev_frame = gray_frame
            try:
                renderer_queue.put((frame, detections), timeout=0.5)  # Reduced timeout
            except mp.queues.Full:
                if stop_signal.is_set():
                    break
                continue
            counter += 1

    renderer_queue.put(None)  # Signal end of stream to renderer