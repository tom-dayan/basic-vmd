import cv2
import multiprocessing as mp
from basic_vmd import detector as basic_vmd_detector
from typing import Any

def detector(detector_queue: mp.Queue, renderer_queue: mp.Queue, stop_signal: Any) -> None:
    """
    Processes frames from the detector queue to detect motion using basic_vmd.py 
    and sends results to the renderer.

    Args:
        detector_queue (mp.Queue): Queue to receive frames from the streamer.
        renderer_queue (mp.Queue): Queue to send frames and detections to the renderer.
        stop_signal (mp.Event): Event to signal stopping the detector.
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