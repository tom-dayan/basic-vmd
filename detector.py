import cv2
from typing import List, Tuple
import multiprocessing as mp
from basic_vmd import detector as basic_vmd_detector  # Import the provided detector function

def detector(detector_queue: mp.Queue, renderer_queue: mp.Queue) -> None:
    """
    Processes frames from the detector queue to detect motion using basic_vmd.py 
    and sends results to the renderer.

    Args:
        detector_queue (mp.Queue): Queue to receive frames from the streamer.
        renderer_queue (mp.Queue): Queue to send frames and detections to the renderer.
    """
    prev_frame = None
    counter = 0

    while True:
        frame = detector_queue.get()
        if frame is None:
            renderer_queue.put(None)  # Signal end of stream
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if counter == 0:
            # Initialize the previous frame on the first iteration
            prev_frame = gray_frame
            counter += 1
        else:
            # Use the provided detector function
            detections = basic_vmd_detector(gray_frame, prev_frame)
            prev_frame = gray_frame
            renderer_queue.put((frame, detections))
            counter += 1
