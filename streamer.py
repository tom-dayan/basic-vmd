import cv2
import multiprocessing as mp

def streamer(video_path: str, detector_queue: mp.Queue) -> None:
    """
    Read frames from a video file and sends them to the detector queue.

    Args:
        video_path (str): Path to the video file.
        detector_queue (mp.Queue): Queue to send frames to the detector.
    """
    cap = cv2.VideoCapture(video_path) 
    while True:
        ret, frame = cap.read()
        if not ret:
            detector_queue.put(None)  # Signal end of stream
            break
        detector_queue.put(frame)
    cap.release()
