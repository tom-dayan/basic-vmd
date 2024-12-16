import cv2
import imutils

"""
Basic motion detection algorithm.

Compares two consecutive grayscale frames to detect motion.
Outputs bounding boxes for regions with detected motion.
"""

def detector(gray_frame: cv2.cvtColor, prev_frame: cv2.cvtColor):
    """
    Detects motion between consecutive frames.

    Args:
        gray_frame: Current grayscale frame.
        prev_frame: Previous grayscale frame.

    Returns:
        List of bounding boxes (x, y, w, h) for motion regions.
    """
    diff = cv2.absdiff(gray_frame, prev_frame)
    thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    # Extract bounding boxes for detected motion
    detections = [cv2.boundingRect(c) for c in cnts if cv2.contourArea(c) > 500]
    return detections
