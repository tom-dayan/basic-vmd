import cv2
import imutils

def detector(gray_frame, prev_frame):
    """
    Detects motion by comparing the current frame with the previous frame.

    Args:
        gray_frame (ndarray): The current grayscale frame.
        prev_frame (ndarray): The previous grayscale frame.

    Returns:
        list: A list of bounding boxes (x, y, w, h) for detected motion.
    """
    diff = cv2.absdiff(gray_frame, prev_frame)
    thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    # Extract bounding boxes for detected motion
    detections = [cv2.boundingRect(c) for c in cnts if cv2.contourArea(c) > 500]
    return detections
