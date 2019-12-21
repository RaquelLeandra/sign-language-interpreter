import cv2
import numpy as np


class HandExtractor:

    def __init__(self, hsv_mask_lower, hsv_mask_upper, roi_expansion):
        self.hsv_mask_lower = hsv_mask_lower
        self.hsv_mask_upper = hsv_mask_upper
        self.roi_expansion = roi_expansion

    def extract(self, frame, box):
        # Expand roi
        box.expand(self.roi_expansion)
        roi = box.get_roi(frame)

        # Color mask
        mask = cv2.inRange(cv2.cvtColor(roi, cv2.COLOR_BGR2HSV), self.hsv_mask_lower, self.hsv_mask_upper)

        # Morphology
        kernel = np.ones((2, 2), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.dilate(mask, kernel, iterations=3)

        # Pick best contour as mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # If no mask could be found
        if not contours:
            return roi, mask, None

        distances = [box.manhattan_to_contour(contour, frame.shape) for contour in contours]
        contour = contours[np.argmin(distances)]
        mask = cv2.fillPoly(np.zeros(shape=mask.shape, dtype=np.uint8), pts=[contour], color=(255, 255, 255))

        # Extract color hand
        x, y, w, h = cv2.boundingRect(contour)
        hand = cv2.bitwise_and(roi, roi, mask=mask)
        hand = hand[y:y + h, x:x + w]

        return roi, mask, hand
