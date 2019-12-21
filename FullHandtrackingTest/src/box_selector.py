from colorsys import rgb_to_hsv
from time import time

import numpy as np

from src.fixed_fifo import FixedFIFO


class BoxSelector:

    def __init__(self, threshold, max_missing_time, buffer_size, max_area, kmeans_options, color_options):
        assert 0.0 < max_area <= 1.0
        self.max_missing_time = max_missing_time  # Seconds
        self.max_area = max_area  # Normalized
        self.threshold = threshold
        self.buffer_size = buffer_size
        self.__last_box = None
        self.__last_time = 0
        self.__last_scores = FixedFIFO(max_size=self.buffer_size)

        self.__kmeans_colors = kmeans_options['colors']
        self.__kmeans_criteria = kmeans_options['criteria']
        self.__kmeans_flags = kmeans_options['flags']
        self.__kmeans_attempts = kmeans_options['attempts']
        
        self.__pal_min_sat = color_options['min_sat']
        self.__pal_min_val = color_options['min_val']
        self.__pal_min_hue = color_options['min_hue']
        self.__pal_max_hue = color_options['max_hue']

        assert self.__pal_max_hue < self.__pal_min_hue  # Because it's cyclic with 0 on the center (range [0.0, 1.0])

    def select_box(self, frame, boxes, reset=False):
        boxes = [box for box in boxes if box.area < self.max_area]  # Discard boxes that are too big

        # If no "hands" are found TODO: Does this work in any useful way ?
        if len(boxes) == 0:
            missing_time = time() - self.__last_time
            if missing_time > self.max_missing_time:
                self.__last_box = None
            return self.__last_box
        else:
            self.__last_time = time()

        # Get boxes with best scores and correct colors
        # A best option would be to also check colors for boxes that do not have a bad threshold and discard those with
        # the wrong colors, but that would be too much intensive
        best_boxes = [box for box in boxes 
                      if box.score > self.threshold
                      and self.__adequate_color(frame, box, n_colors=self.__kmeans_colors)]

        # Reset search (or first search)
        if self.__last_box is None or reset:
            self.__last_box = None if not best_boxes else best_boxes[0]
            return self.__last_box

        # Calculate distances between last boxes and new boxes
        distances = [self.__last_box.manhattan_to_box(box) for box in best_boxes]

        # All distances are too big, hand is not expected to move that fast, try again but with all boxes
        if all(x >= 0.3 for x in distances):
            distances = [self.__last_box.manhattan_to_box(box) for box in boxes]
            closest = boxes[np.argmin(distances)]
        else:
            closest = best_boxes[np.argmin(distances)]

        # If the closest of all has a weird color, just assume we lost the hand
        if not self.__adequate_color(frame, closest, n_colors=self.__kmeans_colors):
            self.__last_box = None
            last_time = 0
            self.__last_scores.clear()
            return None

        # Add the score of the selected box to the list
        self.__last_scores.put(closest.score)

        # If this box keeps getting bad results, discard it
        if np.average(self.__last_scores) < self.threshold:
            self.__last_box = None if not best_boxes else best_boxes[0]
            self.__last_scores.clear()
        else:
            self.__last_box = closest

        return self.__last_box

    def __adequate_color(self, frame, box, n_colors=5):
        if box is not None:
            palette = box.kmeans_pixels(frame, n_colors, self.__kmeans_criteria, self.__kmeans_attempts, self.__kmeans_flags)
            return any(self.__color_in_range(BoxSelector.__bgr_to_hsv(color)) for color in palette)
        return False

    def __color_in_range(self, hsv):
        h, s, v = hsv
        return (s > self.__pal_min_sat) and \
               (v > self.__pal_min_val) and \
               (h < self.__pal_max_hue or h > self.__pal_min_hue)

    @staticmethod
    def __bgr_to_hsv(bgr):
        return rgb_to_hsv(bgr[2] / 255., bgr[1] / 255., bgr[0] / 255.)
