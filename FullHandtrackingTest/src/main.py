import os; os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from utils import detector_utils as detector_utils
import cv2
from collections import deque
import tensorflow as tf
import datetime
import numpy as np
import argparse
from time import time
from colorsys import rgb_to_hsv
from keras.models import load_model
from src.test import classify
import matplotlib.pyplot as plt
from src.Box import Box


WIDTH = 180
HEIGHT = 320
THRESHOLD = 0.1
HANDS = 1
DEVICE = 0
BUFFER_SIZE = 30
MAX_AREA = 0.2
HSV_MASK_UPPER = np.array([50, 150, 255])
HSV_MASK_LOWER = np.array([5, 50, 50])
BOX_EXPAND = 1.5


def good_color(frame, box, debug=False):
    if box is None:
        return

    left, right, top, bottom = (int(box[1] * WIDTH), int(box[3] * WIDTH), int(box[0] * HEIGHT), int(box[2] * HEIGHT))
    roi = frame[top:bottom, left:right]

    pixels = np.float32(roi.reshape(-1, 3))
    n_colors = 3
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, .1)
    flags = cv2.KMEANS_RANDOM_CENTERS
    _, labels, palette = cv2.kmeans(pixels, n_colors, None, criteria, 4, flags)
    _, counts = np.unique(labels, return_counts=True)

    # Debugging:
    if debug:
        indices = np.argsort(counts)[::-1]
        freqs = np.cumsum(np.hstack([[0], counts[indices] / counts.sum()]))
        rows = np.int_(roi.shape[0] * freqs)

        dom_patch = np.zeros(shape=roi.shape, dtype=np.uint8)
        for i in range(len(rows) - 1):
            dom_patch[rows[i]:rows[i + 1], :, :] += np.uint8(palette[indices[i]])

        cv2.imshow('p', cv2.resize(dom_patch, (400, 400), interpolation=cv2.INTER_NEAREST))

        dominant = palette[np.argmax(counts)]
        a = rgb_to_hsv(dominant[2] / 255, dominant[1] / 255, dominant[0] / 255)
        b = [rgb_to_hsv(x[2] / 255, x[1] / 255, x[0] / 255) for x in palette]
        print((a[0] * 360, a[1], a[2]))
        print([(a[0] * 360, a[1], a[2]) for a in b])
        print()

    ok = False
    for color in palette:
        h, s, v = rgb_to_hsv(color[2], color[1], color[0])
        if s > 0.1 and v > 0.1 and (h < 0.15 or h > 0.97):
            ok = True
            break

    return ok


last_box = None
last_time = 0
last_scores = deque()
def select_box(frame, boxes, scores, threshold=0.15, max_missing_time=1, reset=False):
    global last_box, last_time, last_scores

    # Convert to Box objects
    boxes = [Box(box_data) for box_data in boxes]

    # Discard boxes that are too big
    boxes = [box for box in boxes if box.area < MAX_AREA]

    # If no "hands" are found
    if len(boxes) == 0:
        missing_time = time() - last_time
        if missing_time > max_missing_time:
            last_box = None
        return last_box
    else:
        last_time = time()  # TODO

    # Get boxes with best scores
    best_boxes = []
    best_scores = []
    for i, box in enumerate(boxes):
        if scores[i] > threshold and good_color(frame, box):
            best_boxes.append(box)
            best_scores.append(scores[i])

    # Reset search (or first search)
    if last_box is None or reset:
        last_box = None if not best_boxes else best_boxes[0]
        return last_box

    # Calculate distances between last boxes and new boxes
    distances = [last_box.manhattan_to_box(box) for box in best_boxes]

    # All distances are too big, hand is not expected to move that fast, try again but with all boxes
    if all(x >= 0.3 for x in distances):
        distances = [last_box.manhattan_to_box(box) for box in boxes]
        index = np.argmin(distances)
        closest = boxes[index]
        closest_score = scores[index]
        # If the closest of all has a weird color, just assume we lost the hand
        if not good_color(frame, closest):
            last_box = None
            last_scores = deque()
            last_time = 0
            return
    else:
        index = np.argmin(distances)
        closest = best_boxes[index]
        closest_score = best_scores[index]

    last_scores.append(closest_score)
    if len(last_scores) > BUFFER_SIZE:
        last_scores.popleft()

    # print(np.average(last_scores))
    # If this box keeps getting bad results, discard it
    if np.average(last_scores) < threshold:
        last_box = None if not best_boxes else best_boxes[0]
        last_scores = deque()
        return last_box
    else:
        last_box = closest
        return last_box


def extract(frame, box):
    # Expand roi
    box.expand(BOX_EXPAND)
    roi = box.get_roi(frame)

    # Color mask
    mask = cv2.inRange(cv2.cvtColor(roi, cv2.COLOR_BGR2HSV), HSV_MASK_LOWER, HSV_MASK_UPPER)

    # Morphology
    kernel = np.ones((2, 2), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.dilate(mask, kernel, iterations=3)

    # Pick best contour as mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    distances = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        y = (y + h // 2) / HEIGHT + box.top
        x = (x + w // 2) / WIDTH + box.left
        distances.append(box.manhattan_to_point((x, y)))

    index = np.argmin(distances)
    contour = contours[index]
    x, y, w, h = cv2.boundingRect(contour)
    mask = cv2.fillPoly(np.zeros(shape=mask.shape, dtype=np.uint8), pts=[contour], color=(255, 255, 255))

    # Extract color hand
    hand = cv2.bitwise_and(roi, roi, mask=mask)
    hand = hand[y:y+h, x:x+w]

    return roi, mask, hand


def main():
    assert WIDTH < HEIGHT

    detection_graph, sess = detector_utils.load_inference_graph()
    cap = cv2.VideoCapture(DEVICE)
    model = load_model('model.h5')

    while True:
        t0 = time()
        ret, frame = cap.read()

        # img = ImageGrab.grab()
        # img = np.array(img.getdata(), dtype='uint8').reshape((img.size[1], img.size[0], 3))
        # img = img[363:652, 1196:1709]
        # frame = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        # ret = True

        h, w, ch = frame.shape
        t1 = time()

        if not ret:
            break

        # Force height > width from input
        if w > h:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            w, h = h, w  # Swap

        scale_w = WIDTH / w
        scale_h = HEIGHT / h

        # If scales do not match, crop from size
        if scale_w > scale_h:
            desired_h = int(HEIGHT / scale_w)
            top = (h - desired_h) // 2
            bot = (h + desired_h) // 2
            frame = frame[top:bot, :, :]
            h = desired_h
        elif scale_h > scale_w:
            desired_w = int(WIDTH / scale_h)
            left = (w - desired_w) // 2
            right = (w + desired_w) // 2
            frame = frame[:, left:right, :]
            w = desired_w

        frame = cv2.resize(frame, (WIDTH, HEIGHT), interpolation=cv2.INTER_LINEAR)
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        t2 = time()

        boxes, scores = detector_utils.detect_objects(frame, detection_graph, sess)
        t3 = time()

        box = select_box(frame, boxes, scores)
        t4 = time()
        # t3 = time()
        # t4 = time()
        # box = [0.25, 0.25, 0.60, 0.75]

        if box is not None:
            roi, mask, hand = extract(frame, box)
            hand = cv2.cvtColor(hand, cv2.COLOR_BGR2RGB)
            hand = cv2.resize(hand, (256, 256))
            t5 = time()

            labels = classify(model, hand)
            t6 = time()
            print(labels)

            h = roi.shape[0]
            w = roi.shape[1]
            # cv2.imshow('roi', cv2.resize(roi, (int(w * 1.5), int(h * 1.5))))
            cv2.imshow('mask', cv2.resize(mask, (int(w * 3), int(h * 3))))
            cv2.imshow('h', cv2.resize(hand, (int(w * 3), int(h * 3))))
        else:
            t5 = t4

        # print('Read: {:.2f} | Pre: : {:.2f} | Detector: {:.2f} | Post-1: {:.2f} | Post-2: {:.2f} | Top-Scores: {:s}'.format(
        #     t1 - t0, t2 - t1, t3 - t2, t4 - t3, t5 - t4, ', '.join(format(x, '.2f') for x in scores[:5])
        # ))
        box.draw(frame)
        cv2.imshow('Detection', frame)
        if cv2.waitKey(33) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
