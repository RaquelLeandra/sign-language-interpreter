import os; os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # USE CPU

from time import time

import cv2
import numpy as np
from keras.models import load_model

from src.handtracking.utils import detector_utils as detector_utils
from src.video_to_sign.box import Box
from src.video_to_sign.box_selector import BoxSelector
from src.video_to_sign.hand_extractor import HandExtractor

# Note that you need to clone subomdule handtracking to access detector_utils
# If using IntellIJ mark the submodule folder and the src folder as sources root

DEBUG_INFO = True

# Frame config
DEVICE = 0
WIDTH = 180
HEIGHT = 320

# Box detector config
HANDS = 1

# Box selector config
THRESHOLD = 0.15
MAX_MISSING_TIME = 1
BUFFER_SIZE = 30
MAX_AREA = 0.2
KMEANS_OPTIONS = {
    'colors': 3,
    'criteria': (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, .1),
    'flags': cv2.KMEANS_RANDOM_CENTERS,
    'attempts': 4,
}
COLOR_OPTIONS = {
    'min_sat': 0.1,
    'min_val': 0.1,
    'min_hue': 0.97,
    'max_hue': 0.15
}

# Hand extractor config
# TODO same range for good_color and mask ? Or extract the mask range from the roi palette ?
HSV_MASK_UPPER = np.array([15, 150, 255])  # H in [0, 179], S in [0, 255], V in [0, 255]
HSV_MASK_LOWER = np.array([0, 50, 50])
BOX_EXPAND = 1.5

# Hand classifier config
ALPHABET = np.array([c for c in '0123456789abcdefghijklmnopqrstuvwxyz'])
CLASSIFIER_MODEL_PATH = '../../data/classifier/best_model_results/experiment_13_median_model_asl_dataset_0.8111110925674438.h5'
CLASSIFIER_SIZE = (256, 256)


def convert_for_classifier(img):
    return cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), CLASSIFIER_SIZE)


def classify(model, img, best_n=4):
    img = np.expand_dims(img, axis=0)
    predicts = model.predict(img)[0]
    bests_idx = np.argpartition(predicts, -best_n)[-best_n:]
    bests_idx = bests_idx[np.argsort(predicts[bests_idx])[::-1]]  # Sort bests
    return ALPHABET[bests_idx]


def detect_boxes(frame, detection_graph, sess):
    boxes, scores = detector_utils.detect_objects(frame, detection_graph, sess)
    return [Box(box_data, score) for box_data, score in zip(boxes, scores)]


def pre_process_frame(frame):
    h, w, ch = frame.shape

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

    elif scale_h > scale_w:
        desired_w = int(WIDTH / scale_h)
        left = (w - desired_w) // 2
        right = (w + desired_w) // 2
        frame = frame[:, left:right, :]

    return cv2.resize(frame, (WIDTH, HEIGHT), interpolation=cv2.INTER_LINEAR)


def main():
    assert WIDTH < HEIGHT

    detection_graph, sess = detector_utils.load_inference_graph()
    cap = cv2.VideoCapture(DEVICE)
    model = load_model(CLASSIFIER_MODEL_PATH)
    box_selector = BoxSelector(THRESHOLD, MAX_MISSING_TIME, BUFFER_SIZE, MAX_AREA, KMEANS_OPTIONS, COLOR_OPTIONS)
    hand_extractor = HandExtractor(HSV_MASK_LOWER, HSV_MASK_UPPER, BOX_EXPAND)

    while True:
        t0 = time()

        ret, frame = cap.read()
        if not ret: break
        t1 = time()

        frame = pre_process_frame(frame)  # Note that color for opencv is BGR
        t2 = time()

        boxes = detect_boxes(frame, detection_graph, sess)
        t3 = time()

        box = box_selector.select_box(frame, boxes)
        t4 = time()

        # Box found
        roi, mask, hand, labels = None, None, None, None
        if box is not None:
            roi, mask, hand = hand_extractor.extract(frame, box)
        t5 = time()

        if hand is not None:
            labels = classify(model, convert_for_classifier(hand))
        t6 = time()

        if DEBUG_INFO:
            if box is not None:
                cv2.imshow('roi',  cv2.resize(roi,  (int(roi.shape[1]  * 2), int(roi.shape[0]  * 2))))
                cv2.imshow('mask', cv2.resize(mask, (int(mask.shape[1] * 2), int(mask.shape[0] * 2))))
            if hand is not None:
                cv2.imshow('hand', cv2.resize(hand, (int(hand.shape[1] * 2), int(hand.shape[0] * 2))))

            print('\rRead: {:.2f} Pre: {:.2f} Detector: {:.2f} Selector: {:.2f} Extractor: {:.2f} Classifier {:.2f} '
                  'Most probable classifications: {:s}'
                  .format(t1 - t0, t2 - t1, t3 - t2, t4 - t3, t5 - t4, t6 - t5,
                          str(labels) if box is not None else 'Hand not detected'), end='')

        # Display basic info
        if box:
            box.draw(frame)
        cv2.imshow('Detection', frame)
        if cv2.waitKey(33) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
