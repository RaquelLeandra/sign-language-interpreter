from pathlib import Path
import cv2
import numpy as np
import math
from itertools import combinations


MASK_PATH = Path('../data/masks')
COCO_IMG_PATH = Path('D:\\Users\\jsier\\Desktop\\COCO_data\\train2014')
RESIZED_IMG_PATH = Path('../data/resized/images')
RESIZED_MASKS_PATH = Path('../data/resized/masks')
SIZE = 256
DEBUG = False


def boundaries(x, y, w, h):
    y_min = 0
    y_max = h
    x_min = 0
    x_max = w
    if h > w:  # Remove from h
        half = int(math.ceil(w / 2))
        y_min = y - half
        y_max = y + half

        if y_min < 0:
            y_max += abs(y_min)
            y_min = 0
        elif y_max > h:
            y_min -= y_max - h
            y_max = h

    elif w > h:  # Remove from w
        half = int(math.ceil(h / 2))
        x_min = x - half
        x_max = x + half

        if x_min < 0:
            x_max += abs(x_min)
            x_min = 0
        elif x_max > w:
            x_min -= x_max - w
            x_max = w

    # TODO Temporary fix
    if abs(y_min - y_max) > abs(x_min - x_max):
        y_min += 1
    elif abs(y_min - y_max) < abs(x_min - x_max):
        x_max += 1
    if abs(y_min - y_max) > abs(x_min - x_max):
        print('Error in size')

    return y_min, y_max, x_min, x_max


def main():

    masks = []
    for file_name in MASK_PATH.iterdir():
        if file_name.is_file() and file_name.stem.endswith('_mask'):
            masks.append(file_name.stem[:-5])

    masks_done = 0
    for file_name in COCO_IMG_PATH.iterdir():
        if file_name.stem in masks:
            frame = cv2.imread(str(COCO_IMG_PATH / file_name))
            mask = cv2.imread(str(MASK_PATH / file_name.stem) + '_mask.jpg')

            # --- Make image squared centered on hands
            # 1) Locate best center of the segmentation
            threshold = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            _, threshold = cv2.threshold(threshold, 128, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            xs = []
            ys = []
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                x += w // 2
                y += h // 2
                xs.append(x)
                ys.append(y)

            # If no contours, this image is mistakenly labeled
            if len(contours) == 0:
                continue

            # 2) If distances between any of the blobs of the segmentation is too big, discard a blob
            h, w, ch = frame.shape

            # Care, complexity is exponential, but no more than a few blobs are expected
            indexes = np.arange(len(contours))
            combs = []
            for i in range(len(contours), 0, -1):
                combs += combinations(indexes, i)

            selection = None
            for comb in combs:

                ok = True
                for i, (x1, y1) in enumerate(zip(xs, ys)):
                    for j, (x2, y2) in enumerate(zip(xs, ys)):
                        if i not in comb or j not in comb:
                            continue

                        dx = x2 - x1
                        dy = y2 - y1
                        dist = math.sqrt(dx * dx + dy * dy)
                        if dist > min(w, h):
                            ok = False
                            break

                if ok:
                    selection = comb
                    break

            xs_aux = [xs[index] for index in selection]
            ys_aux = [ys[index] for index in selection]
            xs = xs_aux
            ys = ys_aux

            # center:
            x = int(np.average(xs))
            y = int(np.average(ys))

            # 2) Cut from the biggest size to make the image squared
            y_min, y_max, x_min, x_max = boundaries(x, y, w, h)
            crop_mask = mask[y_min:y_max, x_min:x_max]
            crop_frame = frame[y_min:y_max, x_min:x_max]

            resized_mask = cv2.resize(crop_mask, (SIZE, SIZE), interpolation=cv2.INTER_LINEAR)
            resized_frame = cv2.resize(crop_frame, (SIZE, SIZE), interpolation=cv2.INTER_LINEAR)

            cv2.imwrite(str(RESIZED_IMG_PATH / str(masks_done)) + '_' + str(SIZE) + '.jpg', resized_frame)
            cv2.imwrite(str(RESIZED_MASKS_PATH / str(masks_done)) + '_' + str(SIZE) + '_mask.jpg', resized_mask)

            if DEBUG:
                cv2.imshow('crop', resized_frame)
                cv2.imshow('ori', resized_mask)
                cv2.waitKey(1000)

            masks_done += 1
            masks.remove(file_name.stem)
            print('\rRemaining:', len(masks), end='   ')

    print()
    print('Done')


if __name__ == '__main__':
    main()
