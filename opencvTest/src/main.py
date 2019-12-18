from pathlib import Path

import json
import cv2
import numpy as np


MASK_PATH = Path('../data/masks')
COCO_IMG_PATH = Path('D:\\Users\\jsier\\Desktop\\COCO_data\\train2014')
COCO_ANN_PATH = Path('D:\\Users\\jsier\\Desktop\\COCO_data\\annotations_trainval2014\\annotations\\instances_train2014.json')
PERSON_CATEGORY = 1
DRAW_RADIUS = 20


paint = False
next_img = False
mouse_x = -1
mouse_y = -1


def on_mouse(event, x, y, flags, param):
    global paint, mouse_x, mouse_y, next_img

    if event == cv2.EVENT_LBUTTONDOWN:
        mouse_x = x
        mouse_y = y
        paint = True
    elif event == cv2.EVENT_RBUTTONDOWN:
        next_img = True
    elif event == cv2.EVENT_MOUSEMOVE:
        mouse_x = x
        mouse_y = y
    elif event == cv2.EVENT_LBUTTONUP:
        paint = False
        mouse_x = x
        mouse_y = y


def category_file_names(category):

    with open(COCO_ANN_PATH) as json_file:
        data = json.load(json_file)

    image_ids = set()
    for annotation in data['annotations']:
        if annotation['category_id'] == category:
            image_ids.add(annotation['image_id'])

    file_paths = []
    for image in data['images']:
        if image['id'] in image_ids:
            file_paths.append(image['file_name'])

    return file_paths


def main():
    global next_img, paint, mouse_x, mouse_y, start

    file_names = category_file_names(PERSON_CATEGORY)
    cv2.namedWindow('Frame')
    cv2.setMouseCallback('Frame', on_mouse)

    completed = []
    for file_name in MASK_PATH.iterdir():
        if file_name.is_file() and file_name.stem.endswith('_mask'):
            completed.append(file_name.stem[:-5])

    n_saved = len(completed)
    for file_name in file_names:

        if file_name[:-4] in completed:
            print(file_name, '[skipped]')
            continue

        frame = cv2.imread(str(COCO_IMG_PATH / file_name))
        rows, cols, _ = frame.shape
        mask = np.zeros((rows, cols, 3), np.uint8)

        modified = False
        while not next_img:
            if paint and mouse_x >= 0 and mouse_y >= 0:
                cv2.circle(mask, (mouse_x, mouse_y), DRAW_RADIUS, (255, 255, 255), thickness=-1)
                modified = True

            blended = cv2.addWeighted(frame, 0.8, mask, 0.2, 0)
            cv2.imshow('Frame', blended)
            cv2.imshow('Mask', mask)
            cv2.waitKey(10)

        if modified:
            n_saved += 1
            file_name = str(MASK_PATH / (file_name.replace('.jpg', '') + '_mask.jpg'))
            print('Saved: {:d}, {:s}'.format(n_saved, file_name))
            cv2.imwrite(file_name, mask)

        next_img = False


if __name__ == '__main__':
    main()
