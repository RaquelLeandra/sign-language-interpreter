import os; os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import cv2
from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
from time import time

ALPHABET = np.array([c for c in '0123456789abcdefghijklmnopqrstuvwxyz'])


def classify(model, img, best_n=4):
    img = np.expand_dims(img, axis=0)
    predicts = model.predict(img)[0]
    bests_idx = np.argpartition(predicts, -best_n)[-best_n:]
    bests_idx = bests_idx[np.argsort(predicts[bests_idx])[::-1]]  # Sort bests
    return ALPHABET[bests_idx]


def main():
    model = load_model('model.h5')

    img = cv2.imread('y.png')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (256, 256))

    plt.imshow(img)
    plt.show()

    t0 = time()
    k = classify(model, img)
    t1 = time()
    print(t1 - t0, k)


if __name__ == '__main__':
    main()
