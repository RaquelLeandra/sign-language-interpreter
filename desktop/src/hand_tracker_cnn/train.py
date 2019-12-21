import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


from pathlib import Path
import numpy as np
from keras.preprocessing.image import img_to_array, load_img
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import random
import matplotlib.pyplot as plt
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers import Input, BatchNormalization, Activation, Dropout
from keras.layers.merge import concatenate
from keras.models import Model
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

RESIZED_IMG_PATH = Path('../data/resized/images')
RESIZED_MASKS_PATH = Path('../data/resized/masks')
SIZE = 256
TRAIN = True


def get_data(imgs_path: Path, masks_path: Path):
    ids = list(imgs_path.iterdir())

    X = np.zeros((len(ids), SIZE, SIZE, 3), dtype=np.float32)
    y = np.zeros((len(ids), SIZE, SIZE, 1), dtype=np.float32)

    for n, id_ in enumerate(tqdm(ids)):
        img = img_to_array(load_img(id_))
        mask = img_to_array(load_img(masks_path / (id_.stem + '_mask.jpg'), color_mode='grayscale'))
        X[n] = img / 255.0
        y[n] = mask / 255.0

    return X, y


def conv2d_block(input_tensor, n_filters, kernel_size=3, batchnorm=True):
    # first layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
               padding="same")(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)
    # second layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
               padding="same")(x)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x


def get_unet(input_img, n_filters=16, dropout=0.5, batchnorm=True):
    # contracting path
    c1 = conv2d_block(input_img, n_filters=n_filters*1, kernel_size=3, batchnorm=batchnorm)
    p1 = MaxPooling2D((2, 2)) (c1)
    p1 = Dropout(dropout*0.5)(p1)

    c2 = conv2d_block(p1, n_filters=n_filters*2, kernel_size=3, batchnorm=batchnorm)
    p2 = MaxPooling2D((2, 2)) (c2)
    p2 = Dropout(dropout)(p2)

    c3 = conv2d_block(p2, n_filters=n_filters*4, kernel_size=3, batchnorm=batchnorm)
    p3 = MaxPooling2D((2, 2)) (c3)
    p3 = Dropout(dropout)(p3)

    c4 = conv2d_block(p3, n_filters=n_filters*8, kernel_size=3, batchnorm=batchnorm)
    p4 = MaxPooling2D(pool_size=(2, 2)) (c4)
    p4 = Dropout(dropout)(p4)

    c5 = conv2d_block(p4, n_filters=n_filters*16, kernel_size=3, batchnorm=batchnorm)

    # expansive path
    u6 = Conv2DTranspose(n_filters*8, (3, 3), strides=(2, 2), padding='same') (c5)
    u6 = concatenate([u6, c4])
    u6 = Dropout(dropout)(u6)
    c6 = conv2d_block(u6, n_filters=n_filters*8, kernel_size=3, batchnorm=batchnorm)

    u7 = Conv2DTranspose(n_filters*4, (3, 3), strides=(2, 2), padding='same') (c6)
    u7 = concatenate([u7, c3])
    u7 = Dropout(dropout)(u7)
    c7 = conv2d_block(u7, n_filters=n_filters*4, kernel_size=3, batchnorm=batchnorm)

    u8 = Conv2DTranspose(n_filters*2, (3, 3), strides=(2, 2), padding='same') (c7)
    u8 = concatenate([u8, c2])
    u8 = Dropout(dropout)(u8)
    c8 = conv2d_block(u8, n_filters=n_filters*2, kernel_size=3, batchnorm=batchnorm)

    u9 = Conv2DTranspose(n_filters*1, (3, 3), strides=(2, 2), padding='same') (c8)
    u9 = concatenate([u9, c1], axis=3)
    u9 = Dropout(dropout)(u9)
    c9 = conv2d_block(u9, n_filters=n_filters*1, kernel_size=3, batchnorm=batchnorm)

    outputs = Conv2D(1, (1, 1), activation='softmax') (c9)
    model = Model(inputs=[input_img], outputs=[outputs])

    return model


def plot_sample(X, y, preds, binary_preds, ix=None):
    if ix is None:
        ix = random.randint(0, len(X))

    fig, ax = plt.subplots(1, 4, figsize=(20, 10))
    ax[0].imshow(X[ix])
    ax[0].set_title('IN')

    ax[1].imshow(y[ix].squeeze())
    ax[1].set_title('MASK')

    ax[2].imshow(preds[ix].squeeze(), vmin=0, vmax=1)
    ax[2].set_title('MASK Predicted')

    ax[3].imshow(binary_preds[ix].squeeze(), vmin=0, vmax=1)
    ax[3].set_title('MASK Predicted binary')
    plt.show()


def main():
    X, y = get_data(RESIZED_IMG_PATH, RESIZED_MASKS_PATH)

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.15, random_state=2018)

    # Sample
    ix = random.randint(0, len(X_train))
    fig, ax = plt.subplots(1, 2, figsize=(20, 10))
    ax[0].imshow(X_train[ix])
    ax[0].set_title('IN')
    ax[1].imshow(y_train[ix].squeeze())
    ax[1].set_title('MASK')
    plt.show()

    # Model
    input_img = Input((SIZE, SIZE, 3), name='img')
    model = get_unet(input_img, n_filters=8, dropout=0.05, batchnorm=True)
    if TRAIN:
        model.compile(optimizer=Adam(), loss="binary_crossentropy", metrics=["accuracy"])
        model.summary()

        callbacks = [
            EarlyStopping(patience=10, verbose=1),
            ReduceLROnPlateau(factor=0.1, patience=3, min_lr=0.00001, verbose=1),
            ModelCheckpoint('../../data/hand_tracker_cnn/model-tgs-salt.h5', verbose=1, save_best_only=True, save_weights_only=True)
        ]
        results = model.fit(X_train, y_train, batch_size=32, epochs=30, callbacks=callbacks,
                            validation_data=(X_valid, y_valid))

        plt.figure(figsize=(8, 8))
        plt.title("Learning curve")
        plt.plot(results.history["loss"], label="loss")
        plt.plot(results.history["val_loss"], label="val_loss")
        plt.plot(np.argmin(results.history["val_loss"]), np.min(results.history["val_loss"]), marker="x", color="r",
                 label="best model")
        plt.xlabel("Epochs")
        plt.ylabel("log_loss")
        plt.legend()
        plt.show()

    model.load_weights('../../data/hand_tracker_cnn/model-tgs-salt.h5')
    preds_train = model.predict(X_train, verbose=1)
    preds_val = model.predict(X_valid, verbose=1)

    # Threshold predictions
    preds_train_t = (preds_train > 0.5).astype(np.uint8)
    preds_val_t = (preds_val > 0.5).astype(np.uint8)
    plot_sample(X_train, y_train, preds_train, preds_train_t)
    plot_sample(X_valid, y_valid, preds_val, preds_val_t)
    plot_sample(X_valid, y_valid, preds_val, preds_val_t)


if __name__ == '__main__':
    main()
