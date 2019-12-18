import datetime
import os
import re
from collections import Counter
from time import time

import keras
import numpy as np
import pandas as pandas
import seaborn as sns
import tensorflow as tf
from keras import Model, layers
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, GlobalAveragePooling2D, Dropout
from keras.models import Sequential
from keras.utils import np_utils
from keras_preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

sns.set()


def dataset_resource(dataset_name):
    current_dir = os.path.dirname(__file__)
    resource_dir = os.path.abspath(os.path.join(current_dir, '../data/'))
    return os.path.join(resource_dir, dataset_name)


def model_resource(model_name):
    current_dir = os.path.dirname(__file__)
    resource_dir = os.path.abspath(os.path.join(current_dir, '../data/models'))
    os.makedirs(resource_dir, exist_ok=True)
    return os.path.join(resource_dir, model_name)


def history_resource(experiment_name):
    results_path = '../results/experiments/history/'
    os.makedirs(results_path, exist_ok=True)
    return os.path.join(results_path, experiment_name)


def confusion_matrix_resource(experiment_name):
    results_path = '../results/experiments/confusion_matrix/'
    os.makedirs(results_path, exist_ok=True)
    return os.path.join(results_path, experiment_name)


def experiment_resource(model_name):
    current_dir = os.path.dirname(__file__)
    resource_dir = os.path.abspath(os.path.join(current_dir, '../data/experiments'))
    os.makedirs(resource_dir, exist_ok=True)
    return os.path.join(resource_dir, model_name)


def write_the_model(model, experiment_name):
    with open(experiment_resource(experiment_name + '.txt'), 'w') as fh:
        # Pass the file handle in as a lambda function to make it callable
        model.summary(print_fn=lambda x: fh.write(x + '\n'))


def baseline_model(input_shape, num_classes, experiment_name):
    # Two hidden layers
    model = Sequential()
    model.add(Conv2D(8, 3, 3, activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(8, 3, 3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(8, activation='relu'))
    model.add(Dense(num_classes, activation=(tf.nn.softmax)))

    # Compile the NN
    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

    # print(model.summary())
    write_the_model(model, experiment_name)
    return model


def median_model(input_shape, num_classes, experiment_name):
    # Two hidden layers
    model = Sequential()
    model.add(Conv2D(64, 3, 3, activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(16, 3, 3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(16, 3, 3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(8, 3, 3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(54, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    # Compile the NN
    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

    # print(model.summary())
    write_the_model(model, experiment_name)
    return model


def mobilenet_net(input_shape, num_classes, experiment_name):
    """
    https://arxiv.org/pdf/1711.05225.pdf
    121-layer convolutional neural network
    (224, 224, 3)
    """
    base_model = keras.applications.mobilenet_v2.MobileNetV2(weights='imagenet', input_shape=input_shape,
                                                             include_top=False)

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dense(512, activation='relu')(x)
    preds = Dense(num_classes, activation='softmax')(x)  # final layer with softmax activation

    model = Model(inputs=base_model.input, outputs=preds)

    for layer in model.layers:
        layer.trainable = False
    # or if we want to set the first 20 layers of the network to be non-trainable
    for layer in model.layers[:20]:
        layer.trainable = False
    for layer in model.layers[20:]:
        layer.trainable = True

    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
    write_the_model(model, experiment_name)
    return model


def mobilenet_net_fine_tuning(input_shape, num_classes, experiment_name):
    """
    https://arxiv.org/pdf/1711.05225.pdf
    121-layer convolutional neural network
    (224, 224, 3)
    """
    mobilenet = keras.applications.mobilenet_v2.MobileNetV2(weights='imagenet', input_shape=input_shape,
                                                            include_top=False, classes=num_classes)
    model = Sequential()

    # Add the vgg convolutional base model
    model.add(mobilenet)

    # Add new layers
    model.add(layers.Flatten())
    model.add(layers.Dense(1024, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(num_classes, activation='softmax'))
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
    write_the_model(model, experiment_name)
    return model


def vgg16_model(input_shape, num_classes, experiment_name):
    """
    https://crherlihy.github.io/project/chestxray/cse6250_final_report.pdf
    :return:
    """
    model = keras.applications.vgg16.VGG16(include_top=True, weights=None, input_shape=input_shape,
                                           classes=num_classes)
    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
    write_the_model(model, experiment_name)
    return model


def vgg19_model(input_shape, num_classes, experiment_name):
    """
    https://crherlihy.github.io/project/chestxray/cse6250_final_report.pdf
    :return:
    """
    model = keras.applications.vgg19.VGG19(include_top=True, weights=None, input_shape=input_shape,
                                           classes=num_classes)
    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
    write_the_model(model, experiment_name)
    return model


def resnet_model(input_shape, num_classes, experiment_name):
    """
    https://crherlihy.github.io/project/chestxray/cse6250_final_report.pdf
    :return:
    """
    model = keras.applications.resnet.ResNet101(include_top=True, weights=None, input_shape=input_shape,
                                                classes=num_classes)
    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
    write_the_model(model, experiment_name)
    return model


def preprocessing(x, y):
    num_classes = len(Counter(y))
    x = x.astype('float32')
    normalized_x = x / np.max(x)
    categorical_y = np_utils.to_categorical(y, num_classes)
    return normalized_x, categorical_y


def plot_training_history(history, experiment_name):
    # Plot training & validation accuracy values
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig(history_resource(experiment_name + '_hist_accuracy'), bbox_inches='tight', dpi=200)
    plt.close()
    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig(history_resource(experiment_name + '_hist_loss'), bbox_inches='tight', dpi=200)
    plt.close()


def plot_confusion_matrix(y_true, y_pred, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues, experiment_name='experiment_cm'):
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(20, 18))
    sns.heatmap(data=cm, cmap=cmap, annot=True, xticklabels=classes, yticklabels=classes, linewidths=.5)
    b, t = plt.ylim()  # discover the values for bottom and top
    b += 0.5  # Add 0.5 to the bottom
    t -= 0.5  # Subtract 0.5 from the top
    plt.ylim(b, t)  # update the ylim(bottom, top) values
    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.xticks(rotation=0)
    plt.savefig(confusion_matrix_resource(experiment_name), bbox_inches='tight', dpi=200)
    plt.close()


def train_and_classify(train_it, val_it, test_it, num_classes, model_generator, experiment_name, dataset_name,
                       n_epochs=1, model_name='model'):
    batchX, batchy = train_it.next()
    experiment_name = experiment_name + '_' + model_name
    batch_size, img_rows, img_cols, channels = batchX.shape
    input_shape = (img_rows, img_cols, channels)

    # Define the NN architecture
    model = model_generator(input_shape, num_classes, experiment_name)
    initime = time()
    # Start training
    history = model.fit_generator(train_it, validation_data=val_it, epochs=n_epochs)
    plot_training_history(history, experiment_name)

    training_time = time() - initime

    # Evaluate the model with test set
    score = model.evaluate_generator(test_it, verbose=0)
    print('test loss:', score[0])
    print('test accuracy:', score[1])

    test_it.reset()

    Y_pred = model.predict_generator(test_it)
    y_pred = np.argmax(Y_pred, axis=-1)
    y_test = test_it.classes[test_it.index_array]
    true_classes = list(test_it.class_indices.keys())

    print('Analysis of results')

    print(classification_report(y_test, y_pred, target_names=true_classes))
    print(confusion_matrix(y_test, y_pred))

    # Saving model and weights
    model.save(model_resource(experiment_name + '_model_{}_{}.h5'.format(dataset_name, str(score[1]))))
    model_json = model.to_json()
    with open(model_resource(experiment_name + '_model_{}.json'.format(dataset_name)), 'w') as json_file:
        json_file.write(model_json)
    weights_file = experiment_name + "_weights_" + dataset_name + str(score[1]) + ".hdf5"
    model.save_weights(model_resource(weights_file), overwrite=True)
    with open(experiment_resource(experiment_name + '.txt'), 'a') as fh:
        fh.write('\nTraining time: {}\n'.format(datetime.timedelta(seconds=training_time)))
        fh.write('\ntest loss: {}\n'.format(score[0]))
        fh.write('test accuracy: {}\n'.format(score[1]))
        fh.write('Analysis of results:\n')
        fh.write(classification_report(y_test, y_pred, target_names=true_classes))
        fh.write('\nConfusion matrix:\n')
        fh.write(str(confusion_matrix(y_test, y_pred)))
    plot_confusion_matrix(y_test, y_pred, true_classes, experiment_name=experiment_name)
    return score[0], score[1], training_time


def load_data_iterators(data_dir):
    train_data_dir = os.path.join(data_dir, 'train')
    # validation_data_dir = os.path.join(data_dir, 'val')
    test_data_dir = os.path.join(data_dir, 'test')

    train_datagen = ImageDataGenerator(samplewise_center=True, samplewise_std_normalization=True, validation_split=0.1)
    test_datagen = ImageDataGenerator(samplewise_center=True, samplewise_std_normalization=True)

    train_it = train_datagen.flow_from_directory(
        train_data_dir,
        subset='training', class_mode='categorical'
    )

    val_it = train_datagen.flow_from_directory(
        train_data_dir,
        subset='validation', class_mode='categorical'
    )

    test_it = test_datagen.flow_from_directory(test_data_dir, class_mode='categorical')
    return train_it, val_it, test_it


def natural_sort(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)


def get_experiment_name_local():
    resource_dir = experiment_resource('')
    if not os.listdir(resource_dir):
        return 'experiment_0'
    last_experiment = natural_sort(os.listdir(resource_dir))
    experiment_index = int(last_experiment[-1].replace('.txt', '').split('_')[-2]) + 1
    experiment_name = 'experiment_{}'.format(experiment_index)
    return experiment_name


if __name__ == '__main__':
    dataset = 'asl_dataset'
    models = {  # 'baseline': baseline_model,
        # 'vgg16': vgg16_model,
        # 'vgg19': vgg19_model,
        # 'mobilenet': mobilenet_net,
        # 'finetuningmobilenet': mobilenet_net_fine_tuning,
        # 'resnet': resnet_model
        'median': median_model
    }
    results = pandas.DataFrame(columns=list(models.keys()), index=['Loss', 'Accuracy', 'Training time'])
    for model in models:
        experiment_name = get_experiment_name_local()
        print('Doing experiment {} with model {}'.format(experiment_name, model))
        train_it, val_it, test_it = load_data_iterators(dataset_resource(dataset))
        loss, accuracy, training_time = train_and_classify(train_it, val_it, test_it, num_classes=36,
                                                           model_generator=models[model],
                                                           experiment_name=experiment_name, dataset_name=dataset,
                                                           n_epochs=20, model_name=model)
        results.loc[:, model] = [loss, accuracy, training_time]
    results.to_csv(experiment_resource(experiment_name + '.csv'))
