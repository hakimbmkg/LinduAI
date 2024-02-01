#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 4 09:07:23 2022

@author: hakimbmkg
"""

import os
# import librosa
# import librosa.display

from obspy.core.stream import read
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import matplotlib.pyplot as plt
import numpy as np
import datetime
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

from pathlib import Path

class Modelsmag():
    """
    class for make model and main apps for training and testing
    reference from tensorflow CNN
    """

    def __init__(self) -> None:
        pass

    global directory
    directory = os.getcwd()

    def cnnmodels(num_classes, img_height, img_width):
        model = Sequential([
            tf.keras.layers.experimental.preprocessing.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
            layers.Conv2D(16, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(32, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Dropout(0.2),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dense(num_classes)
            ])
        return(model)

    def trainmodels(path, n_epoch, summary_models = None):
        data_dir = Path(path)
        files_count = len(list(data_dir.glob('*/*.png')))
        print(f'**Summary** files on folder is {files_count} files')

        batch_size = 32
        img_height = 180
        img_width  = 180

        filenames = tf.io.gfile.glob(str(data_dir) + '/*/*')
        filenames = tf.random.shuffle(filenames)
        num_samples = len(filenames)

        train_ds = tf.keras.preprocessing.image_dataset_from_directory(
            data_dir,
            validation_split=0.2,
            subset="training",
            seed=123,
            image_size=(img_height, img_width),
            batch_size=batch_size)

        val_ds = tf.keras.preprocessing.image_dataset_from_directory(
            data_dir,
            validation_split=0.2,
            subset="validation",
            seed=123,
            image_size=(img_height, img_width),
            batch_size=batch_size)

        class_names = train_ds.class_names
        print(f'**Summary** class is {class_names}')

        plt.figure(figsize=(10, 10))
        for images, labels in train_ds.take(1):
            for i in range(9):
                ax = plt.subplot(3, 3, i + 1)
                plt.imshow(images[i].numpy().astype("uint8"))
                plt.title(class_names[labels[i]])
                plt.axis("off")
                plt.savefig(directory+'/input/tmp_files/specvslabels.png', bbox_inches='tight', dpi=300)
      

        for image_batch, labels_batch in train_ds:
            print(f'**Summary** image batch shape {image_batch.shape}')
            print(f'**Summary** labels batch shape{labels_batch.shape}')
            break

        AUTOTUNE = tf.data.AUTOTUNE
        train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
        val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

        normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1./255, offset=0.0)
        normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
        image_batch, labels_batch = next(iter(normalized_ds))
        first_image = image_batch[0]
        print(np.min(first_image), np.max(first_image)) 

        num_classes = len(class_names)
        model = Modelsmag.cnnmodels(num_classes, img_height, img_width)
        model.compile(optimizer='adam',loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics=['accuracy'])

        if summary_models == 'summary':
            model.summary()

        epochs = n_epoch
        history = model.fit(train_ds,validation_data=val_ds,epochs=epochs)

        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']

        loss = history.history['loss']
        val_loss = history.history['val_loss']

        epochs_range = range(epochs)

        filepath = 'input/models_mags/'
        tf.keras.models.save_model(model, filepath)

        plt.figure(figsize=(8, 8))
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, acc, label='Training Accuracy')
        plt.plot(epochs_range, val_acc, label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.title('Training and Validation Accuracy')

        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, loss, label='Training Loss')
        plt.plot(epochs_range, val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.title('Training and Validation Loss')
        # plt.show()
        plt.savefig(directory+'/input/tmp_files/trainvsval.png', bbox_inches='tight', dpi=300)
        plt.close()

        # return history, model

    # def predictedmag(path, models):
    #     # files_path = Path(path)
    #     st          = read(path)
    #     data        = st[0].data.astype('float32')
    #     sr          = int(st[0].stats.sampling_rate)
    #     max_points  = int(st[0].stats.npts)
    #     Offset      = 0

    #     hop_length = 128
    #     n_fft = 256
    #     cmap = 'jet'
    #     bins_per_octave = 12
    #     auto_aspect = False
    #     y_axis = "linear"  
    #     fmin = None
    #     fmax = 5.0

    #     # Librosa spectrogram
    #     D = librosa.amplitude_to_db(np.abs(librosa.stft(data, hop_length=hop_length, n_fft=n_fft)), ref=np.max)
    #     fig, ax = plt.subplots()
    #     img = librosa.display.specshow(D, y_axis=y_axis, sr=sr,
    #     hop_length=hop_length, x_axis='time', ax=ax, cmap=cmap, bins_per_octave=bins_per_octave,
    #     auto_aspect=auto_aspect)
        
    #     if fmin is not None:
    #         fmin0 = fmin
    #     else:
    #         fmin0 = 0

    #     if fmax is not None:
    #         fmax0 = fmax
    #     else:
    #         fmax0 = sr/2

    #     ax.set_ylim([fmin, fmax])
    #     fig.colorbar(img, ax=ax, format="%+2.f dB")
    #     file_names = str(datetime.datetime.now())+'.png'
    #     plt.savefig(directory+'/input/tmp_files/'+file_names, bbox_inches='tight', dpi=300)
    #     plt.close()
    #     Modelsmag.predicted(directory+'/input/tmp_files/'+file_names, models)

    def predicted(path, models):
        class_names =  ['1.78', '1.96', '2.04', '2.05', '2.06', '2.12', '2.13', '2.14', '2.17', '2.18', '2.19', '2.2', '2.22', '2.24', '2.25', '2.28', '2.3', '2.31', '2.32', '2.34', '2.35', '2.36', '2.39', '2.4', '2.41', '2.42', '2.43', '2.44', '2.46', '2.47', '2.48', '2.5', '2.51', '2.52', '2.53', '2.54', '2.55', '2.56', '2.57', '2.58', '2.59', '2.6', '2.61', '2.62', '2.63', '2.64', '2.65', '2.66', '2.67', '2.68', '2.69', '2.71', '2.72', '2.73', '2.74', '2.75', '2.76', '2.77', '2.78', '2.79', '2.8', '2.81', '2.82', '2.83', '2.84', '2.85', '2.86', '2.87', '2.88', '2.89', '2.9', '2.91', '2.92', '2.93', '2.94', '2.95', '2.96', '2.97', '2.98', '2.99', '3.0', '3.01', '3.02', '3.03', '3.04', '3.05', '3.06', '3.07', '3.08', '3.09', '3.1', '3.11', '3.12', '3.13', '3.14', '3.15', '3.16', '3.17', '3.18', '3.19', '3.2', '3.21', '3.22', '3.23', '3.24', '3.25', '3.26', '3.27', '3.28', '3.29', '3.3', '3.31', '3.32', '3.33', '3.34', '3.35', '3.36', '3.37', '3.38', '3.39', '3.4', '3.41', '3.42', '3.43', '3.44', '3.45', '3.46', '3.47', '3.48', '3.49', '3.5', '3.51', '3.52', '3.53', '3.54', '3.55', '3.56', '3.57', '3.58', '3.59', '3.6', '3.61', '3.62', '3.63', '3.64', '3.65', '3.66', '3.67', '3.68', '3.69', '3.7', '3.71', '3.72', '3.73', '3.74', '3.75', '3.76', '3.77', '3.78', '3.79', '3.8', '3.81', '3.82', '3.83', '3.84', '3.85', '3.86', '3.87', '3.88', '3.89', '3.9', '3.91', '3.92', '3.93', '3.94', '3.95', '3.96', '3.97', '3.98', '3.99', '4.0', '4.01', '4.02', '4.03', '4.04', '4.05', '4.06', '4.08', '4.09', '4.1', '4.11', '4.12', '4.13', '4.14', '4.15', '4.16', '4.17', '4.18', '4.19', '4.2', '4.21', '4.22', '4.23', '4.24', '4.26', '4.27', '4.28', '4.29', '4.3', '4.32', '4.33', '4.34', '4.35', '4.36', '4.37', '4.38', '4.39', '4.4', '4.42', '4.43', '4.44', '4.45', '4.46', '4.47', '4.48', '4.49', '4.5', '4.51', '4.52', '4.53', '4.54', '4.55', '4.56', '4.57', '4.58', '4.59', '4.6', '4.61', '4.62', '4.63', '4.64', '4.65', '4.66', '4.67', '4.68', '4.69', '4.7', '4.71', '4.72', '4.73', '4.74', '4.75', '4.76', '4.8', '4.82', '4.83', '4.84', '4.85', '4.86', '4.87', '4.88', '4.89', '4.9', '4.91', '4.92', '4.93', '4.94', '4.95', '4.97', '4.98', '4.99', '5.0', '5.02', '5.03', '5.04', '5.05', '5.07', '5.08', '5.09', '5.11', '5.14', '5.16', '5.19', '5.23', '5.26', '5.28', '5.3', '5.38', '5.42', '5.51', '5.55', '5.61', '5.71', '5.74', '6.11', '6.38']
        model = tf.keras.models.load_model(models)

        files       = path
        files_      = tf.keras.preprocessing.image.load_img(files, target_size=(180,180))
        files_array = tf.keras.preprocessing.image.img_to_array(files_)
        files_array = tf.expand_dims(files_array, 0)

        prediction  = model.predict(files_array)
        score       = tf.nn.softmax(prediction[0])

        print('This Predection of Magnitude is  =={}== with a == {:.2f} == percent confidence'.format(class_names[np.argmax(score)], 100 * np.max(score)))
        

