#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 24 07:06:13 2021

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

class Models():
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

    def trainmodels(path, n_epoch,summary_models = None):
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
        model = Models.cnnmodels(num_classes, img_height, img_width)
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

        filepath = 'input/models_noise/'
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

    # def predictedstationquality(path, models):
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
    #     y_axis = "linear"  # linear or log
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
    #     Models.predicted(directory+'/input/tmp_files/'+file_names, models)

    def predicted(path, models):
        class_names = ['Baik','Buruk','Sedang'] 
        model = tf.keras.models.load_model(models)

        files       = path
        files_      = tf.keras.preprocessing.image.load_img(files, target_size=(180,180))
        files_array = tf.keras.preprocessing.image.img_to_array(files_)
        files_array = tf.expand_dims(files_array, 0)

        prediction  = model.predict(files_array)
        score       = tf.nn.softmax(prediction[0])

        print('This index station is =={}== with a == {:.2f}% ==  confidence'.format(class_names[np.argmax(score)], 100 * np.max(score)))


