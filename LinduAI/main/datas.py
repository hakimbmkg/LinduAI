#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 26 09:31:23 2021

@author: hakimbmkg
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import pathlib
import shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
# import h5py
from tensorflow.keras import layers
from tensorflow.keras import models
from IPython import display
from obspy import read
from tqdm import tqdm

seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)

@tf.function
class Datas:
    """
    class for conditioning data, ex: split train and test, shuffle data, and merge to hdf5 file for datase
    """

    global directory
    directory = os.getcwd()

    def split_labels(csv_path):
        """
        read data from csv path and make train and test
        """

        if not os.path.exists(directory+'/input/dataset/'):
            os.makedirs(directory+'/input/dataset')
            print('== folder /input/dataset/ created')

        data_path           = pd.read_csv(csv_path)
        data_path_class     = data_path[['files_name','class']]
        arr_data_path       = data_path_class.to_numpy()
        labels = data_path['class'].value_counts().index.tolist()

        for a,b in enumerate(labels):
            if not os.path.exists(directory+'/input/dataset/'+b):
                os.makedirs(directory+'/input/dataset/'+b)
                print(f'== folder /input/dataset/{b} created')

            for x in arr_data_path:
                if x[1] == b:
                    src_ = directory+'/input/waveform/'+x[0]
                    des_ = directory+'/input/dataset/'+b+'/'
                    if not os.path.exists(directory+'/input/dataset/'+b+'/'+x[0]):
                        try:
                            shutil.copy(src_, des_)
                        except shutil.SameFileError:
                            print(f'Source and destination represents the same file.')
                        except IsADirectoryError:
                            print(f'Destination is a directory.')
                        except PermissionError:
                            print(f'Permission denied. check your permision')
                        except:
                            print(f'Error occurred while copying file.')


    def split_labels_mags(csv_path):
        """
        read data from csv path and make train and test
        """

        if not os.path.exists(directory+'/input/dataset_EQ/datasEQ_spectogram/'):
            os.makedirs(directory+'/input/dataset_EQ/datasEQ_spectogram/')
            print('== folder /input/dataset_EQ/datasEQ_spectogram/ created')

        data_path           = pd.read_csv(csv_path)
        data_path_class     = data_path[['trace_name','source_magnitude','receiver_code']]
        arr_data_path       = data_path_class.to_numpy()
        labels = data_path['source_magnitude'].value_counts().index.tolist()

        for a,b in enumerate(labels):
            fold_b = str(b)
            # print(fold_b)
            if not os.path.exists(directory+'/input/dataset_EQ/datasEQ_spectogram/'+fold_b):
                os.makedirs(directory+'/input/dataset_EQ/datasEQ_spectogram/'+fold_b)
                print(f'== folder /input/dataset_EQ/datasEQ_spectogram/{fold_b} created')

            for x in arr_data_path:
                if x[1] == b:
                    src_ = directory+'/input/dataset_EQ/spectogram/'+x[0]+'.png'
                    des_ = directory+'/input/dataset_EQ/datasEQ_spectogram/'+fold_b+'/'
                    if not os.path.exists(directory+'/input/dataset_EQ/datasEQ_spectogram/'+fold_b+'/'+x[0]):
                        try:
                            shutil.copy(src_, des_)
                        except shutil.SameFileError:
                            print(f'Source and destination represents the same file.')
                        except IsADirectoryError:
                            print(f'Destination is a directory.')
                        except PermissionError:
                            print(f'Permission denied. check your permision')
                        except:
                            print(f'Error occurred while copying file. {x[0]} - {x[2]} - please check path files')


    def shuffle_datas(n_trains, n_vals, n_tests):
        ## reference tensorflow for wav files see tensorflow pages
        data_dir = pathlib.Path(directory+'/input/dataset/')
        commands = np.array(tf.io.gfile.listdir(str(data_dir)))
        filenames = tf.io.gfile.glob(str(data_dir)+'/*/*')
        filenames = tf.random.shuffle(filenames)
        num_samples = len(filenames)

        n_train = int(round((num_samples*int(n_trains))/100))
        n_val   = int(round((num_samples*int(n_vals))/100))
        n_test  = int(round((num_samples*int(n_tests))/100))

        train_files = filenames[:n_train]
        val_files = filenames[n_train: n_train + n_val]
        test_files = filenames[-n_test:]

        print('Training set size', len(train_files))
        print('Validation set size', len(val_files))
        print('Test set size', len(test_files))

        return train_files, val_files, test_files

    def recondition_mseeds(func_):
        print(func_)

    