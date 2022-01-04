#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 18 06:01:03 2021

@author: hakimbmkg
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
plt.rcParams['agg.path.chunksize'] = 1000000
from tqdm import tqdm
import numpy as np
from obspy import read
from obspy import UTCDateTime
import tensorflow as tf
import shutil

class Transform:
    """
    Class for preprocessing data step two, transform data waveform.
    """

    global directory
    directory = os.getcwd()

    def conditioning_waveform(path):
        """
        funtion for conditioning 3 channels, BHZ, BHN, BHE, SHZ, SHN, SHE
        resample to 100Hz for standardization all mseed
        """
        
        if os.path.exists(directory+'/input/labels_csv/inaseisform.csv'):
            os.remove(directory+'/input/labels_csv/inaseisform.csv')

        read_csv = pd.read_csv(directory+'/'+path)
        init_stations = read_csv[['files_name','start_time','end_time','network','stations_code','classID','class']]
        arr_init_stations = init_stations.to_numpy()

        df = pd.DataFrame(columns=[
            'files_name',
            'start_time',
            'end_time',
            'network',
            'stations_code',
            'classID',
            'class'
        ])
        df.to_csv(directory+'/input/labels_csv/inaseisform.csv', mode='a', header=True, index=False)
        
        for n,i in enumerate(tqdm(arr_init_stations, desc='Conditioning/Split Process')):
            st = read(directory+'/input/waveform/'+i[0])
            st_net  = st[0].stats['network']
            st_code = st[0].stats['station']
            st_loc  = st[0].stats['location']
            st_fn   = st_net+'.'+st_code+'.'+st_loc+'.'

            if st.select(id=st_fn+'BHN'):
                new_st = st.select(id=st_fn+'BHN').resample(100.0)
                # print(f'resample to 100Hz for station {i[4]}')
                # print(new_st)
                new_fnN = st_fn+'BHN.'+str(UTCDateTime(i[1]))+'__'+str(UTCDateTime(i[2]))
                data_ = {
                    'files_name'        : new_fnN,
                    'start_time'        : str(UTCDateTime(i[1])),
                    'end_time'          : str(UTCDateTime(i[2])),
                    'network'           : [i[3]],
                    'stations_code'     : [i[4]],
                    'classID'           : [i[5]],
                    'class'             : [i[6]]
                    }
                df = pd.DataFrame(data_)
                df.to_csv(directory+'/input/labels_csv/inaseisform.csv', mode='a', header=False, index=False)
                try:
                    if not os.path.exists(directory+'/input/waveform/'+new_fnN):
                        new_st.write(directory+'/input/waveform/'+new_fnN, format='MSEED')
                        print(f'Success write mseed file for station {i[4]}')
                        print(f'===========================================')
                    else:
                        print(f'file mseed for station {i[4]} is exist')
                        print(f'===========================================')
                except:
                    print(f'***!warning!*** >> cant write mseed for station {i[4]}')


            if st.select(id=st_fn+'BHE'):
                new_st = st.select(id=st_fn+'BHE').resample(100.0)
                new_fnE = st_fn+'BHE.'+str(UTCDateTime(i[1]))+'__'+str(UTCDateTime(i[2]))  
                data_ = {
                    'files_name'        : new_fnE,
                    'start_time'        : str(UTCDateTime(i[1])),
                    'end_time'          : str(UTCDateTime(i[2])),
                    'network'           : [i[3]],
                    'stations_code'     : [i[4]],
                    'classID'           : [i[5]],
                    'class'             : [i[6]]
                    }
                df = pd.DataFrame(data_)
                df.to_csv(directory+'/input/labels_csv/inaseisform.csv', mode='a', header=False, index=False)
                try:
                    if not os.path.exists(directory+'/input/waveform/'+new_fnE):
                        new_st.write(directory+'/input/waveform/'+new_fnE, format='MSEED')
                        print(f'Success write mseed file for station {i[4]}')
                        print(f'===========================================')
                    else:
                        print(f'file mseed for station {i[4]} is exist')
                        print(f'===========================================')
                except:
                    print(f'***!warning!*** >> cant write mseed for station {i[4]}')



            if st.select(id=st_fn+'BHZ'):
                new_st = st.select(id=st_fn+'BHZ').resample(100.0)
                new_fnZ = st_fn+'BHZ.'+str(UTCDateTime(i[1]))+'__'+str(UTCDateTime(i[2]))
                data_ = {
                    'files_name'        : new_fnZ,
                    'start_time'        : str(UTCDateTime(i[1])),
                    'end_time'          : str(UTCDateTime(i[2])),
                    'network'           : [i[3]],
                    'stations_code'     : [i[4]],
                    'classID'           : [i[5]],
                    'class'             : [i[6]]
                    }
                df = pd.DataFrame(data_)
                df.to_csv(directory+'/input/labels_csv/inaseisform.csv', mode='a', header=False, index=False)
                try:
                    if not os.path.exists(directory+'/input/waveform/'+new_fnZ):
                        new_st.write(directory+'/input/waveform/'+new_fnZ, format='MSEED')
                        print(f'Success write mseed file for station {i[4]}')
                        print(f'===========================================')
                    else:
                        print(f'file mseed for station {i[4]} is exist')
                        print(f'===========================================')
                except:
                    print(f'***!warning!*** >> cant write mseed for station {i[4]}')

                
            if st.select(id=st_fn+'SHN'):
                new_st = st.select(id=st_fn+'SHN').resample(100.0)
                new_fnSN = st_fn+'SHN.'+str(UTCDateTime(i[1]))+'__'+str(UTCDateTime(i[2]))
                data_ = {
                    'files_name'        : new_fnSN,
                    'start_time'        : str(UTCDateTime(i[1])),
                    'end_time'          : str(UTCDateTime(i[2])),
                    'network'           : [i[3]],
                    'stations_code'     : [i[4]],
                    'classID'           : [i[5]],
                    'class'             : [i[6]]
                    }
                df = pd.DataFrame(data_)
                df.to_csv(directory+'/input/labels_csv/inaseisform.csv', mode='a', header=False, index=False)
                try:
                    if not os.path.exists(directory+'/input/waveform/'+new_fnSN):
                        new_st.write(directory+'/input/waveform/'+new_fnSN, format='MSEED')
                        print(f'Success write mseed file for station {i[4]}')
                        print(f'===========================================')
                    else:
                        print(f'file mseed for station {i[4]} is exist')
                        print(f'===========================================')
                except:
                    print(f'***!warning!*** >> cant write mseed for station {i[4]}')

                
            if st.select(id=st_fn+'SHE'):
                new_st = st.select(id=st_fn+'SHE').resample(100.0)
                new_fnSE = st_fn+'SHE.'+str(UTCDateTime(i[1]))+'__'+str(UTCDateTime(i[2]))
                data_ = {
                    'files_name'        : new_fnSE,
                    'start_time'        : str(UTCDateTime(i[1])),
                    'end_time'          : str(UTCDateTime(i[2])),
                    'network'           : [i[3]],
                    'stations_code'     : [i[4]],
                    'classID'           : [i[5]],
                    'class'             : [i[6]]
                    }
                df = pd.DataFrame(data_)
                df.to_csv(directory+'/input/labels_csv/inaseisform.csv', mode='a', header=False, index=False)
                try:
                    if not os.path.exists(directory+'/input/waveform/'+new_fnSE):
                        new_st.write(directory+'/input/waveform/'+new_fnSE, format='MSEED')
                        print(f'Success write mseed file for station {i[4]}')
                        print(f'===========================================')
                    else:
                        print(f'file mseed for station {i[4]} is exist')
                        print(f'===========================================')
                except:
                    print(f'***!warning!*** >> cant write mseed for station {i[4]}')
                

            if st.select(id=st_fn+'SHZ'):
                new_st = st.select(id=st_fn+'SHZ').resample(100.0)
                new_fnSZ = st_fn+'SHZ.'+str(UTCDateTime(i[1]))+'__'+str(UTCDateTime(i[2]))

                data_ = {
                    'files_name'        : new_fnSZ,
                    'start_time'        : str(UTCDateTime(i[1])),
                    'end_time'          : str(UTCDateTime(i[2])),
                    'network'           : [i[3]],
                    'stations_code'     : [i[4]],
                    'classID'           : [i[5]],
                    'class'             : [i[6]]
                    }
                df = pd.DataFrame(data_)
                df.to_csv(directory+'/input/labels_csv/inaseisform.csv', mode='a', header=False, index=False)
                try:
                    if not os.path.exists(directory+'/input/waveform/'+new_fnSZ):
                        new_st.write(directory+'/input/waveform/'+new_fnSZ, format='MSEED')
                        print(f'Success write mseed file for station {i[4]}')
                        print(f'===========================================')
                    else:
                        print(f'file mseed for station {i[4]} is exist')
                        print(f'===========================================')
                # except:
                #     print(f'***!warning!*** >> cant write mseed for station {i[4]}')
                except i.Error as e:
                    print("line: {}, error: {}".format(n.line_num, e))
                except StopIteration:
                    break
        print (f'\n ***finished*** \n')
        # print("\n NdasQ Mumet Mas, dikongkon lopang loping wae ... ora mari mari ...!\n")

    def make_spectogram(path):
        if not os.path.exists(directory+'/input/spectogram/'):
            os.makedirs(directory+'/input/spectogram')
            print('== folder /input/spectogram/ created')

        read_csv = pd.read_csv(directory+'/'+path)
        init_stations = read_csv[['files_name','start_time','end_time','network','stations_code','classID','class']]
        arr_init_stations = init_stations.to_numpy()

        for i in arr_init_stations:
            st = read(directory+'/input/waveform/'+i[0])
            # print(st.__str__(extended=True))

            data = st[0].data.astype('float32')
            sr = int(st[0].stats.sampling_rate)
            max_points = int(st[0].stats.npts)
            offset = 0

            hop_length = 128
            n_fft = 256
            cmap = 'jet'
            bins_per_octave = 12
            auto_aspect = False
            y_axis = "linear"  # linear or log
            fmin = None
            fmax = 5.0

            # Librosa spectrogram
            D = librosa.amplitude_to_db(
                np.abs(librosa.stft(data, hop_length=hop_length, n_fft=n_fft)), ref=np.max)

            fig, ax = plt.subplots()

            img = librosa.display.specshow(D, y_axis=y_axis, sr=sr,
                                           hop_length=hop_length, x_axis='time', ax=ax, cmap=cmap, bins_per_octave=bins_per_octave,
                                           auto_aspect=auto_aspect)

            if fmin is not None:
                fmin0 = fmin
            else:
                fmin0 = 0

            if fmax is not None:
                fmax0 = fmax
            else:
                fmax0 = sr/2

            ax.set_ylim([fmin, fmax])
            fig.colorbar(img, ax=ax, format="%+2.f dB")
            plt.savefig(directory+'/input/spectogram/'+i[0]+'.png', bbox_inches='tight', dpi=300)
            plt.close()


    @tf.function
    def make_spectogram_mags(path):
        if not os.path.exists(directory+'/input/dataset_EQ/spectogram/'):
            os.makedirs(directory+'/input/dataset_EQ/spectogram')
            print('== folder /input/spectogram/dataset_EQ/ created')

        read_csv = pd.read_csv(directory+'/'+path)
        init_stations = read_csv[['trace_name','source_magnitude']]
        arr_init_stations = init_stations.to_numpy()

        for i in arr_init_stations:
            if os.path.exists(directory+'/input/dataset_EQ/event/'+i[0]):
                st = read(directory+'/input/dataset_EQ/event/'+i[0])
                # print(st.__str__(extended=True))

                data = st[0].data.astype('float32')
                sr = int(st[0].stats.sampling_rate)
                max_points = int(st[0].stats.npts)
                offset = 0

                hop_length = 128
                n_fft = 256
                cmap = 'jet'
                bins_per_octave = 12
                auto_aspect = False
                y_axis = "linear"  # linear or log
                fmin = None
                fmax = 5.0

                # Librosa spectrogram
                D = librosa.amplitude_to_db(
                    np.abs(librosa.stft(data, hop_length=hop_length, n_fft=n_fft)), ref=np.max)

                fig, ax = plt.subplots()

                img = librosa.display.specshow(D, y_axis=y_axis, sr=sr,
                                               hop_length=hop_length, x_axis='time', ax=ax, cmap=cmap, bins_per_octave=bins_per_octave,
                                               auto_aspect=auto_aspect)

                if fmin is not None:
                    fmin0 = fmin
                else:
                    fmin0 = 0

                if fmax is not None:
                    fmax0 = fmax
                else:
                    fmax0 = sr/2

                ax.set_ylim([fmin, fmax])
                fig.colorbar(img, ax=ax, format="%+2.f dB")
                plt.savefig(directory+'/input/dataset_EQ/spectogram/'+i[0]+'.png', bbox_inches='tight', dpi=300)
                plt.close()

                continue

    def cp_spectogram(path):
        if not os.path.exists(directory+'/input/dataset/'):
            os.makedirs(directory+'/input/dataset')
            print('== folder /input/dataset/ created')

        data_path           = pd.read_csv(path)
        data_path_class     = data_path[['files_name','class']]
        arr_data_path       = data_path_class.to_numpy()
        labels = data_path['class'].value_counts().index.tolist()

        for a,b in enumerate(labels):
            if not os.path.exists(directory+'/input/dataset_spectogram/'+b):
                os.makedirs(directory+'/input/dataset_spectogram/'+b)
                print(f'== folder /input/dataset/{b} created')

            for x in arr_data_path:
                if x[1] == b:
                    src_ = directory+'/input/spectogram/'+x[0]+'.png'
                    des_ = directory+'/input/dataset_spectogram/'+b+'/'
                    if not os.path.exists(directory+'/input/dataset_spectogram/'+b+'/'+x[0]):
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


    def cp_spectogram_mags(path): 
        if not os.path.exists(directory+'/input/dataset_EQ/datasEQ_spectogram/'):
            os.makedirs(directory+'/input/dataset_EQ/datasEQ_spectogram/')
            print('== folder /input/dataset_EQ/datasEQ_spectogram/ created')

        data_path           = pd.read_csv(path)
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

