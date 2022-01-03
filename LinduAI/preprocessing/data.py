#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Des 17 06:13:10 2021

@author: hakimbmkg
"""

import os
import json
from tqdm import tqdm
import pandas as pd
from obspy import UTCDateTime
from obspy.clients.fdsn.client import Client

class Data:
    """
    Class for preprocessing data
    """
    global client
    global directory

    client  = Client('https://geof.bmkg.go.id/',user='pgn',password='InfoPgn!&#')  #jangan lupa dihapus
    directory = os.getcwd()
    
    def make_station(net, dayStart, dayEnd, min_lat, max_lat, min_lng, max_lng, **kwargs):
        """
        Function for make station list
        """

        if not os.path.exists(directory+'/input/station/'):
            os.makedirs(directory+'/input/station')
            print('== folder /input/station/ created')

        if os.path.exists(directory+'/input/station/stations.json'):
            os.remove(directory+'/input/station/stations.json')

        start_time  = UTCDateTime(dayStart)
        end_time    = UTCDateTime(dayEnd)

        inv = client.get_stations(
            network=net, level='channel',
            starttime=start_time, endtime=end_time,
            minlatitude=min_lat, maxlatitude=max_lat,
            minlongitude=min_lng, maxlongitude=max_lng, **kwargs)

        stations = {}
        for inv_ in tqdm(inv):
            network = inv_.code
            for sta in inv_:
                station          = sta.code
                latitude_        = sta.latitude
                longitude_       = sta.longitude
                elvation_        = sta.elevation
                stations[str(station)] = {
                    "network"   : network,
                    "channels"  : ['BHE','BHN','BHZ','SHE','SHN','SHZ'],
                    "coords"    : [latitude_, longitude_, elvation_]
                }
        
        folder_path = directory+'/input/station/'
        with open(folder_path+'stations.json', 'w') as file_:
            json.dump(stations, file_)
        print(f'process completed, stations.json Created')

    def download_waveform_fromcsv(path):
        """
        function for download waveform from csv file
        """
        if not os.path.exists(directory+'/input/waveform/'):
            os.makedirs(directory+'/input/waveform')
            print('== folder /input/waveform/ created')
        
        if not os.path.exists(directory+'/input/labels_csv/'):
            os.makedirs(directory+'/input/labels_csv')
            print('== folder /input/labels_csv/ created')

        if not os.path.exists(directory+'/input/tmp_files/'):
            os.makedirs(directory+'/input/tmp_files')
            print('== folder /input/tmp_files/ created')

        if os.path.exists(directory+'/input/tmp_files/tmp_inaseisform.csv'):
            os.remove(directory+'/input/tmp_files/tmp_inaseisform.csv')

        read_csv = pd.read_csv(directory+'/'+path)
        init_stations = read_csv[['start_time','end_time','network','code','classID','class']]
        arr_init_stations = init_stations.to_numpy()
        xx = len(arr_init_stations)

        df = pd.DataFrame(columns=[
            'files_name',
            'start_time',
            'end_time',
            'network',
            'stations_code',
            'classID',
            'class'
        ])
        df.to_csv(directory+'/input/tmp_files/tmp_inaseisform.csv', mode='a', header=True, index=False)
        
        for num_,i in enumerate(tqdm(arr_init_stations, desc='Process')):
            xx = str(num_)
            try:
                time_   = (str(i[1].replace('-',''))+str(i[0].replace('-','')))
                t1  = UTCDateTime(i[0])
                t2  = UTCDateTime(i[1])
                st  = client.get_waveforms(i[2],i[3],'*','BH?',t1,t2)
                fn  = str(i[2]+'.'+i[3]+'.'+time_+xx)
                # print(fn)
                # print(st)
            except:
                try:
                    time_   = (str(i[1].replace('-',''))+str(i[0].replace('-','')))
                    t1  = UTCDateTime(i[0])
                    t2  = UTCDateTime(i[1])
                    st  = client.get_waveforms(i[2],i[3],'*','SH?',t1,t2)
                    fn  = str(i[2]+'.'+i[3]+'.'+time_+xx)
                    # print(fn)
                    # print(st)
                except:
                    st = 'NODATA'
                    print(f'***!warning!*** >> NoDATA for station {i[3]}')
                    print(f'*******************************************')

            if st == 'NODATA':
                print(f'***!warning!*** >> NoDATA for station {i[3]}')
                print(f'===========================================')
            else:
                print(f'\nData Available for station {i[3]}')
                savefile_path = str(directory+'/input/waveform/'+fn)
                try:
                    if not os.path.exists(directory+'/input/waveform/'+fn):
                        st.write(savefile_path, format='MSEED')
                        print(f'Success write mseed file for station {i[3]}')
                        print(f'===========================================')
                    else:
                        print(f'file mseed for station {i[3]} is exist')
                        print(f'===========================================')

                    data_ = {
                        'files_name'        : [fn],
                        'start_time'        : [t1],
                        'end_time'          : [t2],
                        'network'           : [i[2]],
                        'stations_code'     : [i[3]],
                        'classID'           : [i[4]],
                        'class'             : [i[5]]
                    }
                    df = pd.DataFrame(data_)
                    df.to_csv(directory+'/input/tmp_files/tmp_inaseisform.csv', mode='a', header=False, index=False)

                except:
                    print(f'***!warning!*** >> cant write mseed for station {i[3]}')