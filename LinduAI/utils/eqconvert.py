#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  02 06:24:03 2022

@author: hakimbmkg
"""

from obspy.clients.fdsn.client import Client
from obspy.core.utcdatetime import UTCDateTime
from obspy import read
import pandas as pd
import numpy as np
import json
import os
from tqdm import tqdm
import concurrent.futures


class Eqconvert:
    """
    This is class for convert data from arrival format seiscomp and next step download by event from FDSN 
    """

    def created_station(net, dayStart, dayEnd, min_lat, max_lat, min_lng, max_lng, url,user=None,pawd=None):
        """
        Created Station Using Selected Region From FDSN
        This param is : \n
        net         : [str] network of station
        sta         : [str] name of station
        dayStart    : [day format - YYYY-MM-DDTHH:MM:SS] start time
        dayEnd      : [day format - YYYY-MM-DDTHH:MM:SS] end time
        min_lat     : [float] - min of latitude
        max_lat     : [float] - max of latitude
        min_lng     : [float] - min of longitude
        max_lng     : [float] - max of longitude
        url         : [str] url of FDSN
        user        : [str] default None
        pawd        : [str] default None

        """
        fd = os.getcwd()
        if not os.path.exists(fd+'/input/dataset_EQ/rawdata/'): 
            os.makedirs(fd+'/input/dataset_EQ/rawdata/')
            print('folder created')
        else:
            print('folder is exist')

        if not os.path.exists(fd+'/input/dataset_EQ/rawdata/station/'):
            os.makedirs(fd+'/input/dataset_EQ/rawdata/station/')
            print('folder created')
        else:
            print('folder is exist')

        url_ = str(url)
        usr_ = str(user)
        pwd_ = str(pawd)
        c = Client(url_,user=usr_,password=pwd_) 
        start_t = UTCDateTime(dayStart)
        end_t = UTCDateTime(dayEnd)

        inv = c.get_stations(
            network=net, level='channel',
            starttime=start_t, endtime=end_t,
            minlatitude=min_lat, maxlatitude=max_lat,
            minlongitude=min_lng, maxlongitude=max_lng)

        stations = {}
        for inv_ in inv:
            network = inv_.code
            for sta in inv_:
                station          = sta.code
                latitude_        = sta.latitude
                longitude_       = sta.longitude
                elvation_        = sta.elevation
                # channels_        = sta.channels[:1][0]
                # print (f'{str(network)} - {str(station)} ')
                stations[str(station)] = {
                    "network"   : network,
                    "channels"  : ['BHE','BHN','BHZ'],
                    "coords"    : [latitude_, longitude_, elvation_]
                 }
        fd_ = fd+'/input/dataset_EQ/rawdata/station/'
        with open(fd_+'stations.json', 'w') as file_:
            json.dump(stations, file_)

    def arrivalformat2csv (file_inp):

        """
        This is a function for reformat from arrival format seiscomp to CSV 
        Important note: the function has a bug, the result of reformat duplicate stations if have P and S time phase. 
        so, the solution you can remove duplicates using pandas module to remove duplicates. \n

        header of DataFrame/CSV reffer from STEAD, please follow :
        Mousavi, S. M., Sheng, Y., Zhu, W., Beroza G.C., (2019). 
        STanford EArthquake Dataset (STEAD): A Global Data Set of Seismic Signals for AI, IEEE Access, 
        doi:10.1109/ACCESS.2019.2947848
        https://github.com/smousavi05/STEAD
        """

        path_fd = os.getcwd()

        if not os.path.exists(path_fd+"/input/"):
            os.makedirs(path_fd+'/input')
            print('folder input created')

        if not os.path.exists(path_fd+"/input/dataset_EQ/"):
            os.makedirs(path_fd+"/input/dataset_EQ/")
            print('folder dataset_EQ created')
            
        if os.path.exists(path_fd+"/input/dataset_EQ/merge.csv"):
            os.remove(path_fd+"/input/dataset_EQ/merge.csv")

        df = pd.DataFrame(columns= [
            'network_code',
            'receiver_code',
            'receiver_type',
            'receiver_latitude',
            'receiver_longitude',
            'receiver_elevation_m',
            'p_arrival_sample',
            'p_status',
            'p_weight',
            'p_travel_sec',
            's_arrival_sample',
            's_status',
            's_weight',
            'source_id',
            'date_',
            'source_origin_time',
            'source_origin_uncertainty_sec',
            'source_latitude',
            'source_longitude',
            'source_error_sec',
            'source_gap_deg',
            'source_horizontal_uncertainty_km',
            'source_depth_km',
            'source_depth_uncertainty_km',
            'source_magnitude',
            'source_magnitude_type',
            'source_magnitude_author',
            'source_mechanism_strike_dip_rake',
            'source_distance_deg',
            'source_distance_km',
            'back_azimuth_deg',
            'snr_db','coda_end_sample',
            'trace_start_time',
            'trace_category',
            'trace_name'])

        df.to_csv(path_fd+'/input/dataset_EQ/merge.csv',mode='a', header=True, index=False)


        fd = open(path_fd+'/'+file_inp,'r')
        l_fd = fd.readlines()
        for i in range(len(l_fd)):
            l_fd[i] = l_fd[i].split()

        pbar = tqdm(desc='convert arr2csv', total=len(l_fd))
        i = 0
        while i < len(l_fd):
            # time.sleep(0.1)
            if len(l_fd[i])>2 and l_fd[i][2][:3] == 'bmg':
                idEvent = l_fd[i][2]

            if len(l_fd[i])>0 and l_fd[i][0] == 'Origin:':
                datex   = l_fd[i+2][1]
                timex   = l_fd[i+3][1]
                lat     = l_fd[i+4][1]
                lng     = l_fd[i+5][1]
                depth   = l_fd[i+6][1]
                auth    = l_fd[i+7][1]
                mode_   = l_fd[i+9][1]
                azgap   = l_fd[i+13][2]
                date_   = (str(datex.replace('-','')))
                time_   = (str(timex.replace(':','')))

            if len(l_fd[i])>3 and l_fd[i][-2]=='preferred':
                m     = l_fd[i][1]
                m_typ = l_fd[i][0]

            if len(l_fd[i])>4 and l_fd[i][4]=='phase':
                try:
                    k=0
                    while k<1:
                        i += 1
                        try:
                            net  = l_fd[i][1]
                            stas = l_fd[i][0]
                            phase= l_fd[i][4]
                            p_tim = ''
                            s_tim = ''
                            if phase == 'P' :    
                                p_tim= l_fd[i][5]

                                if l_fd[i+1][4]  == 'S':
                                    s_tim= l_fd[i+1][5]

                            pick_p = p_tim
                            pick_s = s_tim
                            # get detail from json station
                            station_file = open(path_fd+'/input/dataset_EQ/rawdata/station/stations.json')
                            station_load = json.load(station_file)
                            for sta in station_load:
                                sta_name = sta
                                if sta_name == stas:
                                    lat_sta = station_load[str(sta)]['coords'][0]
                                    lng_sta = station_load[str(sta)]['coords'][1]
                                    elev_sta= station_load[str(sta)]['coords'][2] 

                            lt_st = lat_sta
                            lg_st = lng_sta
                            el_st = elev_sta

                            data_ = {
                                'network_code'                  :   [net],
                                'receiver_code'                 :   [stas],
                                'receiver_type'                 :   ['BHZ'],
                                'receiver_latitude'             :   [lt_st],
                                'receiver_longitude'            :   [lg_st],
                                'receiver_elevation_m'          :   [el_st],
                                'p_arrival_sample'              :   [pick_p],
                                'p_status'                      :   [mode_],
                                'p_weight'                      :   ['None'],
                                'p_travel_sec'                  :   ['None'],
                                's_arrival_sample'              :   [pick_s],
                                's_status'                      :   [mode_],
                                's_weight'                      :   ['None'],
                                'source_id'                     :   [idEvent],
                                'date_'                         :   [datex],
                                'source_origin_time'            :   [datex +' '+timex],
                                'source_origin_uncertainty_sec' :   ['None'],
                                'source_latitude'               :   [lat],
                                'source_longitude'              :   [lng],
                                'source_error_sec'              :   ['None'],
                                'source_gap_deg'                :   [azgap],
                                'source_horizontal_uncertainty_km': ['None'],
                                'source_depth_km'               :   [depth],
                                'source_depth_uncertainty_km'   :   ['None'],
                                'source_magnitude'              :   [m],
                                'source_magnitude_type'         :   [m_typ],
                                'source_magnitude_author'       :   [auth],
                                'source_mechanism_strike_dip_rake': ['None'],
                                'source_distance_deg'           :   ['None'],
                                'source_distance_km'            :   ['None'],
                                'back_azimuth_deg'              :   ['None'],
                                'snr_db'                        :   ['None'],
                                'coda_end_sample'               :   ['None'],
                                'trace_start_time'              :   ['None'],
                                'trace_category'                :   ['None'],
                                'trace_name'                    :   [stas+'.'+net+'_'+date_+time_+'_EV']
                            }
                            df = pd.DataFrame(data_)
                            df.to_csv(path_fd+'/input/dataset_EQ/merge.csv',mode='a', header=False, index=False)

                            pbar.update()
                        except:
                            break
                except:
                    break
            i += 1
    
        Eqconvert.csvcleaning('/input/dataset_EQ/merge.csv')

    def csvcleaning(csv_path):
        path_fd = os.getcwd()
        if not os.path.exists(path_fd+"/input/dataset_EQ"):
            os.makedirs(path_fd+'/input/dataset_EQ')
        if not os.path.exists(path_fd+"/input/dataset_EQ/event"):
            os.makedirs(path_fd+'/input/dataset_EQ/event')
        if not os.path.exists(path_fd+"/input/dataset_EQ/noise"):
            os.makedirs(path_fd+'/input/dataset_EQ/noise')
            print('folder dataset created')

        if os.path.exists(path_fd+"/input/dataset_EQ/merge_clear.csv"):
            os.remove(path_fd+"/input/dataset_EQ/merge_clear.csv")
        path_fd = os.getcwd()
        df = pd.read_csv(path_fd+'/'+csv_path)
        df.dropna(subset=['p_arrival_sample'],inplace=True)
        df.to_csv(path_fd+'/input/dataset_EQ/merge_clear.csv',mode='a', header=True, index=False)
        print (df)

    def downloadseedbycsv(csv_path, url,user=None, pawd=None, n_cpu = os.cpu_count()):
        """
        function for download mseed from FDSN
        This param is : \n
        csv_path    : /your/path/csv
        url         : [str] url of FDSN
        user        : [str] default None
        pawd        : [str] default None
        """
        url_ = str(url)
        usr_ = str(user)
        pwd_ = str(pawd)
        c = Client(url_,user=usr_,password=pwd_) 

        path_fd = os.getcwd()
        df = pd.read_csv(path_fd+'/'+csv_path)
        a = df[['network_code','receiver_code','p_arrival_sample','date_','trace_name']]
        
        #parallel init
        split_df = np.array_split(a,n_cpu)
        df_results = []
        
        def work(x):
            df = x
            arr_df = df.to_numpy()
            for i in arr_df:
                try:
                    t1 = UTCDateTime(i[3]+'T'+i[2])
                    t2 = t1 + 180
                    st = c.get_waveforms(i[0],i[1],'00','BHZ',t1,t2)
                    fn = i[4]
                    # print(st)
                except:
                    try:
                        t1 = UTCDateTime(i[3]+'T'+i[2])
                        t2 = t1 + 180
                        st = c.get_waveforms(i[0],i[1],'01','BHZ',t1,t2)
                        fn = i[4]
                        # print(st)
                    except:
                        try:
                            t1 = UTCDateTime(i[3]+'T'+i[2])
                            t2 = t1 + 180
                            st = c.get_waveforms(i[0],i[1],'*','BHZ',t1,t2)
                            fn = i[4]
                            # print(st)
                        except:
                            print(f'***!warning!*** >> station '+i[1]+' - '+ i[4]+' is null')
                            st = 'null'
                            df.drop(df.loc[df['trace_name']==i[4]].index, inplace=True)
                            #df.to_csv(path_fd+'/input/dataset_EQ/x_merge_stream.csv', header=True, index=False)
                            # print (df)

                if st != 'null':
                    file_path = str(path_fd+'/input/dataset_EQ/event/'+fn)
                    try:
                        if not os.path.isfile(file_path):
                            st.write(file_path, format='MSEED')
                            print(f'==> mseed write for station '+ i[0] +'_'+ i[1])
                    except:
                        print(f'## error write file ##')
            return(df)

        #parallel run
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_cpu) as executor:
            results = [ executor.submit(work,x=df) for df in split_df ]
            for z in concurrent.futures.as_completed(results):
                try:
                    df_results.append(z.result())
                except Exception as ex:
                    print(str(ex))
                    pass
        
        #output processing
        df_results = pd.concat(df_results)
        df_results['marker'] = 1
        joined = pd.merge(df,df_results, on = ['network_code','receiver_code','p_arrival_sample','date_','trace_name'], how='left')
        joined = joined[~pd.isnull(joined['marker'])]
        joined.drop(labels = 'marker', axis = 1,inplace=True)
        joined.to_csv(path_fd+'/input/dataset_EQ/merge_stream.csv', header=True, index=False)
        print('***download finish***')
        
    def checkwaveform(path_wave):
        st = read(path_wave)
        st.plot()


    
