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
from datetime import datetime as dt
import concurrent.futures
from pathlib import Path


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
    
    def _getstation_(net, dayStart, dayEnd, min_lat, max_lat, min_lon, max_lon, url,user=None,pawd=None, dataset_path=os.getcwd(), station_filename=None):
        """
        Create station database in JSON Format from connected FDSN based on location and time period
        Parameters      : \n
        net             : [str] network of station
        sta             : [str] name of station
        dayStart        : [day format - YYYY-MM-DDTHH:MM:SS] start time
        dayEnd          : [day format - YYYY-MM-DDTHH:MM:SS] end time
        min_lat         : [float] - min of latitude
        max_lat         : [float] - max of latitude
        min_lon         : [float] - min of longitude
        max_lon         : [float] - max of longitude
        url             : [str] url of FDSN
        user            : [str] FDSN username, default None
        pawd            : [str] FDSN password, default None
        dataset_path    : [str] absolute path to dataset directory
        """

        if not os.path.exists(os.path.join(dataset_path,"station")):
            os.makedirs(dataset_path+'/station')
            print(f'folder station created at {dataset_path}')
        else:
            print(f'folder {dataset_path}/station exist')
            
        # connect to client 
        client = Client(str(url),user=str(user),password=str(pawd))
        
        # downloading inventory based on query
        inventory = client.get_stations(
            network=net, level='channel',
            starttime=UTCDateTime(dayStart), endtime=UTCDateTime(dayEnd),
            minlatitude=min_lat, maxlatitude=max_lat,
            minlongitude=min_lon, maxlongitude=max_lon)
        
        # processing inventory data to json
        stations = {}
        for inv in inventory:
            net_ = inv.code
            for st in inv:
                station_ = st.code
                # print(str(net_)+"--"+str(station_))
                
                elev_ = st.elevation
                lat_ = st.latitude
                lon_ = st.longitude
                all_channel = [ch.code for ch in st.channels]
                
                # channel priority SH[ENZ] - SH[12Z] - BH[ENZ] - BH[12Z] - HH[ENZ] - HH[12Z] - none
                if len(list(set(filter(lambda x: (x[0:2]=='SH'),all_channel)))) > 0:
                    if len(list(set(filter(lambda x: (x[0:2]=='SH') and x[2:].isalpha(),all_channel)))) == 3:
                        ch_ = list(set(filter(lambda x: (x[0:2]=='SH') and x[2:].isalpha(),all_channel)))
                    else:
                        ch_ = list(set(filter(lambda x: x[0:2]=='SH',all_channel)))
                    ch_.sort()
                elif len(list(set(filter(lambda x: x[0:2]=='BH',all_channel)))) > 0:
                    if len(list(set(filter(lambda x: (x[0:2]=='BH') and x[2:].isalpha(),all_channel)))) == 3:
                        ch_ = list(set(filter(lambda x: (x[0:2]=='BH') and x[2:].isalpha(),all_channel)))
                    else:
                        ch_ = list(set(filter(lambda x: x[0:2]=='BH',all_channel)))
                    ch_.sort()
                elif len(list(set(filter(lambda x: x[0:2]=='HH',all_channel)))) > 0:
                    if len(list(set(filter(lambda x: (x[0:2]=='HH') and x[2:].isalpha(),all_channel)))) == 3:
                        ch_ = list(set(filter(lambda x: (x[0:2]=='HH') and x[2:].isalpha(),all_channel)))
                    else:
                        ch_ = list(set(filter(lambda x: x[0:2]=='HH',all_channel)))
                    ch_.sort()
                else:
                    ch_= []
                
                if len(ch_) != 0:
                    stations[str(station_)] = {
                        "network"   : net_,
                        "channels"  : list(set(ch_)),
                        "coords"    : [lat_,lon_,elev_]
                    }
        
        # writing files
        if station_filename:
            with open(os.path.join(dataset_path,'station',station_filename), 'w') as file_:
                json.dump(stations, file_)
        else:
            with open(os.path.join(dataset_path,'station','stations.json'), 'w') as file_:
                json.dump(stations, file_)
               
    def _arrivalconvert_(dataset_path,arrival_fname,format="stead"):
        # inner function
        def _change_trigger(bool_d, name):
            bool_d = bool_d.fromkeys(bool_d,False)
            bool_d[name] = True
            return(bool_d)
        def _cdate(param):
            return(dt.strptime(param,"%Y-%m-%d"))
        def _ctime(param):
            return(dt.strptime(param,"%H:%M:%S.%f"))
        
        if not os.path.exists(os.path.join(dataset_path,'arrival')):
            print("no arrival folder found in dataset_path")
            
        path = os.path.join(dataset_path,'arrival')

        # check if output files with name exist:
        if os.path.isfile(os.path.join(path,Path(arrival_fname).stem+".csv")) and format == "stead":
            print(f"output file exist! \nplease delete {os.path.join(path,Path(arrival_fname).stem+'.csv')}")
            # exit()
        if os.path.isfile(os.path.join(path,Path(arrival_fname).stem+".dat")) and format == "pha":
            print(f"output file exist! \nplease delete {os.path.join(path,Path(arrival_fname).stem+'.dat')}")
            # exit()
        
        # read station
        try:
            station_dict = json.load(open(os.path.join(dataset_path,'station','stations.json')))
        except:
            print(f"station data not available: {os.path.join(dataset_path,'station','stations.json')}")
            # exit()
        # define dataframe columns for STEAD format
        if format == "stead":
            # adding s_travel_sec 
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
                        's_travel_sec',
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
                        'snr_db',
                        'coda_end_sample',
                        'trace_start_time',
                        'trace_category',
                        'trace_name'])
            columns_ = df.columns
            df.to_csv(os.path.join(path,Path(arrival_fname).stem+".csv"),mode='a', header=True, index=False)
        # define outputfile for pha format
        elif format == "pha":
            df = pd.DataFrame(columns= [
                            'flag',
                            'year',
                            'month',
                            'day',
                            'hour',
                            'minutes',
                            'seconds',
                            'latitude',
                            'longitude',
                            'depth',
                            'magnitude',
                            'horizontal_error',
                            'depth_error',
                            'RMS',
                            'ID',])
        else:
            print(f"output file format not supported --> {format}")
            # exit()
        
        # start read and convert
        # initiate trigger bool and trigger counter
        trigger_d = {
            "event" : False,
            "origin" : False,
            "netmag" : False,
            "phase" : False,
            "stamag" : False,
            }
        trigger_counter = 0
        event_counter = 1

        # run
        fin = open(os.path.join(path,arrival_fname),'r')
        lines = fin.readlines()
        for i, line in enumerate(tqdm(lines)):
            # skip header
            if "for 1 event:" in line.lower():
                continue
            # init trigger
            if "event:" in line.lower():
                trigger_d = _change_trigger(trigger_d,"event")
                # print(f"event in line {i}")
                trigger_counter += 1
            elif "origin:" in line.lower():
                trigger_d = _change_trigger(trigger_d,"origin")
                # print(f"origin in line {i}")
                trigger_counter += 1
            elif "network magnitudes:" in line.lower():
                trigger_d = _change_trigger(trigger_d,"netmag")
                # print(f"network magnitudes in line {i}")
                trigger_counter += 1
            elif "phase arrivals:" in line.lower():
                trigger_d = _change_trigger(trigger_d,"phase")
                # print(f"phase arrivals in line {i}")
                trigger_counter += 1
            elif "station magnitudes:" in line.lower():
                trigger_d = _change_trigger(trigger_d,"stamag")
                # print(f"station magnitudes in line {i}")
                trigger_counter += 1
            else:
                pass
            
            # event block reading
            if "public id" in line.lower() and trigger_d['event']:
                eventid_ = line.split()[2]
            
            # origin block reading
            if trigger_d['origin']:
                if ("date" in line.lower()):
                    date_ = _cdate(line.split()[1])
                elif ("time" in line.lower()) and ("creation" not in line.lower()):
                    time_uncertainty_sec_ = line.split()[-2]
                    time_ = _ctime(line.split()[1])
                    datetime_ = dt.combine(date_.date(),time_.time())
                elif "latitude" in line.lower():
                    lat_ = line.split()[1]
                    lat_uncertainty_km_ = line.split()[-2]
                elif "longitude" in line.lower():
                    lon_ = line.split()[1]
                    lon_uncertainty_km_ = line.split()[-2]
                elif "depth" in line.lower():
                    depth_ = line.split()[1]
                    if "fixed" in line.lower():
                        depth_uncertainty_km_ = 0
                    else:
                        depth_uncertainty_km_ = line.split()[-2]
                elif "azimuthal gap" in line.lower():
                    azgap_ = line.split()[2]
                elif "residual rms" in line.lower():
                    source_error_sec_ = line.split()[-2]
                elif "author" in line.lower():
                    author_ = line.split()[-1]
                elif "mode" in line.lower():
                    mode_ = line.split()[-1]
                elif 'lat_uncertainty_km_' and 'lon_uncertainty_km_' in locals():
                    source_horizontal_uncertainty_km_ = max(lat_uncertainty_km_,lon_uncertainty_km_)
                else:
                    pass
            
            # network magnitude block reading
            if trigger_d['netmag']:
                if "preferred" in line.lower():
                    source_mag_ = line.split()[1]
                    source_mag_type_ = line.split()[0]
            
            # phase arrival block reading + writing output format
            if trigger_d['phase']:
                if len(line.split()) > 3 and "sta" in line.lower() and format == "pha":
                    #pha header write
                    phaheader_ = {
                            'flag'                 :   ["#"], ### done ###
                            'year'                 :   [date_.year], ### done ###
                            'month'                :   [date_.month], ### done ###
                            'day'                  :   [date_.day], ### done ###
                            'hour'                 :   [time_.hour], ### done ###
                            'minutes'              :   [time_.minute], ### done ###
                            'seconds'              :   [time_.strftime("%S.%f")], ### done ###
                            'latitude'             :   [lat_], ### done ###
                            'longitude'            :   [lon_], ### done ###
                            'depth'                :   [depth_], ### done ###
                            'magnitude'            :   [source_mag_], ### done ###
                            'horizontal_error'     :   [source_horizontal_uncertainty_km_], ### done ###
                            'depth_error'          :   [depth_uncertainty_km_], ### done ###
                            'RMS'                  :   [source_error_sec_], ### done ###
                            'ID'                   :   [event_counter], ### done ###
                            }
                    df = pd.concat([df,pd.DataFrame(phaheader_)])
                    
                elif len(line.split()) > 3 and ("sta" not in line.lower()) and ("missing" not in line.lower()):    
                    
                    net_ = line.split()[1]
                    sta_ = line.split()[0]
                    source_dist_deg_ = line.split()[2]
                    source_dist_km_ = float(source_dist_deg_) * 111.11
                    azimuth_ = line.split()[3]
                    
                    # station info
                    if sta_ in station_dict:
                        receiver_type_ = str(station_dict[sta_]['channels'][0][:2])
                        lat_sta_ = float(station_dict[sta_]['coords'][0])
                        lon_sta_ = float(station_dict[sta_]['coords'][1])
                        elev_sta_ = float(station_dict[sta_]['coords'][2])
                    else: # if station is not available in station data 
                        receiver_type_ = None
                        lat_sta_ = None
                        lon_sta_ = None
                        elev_sta_ = None
                    
                    # phase info
                    pha = line.split()[4]
                    if pha == 'P':
                        p_arrival_sample_ = line.split()[5]
                        p_weight_ = line.split()[-2]
                        p_travel_sec_ = abs(_ctime(p_arrival_sample_) - time_).total_seconds()
                        p_status_ = mode_
                        s_arrival_sample_ = None
                        s_weight_ = None
                        s_travel_sec_ = None
                        s_status_ = None
                        tt_pha = p_travel_sec_
                    elif pha == 'S':
                        p_arrival_sample_ = None
                        p_weight_ = None
                        p_travel_sec_ = None
                        p_status_ = None
                        s_arrival_sample_ = line.split()[5]
                        s_weight_ = line.split()[-2]
                        s_travel_sec_ = abs(_ctime(s_arrival_sample_) - time_).total_seconds()
                        s_status_ = mode_
                        tt_pha = s_travel_sec_
                    else:
                        p_arrival_sample_ = None
                        p_weight_ = None
                        p_travel_sec_ = None
                        p_status_ = None
                        s_arrival_sample_ = None
                        s_weight_ = None
                        s_travel_sec_ = None
                        s_status_ = None
                        tt_pha = None
                    
                    # save data stead
                    if format == "stead":
                        data_ = {
                            'network_code'                  :   [net_], ### done ###
                            'receiver_code'                 :   [sta_], ### done ###
                            'receiver_type'                 :   [receiver_type_], ### done ###
                            'receiver_latitude'             :   [lat_sta_], ### done ###
                            'receiver_longitude'            :   [lon_sta_], ### done ###
                            'receiver_elevation_m'          :   [elev_sta_], ### done ###
                            'p_arrival_sample'              :   [p_arrival_sample_], ### done ###
                            'p_status'                      :   [p_status_], ### done ###
                            'p_weight'                      :   [p_weight_], ### done ###
                            'p_travel_sec'                  :   [p_travel_sec_], ### done ###
                            's_arrival_sample'              :   [s_arrival_sample_], ### done ###
                            's_status'                      :   [s_status_], ### done ###
                            's_weight'                      :   [s_weight_], ### done ###
                            's_travel_sec'                  :   [p_travel_sec_], ### done ###
                            'source_id'                     :   [eventid_], ### done ###
                            'date_'                         :   [date_.strftime("%Y/%m/%d")], ### done ###
                            'source_origin_time'            :   [datetime_.strftime("%Y/%m/%d %H:%M:%S.%f")], ### done ###
                            'source_origin_uncertainty_sec' :   [time_uncertainty_sec_], ### done ###
                            'source_latitude'               :   [lat_], ### done ###
                            'source_longitude'              :   [lon_], ### done ###
                            'source_error_sec'              :   [source_error_sec_], ### done ###
                            'source_gap_deg'                :   [azgap_], ### done ###
                            'source_horizontal_uncertainty_km': [source_horizontal_uncertainty_km_], ### done ###
                            'source_depth_km'               :   [depth_], ### done ###
                            'source_depth_uncertainty_km'   :   [depth_uncertainty_km_], ### done ###
                            'source_magnitude'              :   [source_mag_], ### done ###
                            'source_magnitude_type'         :   [source_mag_type_], ### done ###
                            'source_magnitude_author'       :   [author_], ### done ###
                            'source_mechanism_strike_dip_rake': ['None'],
                            'source_distance_deg'           :   [source_dist_deg_], ### done ###
                            'source_distance_km'            :   [source_dist_km_], ### done ###
                            'back_azimuth_deg'              :   [azimuth_],
                            'snr_db'                        :   ['None'],
                            'coda_end_sample'               :   ['None'],
                            'trace_start_time'              :   ['None'],
                            'trace_category'                :   ['None'],
                            'trace_name'                    :   [sta_+'.'+net_+'_'+datetime_.strftime("%Y%m%d%H%M%S.%f")+'_EV']
                            }
                        df = pd.concat([df,pd.DataFrame(data_)])
                    
                    # save data pha  
                    elif format == "pha":
                        phadata_ = {
                            'flag'                 :   [sta_], ### changing data to STA ###
                            'year'                 :   [tt_pha], ### changing data to TT ###
                            'month'                :   ['1'], ### changing data to WEIGHT ### # default value #
                            'day'                  :   [pha], ### changing data to PHA TYPE ###
                            'hour'                 :   [None], ### changing data to None After this ###
                            'minutes'              :   [None], 
                            'seconds'              :   [None], 
                            'latitude'             :   [None], 
                            'longitude'            :   [None], 
                            'depth'                :   [None],
                            'magnitude'            :   [None],
                            'horizontal_error'     :   [None], 
                            'depth_error'          :   [None], 
                            'RMS'                  :   [None], 
                            'ID'                   :   [None], 
                            }
                        df = pd.concat([df,pd.DataFrame(phadata_)])
                else:
                    pass
                
            # data saving
            if trigger_counter == 5:
                #save data stead
                if format == "stead":
                    df = df.replace('',np.nan).groupby('receiver_code', as_index=False).first().fillna('')
                    df[columns_].to_csv(os.path.join(path,Path(arrival_fname).stem+".csv"),mode='a', header=False, index=False)
                    # pruge df
                    df = df[0:0]
                #save data pha
                else:
                    df.reset_index(inplace=True,drop=True)
                    Eqconvert._df2dat(df,evnum=0,path=path,fname=Path(arrival_fname).stem+'.dat', mode = 'a', verbose=False)
                    # pruge df
                    df = df[0:0]    
                # print(f"processing {eventid_}")
                event_counter += 1
                trigger_counter = 0
        # return 0
    
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

    def downloadseedbycsv(dataset_path,csv_filename,url,data_type='event',user=None, pawd=None, dl_channel="all", n_cpu = os.cpu_count()/2,time_before_p = 30, time_after_p = 120): 
        """
        function for download mseed from FDSN
        Parameters      : \n
        dataset_path    : absolute path of dataset folder
        csv_filename    : [str] csv filename with extension
        url             : [str] url of FDSN
        user            : [str] default None
        pawd            : [str] default None
        dl_channel      : [str] channel to be downloaded - default all, could be 'all','vertical', or 'horizontal'
        """
        url_ = str(url)
        usr_ = str(user)
        pwd_ = str(pawd)
        client = Client(url_,user=usr_,password=pwd_) 

        path = os.path.join(dataset_path,'waveform',data_type)
        
        if not os.path.exists(path):
            os.mkdir(path)
        else:
            print(f"{path} exist")
        df = pd.read_csv(os.path.join(dataset_path,'arrival',csv_filename), low_memory=False)
        a = df[['network_code','receiver_code','receiver_type','p_arrival_sample','date_','trace_name']]
        
        # load station json
        station_dict = json.load(open(os.path.join(dataset_path,'station','stations.json')))
        station_dict2 = json.load(open(os.path.join(dataset_path,'station','stations_2009_2015.json')))
        
        # parallel init
        split_df = np.array_split(a,n_cpu)
        df_results = []
        
        def work(df,dl_channel):
            arr_df = df.to_numpy()
            for i in arr_df:
                if dl_channel == 'vertical':
                    try:
                        channel_string = station_dict[i[1]]['channels'][0]
                    except:
                        try:
                            channel_string = station_dict2[i[1]]['channels'][0]
                        except:
                            print(f'station {i[1]} not found in databases')
                elif dl_channel == 'horizontal':
                    try:
                        channel_string = ",".join(station_dict[i[1]]['channels'][1:3])
                    except:
                        try:
                            channel_string = ",".join(station_dict2[i[1]]['channels'][1:3])
                        except:
                            print(f'station {i[1]} not found in databases')
                elif dl_channel == 'all':
                    try:
                        channel_string = ",".join(station_dict[i[1]]['channels'])
                    except:
                        try:
                            channel_string = ",".join(station_dict2[i[1]]['channels'])
                        except:
                            print(f'station {i[1]} not found in databases')
                else:
                    channel_string = dl_channel
                    
                try:
                    t1 = UTCDateTime(i[4]+'T'+i[3])-time_before_p
                    t2 = t1+time_after_p
                    st = client.get_waveforms(i[0],i[1],'00',channel_string,t1,t2)
                    fn = i[5]
                except:
                    try:
                        t1 = UTCDateTime(i[4]+'T'+i[3])-time_before_p
                        t2 = t1+time_after_p
                        st = client.get_waveforms(i[0],i[1],'01',channel_string,t1,t2)
                        fn = i[5]
                    except:
                        try:
                            t1 = UTCDateTime(i[4]+'T'+i[3])-time_before_p
                            t2 = t1+time_after_p
                            st = client.get_waveforms(i[0],i[1],'10',channel_string,t1,t2)
                            fn = i[5]
                        except:
                            try:
                                t1 = UTCDateTime(i[4]+'T'+i[3])-time_before_p
                                t2 = t1+time_after_p
                                st = client.get_waveforms(i[0],i[1],'*',channel_string,t1,t2)
                                fn = i[5]
                            except:
                                print(f'***!warning!*** >> station '+i[1]+' - '+ i[5]+' is null')
                                st = 'null'
                                df.drop(df.loc[df['trace_name']==i[5]].index, inplace=True)


                if st != 'null':
                    file_path = os.path.join(path,fn)
                    try:
                        if not os.path.isfile(file_path):
                            st.write(file_path, format='MSEED')
                            print(f'==> mseed write for station '+ i[0] +'_'+ i[1])
                    except:
                        print(f'## error write file ##')
            return(df)

        # parallel run
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_cpu) as executor:
            results = [ executor.submit(work,df=df,dl_channel=dl_channel) for df in split_df ]
            for z in concurrent.futures.as_completed(results):
                try:
                    df_results.append(z.result())
                except Exception as ex:
                    print(str(ex))
                    pass
        
        # output processing
        df_results = pd.concat(df_results)
        df_results['marker'] = 1
        df_downloaded = pd.merge(df,df_results, on = ['network_code','receiver_code','receiver_type','p_arrival_sample','date_','trace_name'], how='left')
        # make df
        df_failed = df_downloaded[pd.isnull(df_downloaded['marker'])]
        df_downloaded = df_downloaded[~pd.isnull(df_downloaded['marker'])]
        # drop marker
        df_failed.drop(labels = 'marker', axis = 1,inplace=True)
        df_downloaded.drop(labels = 'marker', axis = 1,inplace=True)
        # save file
        df_failed.to_csv(os.path.join(os.path.dirname(path),f'{Path(csv_filename).stem}_failed_to_download_{dt.now().strftime("%d%m%y")}.csv'), header=True, index=False)
        df_downloaded.to_csv(os.path.join(os.path.dirname(path),f'{Path(csv_filename).stem}_downloaded_{dt.now().strftime("%d%m%y")}.csv'), header=True, index=False)
        print('***download finish***')
        
    def checkwaveform(path_wave):
        st = read(path_wave)
        st.plot()

    def _df2dat_(df,evnum=0,path=os.getcwd(),fname='output.dat',mode='w',azgap=False,absolute=False,verbose=True):
    
        #if evnum != 0 then remake event number by the evnum input as first event number
        if evnum != 0 and absolute == False:
            pd.options.mode.chained_assignment = None
            tempheader = df[df[df.columns[0]] == '#']
            tempdata = df[df[df.columns[0]] != '#']
            tempheader[tempheader.columns[-1]] = np.arange(evnum,evnum+len(tempheader),1)
            df = pd.concat([tempdata,tempheader])
            df.sort_index(inplace = True)
            df.reset_index(inplace = True, drop = True)
        else:
            pass
        
        idx = df[df[df.columns[0]] == '#']
        files = open(os.path.join(path,fname), str(mode), newline='\n')
        for i in range(len(df.index)):
            if i in idx.index:
                if azgap == True:
                    tempheader = "{} {: >4.0f} {: >2.0f} {: >2.0f} {: >2.0f} {: >2.0f} {: >4.2f} {: >8.4f} {: >8.4f} {: >3.0f} {: >4.2f} {: >3.0f} {: >3.0f} {: >6.3f} {: >6.0f} {: >6.2f}\n".\
                    format(str(df.iloc[i][0]),float(df.iloc[i][1]),float(df.iloc[i][2]),float(df.iloc[i][3])\
                    ,float((df.iloc[i][4])),float(df.iloc[i][5]),float(df.iloc[i][6]),float(df.iloc[i][7])\
                    ,float(df.iloc[i][8]),float(df.iloc[i][9]),float(df.iloc[i][10]),float(df.iloc[i][11])\
                    ,float(df.iloc[i][12]),float(df.iloc[i][13]),float(df.iloc[i][14]),float(df.iloc[i][15]))
                    files.write(tempheader)
                elif absolute == True:
                    tempheader = "{}  {}\n".format(str(df.iloc[i][0]),int(df.iloc[i][1]))
                    files.write(tempheader)
                else:
                    tempheader = "{} {: >4.0f} {: >2.0f} {: >2.0f} {: >2.0f} {: >2.0f} {: >4.2f} {: >8.4f} {: >8.4f} {: >3.0f} {: >4.2f} {: >3.0f} {: >3.0f} {: >6.3f} {: >6.0f}\n".\
                    format(str(df.iloc[i][0]),float(df.iloc[i][1]),float(df.iloc[i][2]),float(df.iloc[i][3])\
                    ,float((df.iloc[i][4])),float(df.iloc[i][5]),float(df.iloc[i][6]),float(df.iloc[i][7])\
                    ,float(df.iloc[i][8]),float(df.iloc[i][9]),float(df.iloc[i][10]),float(df.iloc[i][11])\
                    ,float(df.iloc[i][12]),float(df.iloc[i][13]),float(df.iloc[i][14]))
                    files.write(tempheader)
            else:
                if absolute == True:
                    tempdata = "{}  {:.6f}  {:.7f}  {}\n".\
                    format(str(df.iloc[i][0]),float(df.iloc[i][1]),float(df.iloc[i][2]),(df.iloc[i][3]))
                    files.write(tempdata)
                else: 
                    tempdata = "     {: <7}{: >7.2f}{: >6}{: >4} \n".\
                    format(str(df.iloc[i][0]),float(df.iloc[i][1]),float(df.iloc[i][2]),(df.iloc[i][3]))
                    files.write(tempdata)
        files.close()  
        if ~verbose:
            return 0
        else: 
            return print("Output finish: {} at {}".format(fname,path))
    
    def _update_csv_with_station_(dataset_path,csv_filename,station_path=None):
        # reading file
        df = pd.read_csv(os.path.join(dataset_path,'arrival',csv_filename),na_values="NaN")
        if station_path:
            station_dict = json.load(open(station_path))
        else:
            station_dict = json.load(open(os.path.join(dataset_path,'station','stations.json')))
        
        # select and import station data to csv
        for index, row in df[(df.receiver_type.isna())].iterrows():
            if row['receiver_code'] in station_dict.keys():
                df.at[index,"receiver_type"] = station_dict[row["receiver_code"]]['channels'][0][:2]
                df.at[index,"receiver_latitude"] = station_dict[row["receiver_code"]]["coords"][0]
                df.at[index,"receiver_longitude"] = station_dict[row["receiver_code"]]["coords"][1]
                df.at[index,"receiver_elevation_m"] = station_dict[row["receiver_code"]]["coords"][2]
            else:
                pass
        
        # saving files
        df.to_csv(os.path.join(dataset_path,'arrival',csv_filename), index=False)
        