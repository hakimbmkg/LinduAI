#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 17 06:24:41 2021

@author: hakimbmkg
"""

"""
note : LinduAI, this aplication for analysis Station Index and Magnitude Classification for prediction Magnitude
Step by Step
first we must download station or make station list (*case study if you want to magnitude classification)

"""
import os
from LinduAI.main.model import Models
from LinduAI.main.modelmag import Modelsmag
from LinduAI.preprocessing.data import Data
from LinduAI.preprocessing.transform import Transform
from LinduAI.main.datas import Datas
from LinduAI.utils.eqconvert import Eqconvert

# param
global directory
directory = os.getcwd()

Net       = "IA"
channel   = "BH?"
dayStart  = "2021-09-01"
dayEnd    = "2021-09-30"
min_lat   = "-12.42"
max_lat   = "8.39"
min_lng   = "92.93"
max_lng   = "143.88"
url       = "https://geof.bmkg.go.id/"
user      = "user"
passwd    = "password#"
n_epoch   = 15

# EQ Magnitude Classifications
#----------------------------
#----------------------------
# step 1 for created station json
# Eqconvert.created_station(Net,dayStart, dayEnd, min_lat, max_lat, min_lng, max_lng, url, user, passwd)
# step 2 for convert arrival format to csv for preparation dataset and labels
# Eqconvert.arrivalformat2csv('arrival_example.txt')
# step 3 download mseed from fdsn by csv  
# Eqconvert.downloadseedbycsv('input/dataset_EQ/merge_clear.csv', url, user, passwd)
# step 4 split labels by magnitude
# Datas.split_labels_mags('input/dataset_EQ/merge_stream.csv')
# step 5 transform waveform to spectogram for trainning
# Transform.make_spectogram_mags('input/dataset_EQ/merge_stream.csv')
# step 6 move spectogra into labels folder
# Transform.cp_spectogram_mags('input/dataset_EQ/merge_stream.csv')
# step 7 train your data with model CNN
# Modelsmag.trainmodels(directory+'/input/dataset_EQ/datasEQ_spectogram/', n_epoch, summary_models='summary')
# step 8 please cek your predicted magnitude
# Modelsmag.predictedmag('/Users/litbanggeo/Documents/KuliahS2FisikaInstrumentasi/ProjectAkhir/NoiseAnalysis/application_seismic_noise/input/dataset_EQ/datasEQ/2.5/CISI.IA_20090616044802.645_EV', 'input/models_mags')
