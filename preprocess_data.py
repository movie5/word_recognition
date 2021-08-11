#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 24 20:46:06 2021

@author: pjh
"""

from glob import glob
import numpy as np
import librosa
import librosa.display
import soundfile as sf
import matplotlib.pyplot as plt
import pandas as pd
import os

def csv_to_wav(data, save_folder, human_index, index):

    for i, item in enumerate(data['Model']):
        if item == 'Sample Interval':
            base = i

    data_part = data.iloc[base+10:,:]['TBS1072C']
    np_data = data_part.to_numpy()
    
    sr = int(1/float(data['TBS1072C'][base]))
    
    np_data = np_data.astype(np.float32)
    np_data = np_data - np.mean(np_data)
    np_data = np_data / np.max(np.abs(np_data))
    
    write_path = save_folder + '/data_' + str(index) + '_'+ str(human_index)
    sf.write(write_path + '.wav', np_data, sr)
    

if __name__ == "__main__":
    abs_path = '/home/pjh/다운로드'
    folder_path = 'wed_six_tweleve'
    save_path = 'data_reco'
    
    folder_path  =os.path.join(abs_path, folder_path)
    save_path = os.path.join(abs_path, save_path)
    folder_list = glob(folder_path + '/*')
    label_list = []
    data_list = []
    
    for item in folder_list:
        
        if os.path.basename(item).startswith('person'):
            label_list.append(item) 
        else:
            data_list.append(item)
    
    sample_data = pd.read_csv(label_list[0])
    labels = sample_data.values[:,0].tolist()
    
    for label in labels:
        os.makedirs(os.path.join(save_path, label), exist_ok = True)


    for label in label_list:
        person_index = label.split('.')[0].split('/')[-1][6:]
        label_data = pd.read_csv(label).values
        for row in label_data:
            save_folder = os.path.join(save_path, row[0])
            start = row[1]
            end = row[2]
            item_row = [item for item in data_list \
                        if (int(item[-7:-4]) >= start) and (int(item[-7:-4]) <= end)]
            temp = [csv_to_wav(pd.read_csv(item), save_folder, person_index, i)\
                    for i, item in enumerate(item_row)]