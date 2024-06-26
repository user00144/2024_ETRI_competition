import pandas as pd
import numpy as np




# mAmb? , mActivity?, AppUsage?
# GPS데이터는 너무 personal함 ..

# data / label 개수 : 105개

# mAcc - 24시간 평균 4,244,143 개의 데이터(x, y, z) 각각 존재
# mlight - 24시간 평균 141 개의 데이터
# wHr - 24시간 평균 1246 개의 데이터
# wlight - 24시간 평균 128개의 데이터

# 별도의 data fit 한 전처리 코드 필요
# mUsage - 24시간 평균 142개의 데이터 (전처리 코드 필요)
# mAmb? - 24시간 평균 710개의 데이터 (전처리 코드 필요)
# mgps? wPedo?



#datetime으로 변환하여 비교
#label을 먼저 가져와서, label을 iteration하면서 subject id, datetime 값 획득
# question type q 전날 09:00 ~ 당일 09:00 (24 시간)
# question type s 전날 21:00 ~ 당일 09:00 (12 시간)


# data -> [numpy array]
def random_select_and_drop(data, num_to_drop) :
    drop_idx = np.random.choice(range(0,len(data)),num_to_drop,replace=False)
    data = np.delete(data, drop_idx)
    return data

def fill_na_random(data, num_to_fill, fill_type) :
    #fill_idx = np.random.choice(range(0,len(data)), num_to_fill, replace=False)
    beta = 0.001
    if data.size != 0 :
        data_range = [np.min(data), np.max(data)]
    else :
        data_range = [0,1]
    if fill_type == 'noise' :
        m = np.concatenate((np.ones(len(data),dtype=bool), np.zeros(num_to_fill, dtype=bool)))
        np.random.shuffle(m)
        out = np.empty(len(data)+num_to_fill, dtype = data.dtype)
        out[m] = data
        out[~m] = np.random.rand(num_to_fill) * data_range[1] + beta
        return out
    elif fill_type == 'zero' :
        m = np.concatenate((np.ones(len(data),dtype=bool), np.zeros(num_to_fill, dtype=bool)))
        np.random.shuffle(m)
        out = np.empty(len(data)+num_to_fill, dtype = np.float32)
        out[m] = data
        out[~m] = 0 + beta
        return out
    else :
        return

def syn_num_data(data, data_type, question_type, fill_type) :
    if data_type == 'mAcc' :
        #syn_num = 4244140
        syn_num = 4244000
    elif data_type == 'mLight' :
        syn_num = 140
    elif data_type == 'wHr' :
        syn_num = 1240
    elif data_type == 'wLight' :
        syn_num = 128
    else :
        syn_num = 0
        
    if question_type == 's' :
        syn_num = syn_num // 2

    fill_drop_range = syn_num - len(data)
#    print(f"fill_drop_range {data_type} : {fill_drop_range}\n")

    if fill_drop_range > 0 :
        if fill_type  : 
            syn_data = fill_na_random(data, abs(fill_drop_range), fill_type)
        else :
            syn_data = fill_na_random(data, abs(fill_drop_range), 'zero')
        return syn_data
    elif  fill_drop_range < 0 :
        syn_data = random_select_and_drop(data, abs(fill_drop_range))
        return syn_data
    else :
        return data

def clamp_data(data, data_type) :
    if data_type == 'mLight' :
        return np.clip(np.log1p(data), 0.0, 10.0)
    elif data_type == 'wHr' :
        return np.clip(data, 0.0, 200.0)
    elif data_type == 'wLight' :
        return np.clip(np.log1p(data), 0.0 , 11.0)
    elif data_type == 'mAcc' :
        return np.clip(data,-180.0,180.0)
    else : return

def Minmax(s):
    if (np.min(s) == 0) & (np.max(s) == 0) :
        return s
    else :
        return (s-np.min(s))/(np.max(s)-np.min(s))

def AbsoluteMinmax(s, data_type) :
    n_min, n_max = 0, 0
    if data_type == 'mLight' :
        n_min = 0
        n_max = 10.0
        return (s-n_min)/(n_max - n_min)
    elif data_type == 'wHr' :
        n_min = 0
        n_max = 200
        return (s-n_min)/(n_max - n_min)
    elif data_type == 'wLight' :
        n_min = 0
        n_max = 11.0
        return (s-n_min)/(n_max - n_min)
    
    elif data_type == 'mAcc' :
        n_min = -180
        n_max = 180
        return (s-n_min)/(n_max - n_min)
    else : return
#df __ mAcc, mLight, wHr, wLight
