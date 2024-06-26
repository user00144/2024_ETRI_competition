from torch.utils.data import Dataset
import torch
import pandas as pd
import os
import numpy as np
from datetime import datetime, timedelta
from datas.utils.dataset_utils import syn_num_data, AbsoluteMinmax, clamp_data


#DATA_PATH = '/workspace/jupyter_workspace/SHARED_FILES/docker_shared/'

#val_path = os.path.join(DATA_PATH, 'val_dataset')
#df_mLight = pd.read_parquet(os.path.join(val_path, 'ch2024_val__m_light.parquet.gzip'))
#df_wHr = pd.read_parquet(os.path.join(val_path, 'ch2024_val__w_heart_rate.parquet.gzip'))
#df_wLight = pd.read_parquet(os.path.join(val_path, 'ch2024_val__w_light.parquet.gzip'))

#각 유저의 첫째날 데이터에 대해서는 따로 추론?
#첫째 날 데이터는 스킵


class ETRI_Dataset(Dataset) :
    def __init__(self, label_path, df_mLight, df_wHr, df_wLight, question_type, fill_type) :
        self.df_label = pd.read_csv(label_path)
        #self.df_mAcc = None
        self.df_mLight = df_mLight
        self.df_wHr = df_wHr
        self.df_wLight = df_wLight
        
        self.question_type = question_type
        self.fill_type = fill_type
        self.flg_mAcc = 0
        self.df_label['date'] = pd.to_datetime(self.df_label['date'])
        self.df_mLight['timestamp'] = pd.to_datetime(self.df_mLight['timestamp'])
        self.df_wHr['timestamp'] = pd.to_datetime(self.df_wHr['timestamp'])
        self.df_wLight['timestamp'] = pd.to_datetime(self.df_wLight['timestamp'])

    def __getitem__(self, idx) :
        if self.question_type == 'q' :
            labels = [[int(self.df_label.iloc[idx].Q1)], [int(self.df_label.iloc[idx].Q2)], [int(self.df_label.iloc[idx].Q3)]]
        elif self.question_type == 's' :
            labels = [[int(self.df_label.iloc[idx].S1)], [int(self.df_label.iloc[idx].S2)], [int(self.df_label.iloc[idx].S3)],[ int(self.df_label.iloc[idx].S4)]]
        else :
            labels = []
        #mAcc_x_values, mAcc_y_values, mAcc_z_values, 
        mLight_values, wHr_values, wLight_values = self.get_user_data_from_label(idx)

        #mAcc_x_values = syn_num_data(mAcc_x_values, 'mAcc', self.question_type, self.fill_type)
        #mAcc_y_values = syn_num_data(mAcc_y_values, 'mAcc', self.question_type, self.fill_type)
        #mAcc_z_values = syn_num_data(mAcc_z_values, 'mAcc', self.question_type, self.fill_type)
        #mAcc_values = torch.stack([torch.from_numpy(mAcc_x_values), torch.from_numpy(mAcc_y_values), torch.from_numpy(mAcc_z_values)],0)


        mLight_values = syn_num_data(mLight_values, 'mLight', self.question_type, self.fill_type)
        wHr_values = syn_num_data(wHr_values, 'wHr', self.question_type, self.fill_type)
        wLight_values = syn_num_data(wLight_values, 'wLight', self.question_type, self.fill_type)


        mLight_values = clamp_data(mLight_values, 'mLight')
        wHr_values = clamp_data(wHr_values, 'wHr')
        wLight_values = clamp_data(wLight_values, 'wLight')


        mLight_values = AbsoluteMinmax(mLight_values, 'mLight')
        wHr_values = AbsoluteMinmax(wHr_values, 'wHr')
        wLight_values = AbsoluteMinmax(wLight_values, 'wLight')
        
        #torch.from_numpy(mAcc_x_values), torch.from_numpy(mAcc_y_values), torch.from_numpy(mAcc_z_values),
        return  torch.from_numpy(mLight_values).float().squeeze(), torch.from_numpy(wHr_values).float().squeeze(), torch.from_numpy(wLight_values).float().squeeze(), torch.Tensor(labels).float()

    def get_user_data_from_label(self, lbl_idx) :
        sub_id = self.df_label.iloc[lbl_idx].subject_id
        
        #if self.flg_mAcc != int(sub_id) :
            #self.df_mAcc = pd.read_parquet(os.path.join(val_path), f'ch2024_val_m_acc_part_{sub_id}.parquet.gzip')
            #self.flg_mAcc = sub_id
            
        lbl_date = self.df_label.iloc[lbl_idx].date
        if self.question_type == 'q' :
            date_start = lbl_date + timedelta(days = -1, hours = 9)
            date_end = lbl_date + timedelta(hours = 9)
            #date_start = lbl_date
            #date_end = lbl_date + timedelta(days=1)            
        elif self.question_type == 's' :
            date_start = lbl_date + timedelta(hours = -3)
            date_end = lbl_date + timedelta(hours=9)
            #date_start = lbl_date
            #date_end = lbl_date + timedelta(hours=12)
        else :
            return
        date_start = date_start.to_pydatetime()
        date_end = date_end.to_pydatetime()
        #        print(f"get data from sub {sub_id} in date {date_start} ~ {date_end}\n")
        df_user_mLight, df_user_wHr, df_user_wLight = self.get_user_dataframe(sub_id, date_start, date_end)

        #df_user_mAcc.x.values, df_user_mAcc.y.values, df_user_mAcc.z.values,
        return df_user_mLight.m_light.values, df_user_wHr.heart_rate.values, df_user_wLight.w_light.values

            
    def get_user_dataframe(self, subject_id, date_start, date_end) :
        #df_user_mAcc = self.df_mAcc[self.df_mAcc['subject_id'] == subject_id & self.df_mAcc['timestamp'].between(date_start, date_end)]
        df_user_mLight = self.df_mLight[self.df_mLight['subject_id'] == subject_id]
        df_user_wHr = self.df_wHr[self.df_wHr['subject_id'] == subject_id]
        df_user_wLight = self.df_wLight[self.df_wLight['subject_id'] == subject_id]

        df_user_mLight = df_user_mLight[df_user_mLight["timestamp"].between(date_start,date_end)]
        df_user_wHr = df_user_wHr[df_user_wHr["timestamp"].between(date_start,date_end)]
        df_user_wLight = df_user_wLight[df_user_wLight["timestamp"].between(date_start,date_end)]

        return df_user_mLight, df_user_wHr, df_user_wLight
    
    def __len__(self) :
        return len(self.df_label)


#=====================================================================================
class ETRI_mAcc_Dataset(Dataset) :
    def __init__(self, label_path, df_mLight, df_wHr, df_wLight, question_type, fill_type, is_test, val_path, test_path) :
        self.df_label = pd.read_csv(label_path)
        self.val_path = val_path
        self.test_path = test_path
        self.is_test = is_test
        self.df_mAcces = []
        if self.is_test :
            self.df_mAcces = [pd.read_parquet(os.path.join(self.test_path, 'ch2024_test__m_acc_part_5.parquet.gzip')),pd.read_parquet(os.path.join(self.test_path, 'ch2024_test__m_acc_part_6.parquet.gzip')),pd.read_parquet(os.path.join(self.test_path, 'ch2024_test__m_acc_part_7.parquet.gzip')),pd.read_parquet(os.path.join(self.test_path, 'ch2024_test__m_acc_part_8.parquet.gzip'))]
        else :
            self.df_mAcces = [pd.read_parquet(os.path.join(self.val_path, 'ch2024_val__m_acc_part_1.parquet.gzip')),pd.read_parquet(os.path.join(self.val_path, 'ch2024_val__m_acc_part_2.parquet.gzip')),pd.read_parquet(os.path.join(self.val_path, 'ch2024_val__m_acc_part_3.parquet.gzip')),pd.read_parquet(os.path.join(self.val_path, 'ch2024_val__m_acc_part_4.parquet.gzip'))]
        self.df_mAcc = None
        self.df_mLight = df_mLight
        self.df_wHr = df_wHr
        self.df_wLight = df_wLight
        
        self.question_type = question_type
        self.fill_type = fill_type
        self.flg_mAcc = 0
        self.df_label['date'] = pd.to_datetime(self.df_label['date'])
        self.df_mLight['timestamp'] = pd.to_datetime(self.df_mLight['timestamp'])
        self.df_wHr['timestamp'] = pd.to_datetime(self.df_wHr['timestamp'])
        self.df_wLight['timestamp'] = pd.to_datetime(self.df_wLight['timestamp'])

    def __getitem__(self, idx) :
        if self.question_type == 'q' :
            labels = [[int(self.df_label.iloc[idx].Q1)], [int(self.df_label.iloc[idx].Q2)], [int(self.df_label.iloc[idx].Q3)]]
        elif self.question_type == 's' :
            labels = [[int(self.df_label.iloc[idx].S1)], [int(self.df_label.iloc[idx].S2)], [int(self.df_label.iloc[idx].S3)],[ int(self.df_label.iloc[idx].S4)]]
        else :
            labels = []
        #
        mAcc_x_values, mAcc_y_values, mAcc_z_values, mLight_values, wHr_values, wLight_values = self.get_user_data_from_label(idx)

        mAcc_x_values = syn_num_data(mAcc_x_values, 'mAcc', self.question_type, self.fill_type)
        mAcc_y_values = syn_num_data(mAcc_y_values, 'mAcc', self.question_type, self.fill_type)
        mAcc_z_values = syn_num_data(mAcc_z_values, 'mAcc', self.question_type, self.fill_type)
        mLight_values = syn_num_data(mLight_values, 'mLight', self.question_type, self.fill_type)
        wHr_values = syn_num_data(wHr_values, 'wHr', self.question_type, self.fill_type)
        wLight_values = syn_num_data(wLight_values, 'wLight', self.question_type, self.fill_type)


        mLight_values = clamp_data(mLight_values, 'mLight')
        wHr_values = clamp_data(wHr_values, 'wHr')
        wLight_values = clamp_data(wLight_values, 'wLight')
        mAcc_x_values = clamp_data(mAcc_x_values, 'mAcc')
        mAcc_y_values = clamp_data(mAcc_y_values, 'mAcc')
        mAcc_z_values = clamp_data(mAcc_z_values, 'mAcc')

        mLight_values = AbsoluteMinmax(mLight_values, 'mLight')
        wHr_values = AbsoluteMinmax(wHr_values, 'wHr')
        wLight_values = AbsoluteMinmax(wLight_values, 'wLight')
        mAcc_x_values = AbsoluteMinmax(mAcc_x_values, 'mAcc')
        mAcc_y_values = AbsoluteMinmax(mAcc_y_values, 'mAcc')
        mAcc_z_values = AbsoluteMinmax(mAcc_z_values, 'mAcc')

        mAcc_values = torch.stack([torch.from_numpy(mAcc_x_values).float(), torch.from_numpy(mAcc_y_values).float(), torch.from_numpy(mAcc_z_values).float()],0)
        
        return  mAcc_values, torch.from_numpy(mLight_values).float().squeeze(), torch.from_numpy(wHr_values).float().squeeze(), torch.from_numpy(wLight_values).float().squeeze(), torch.Tensor(labels).float()

    def get_user_data_from_label(self, lbl_idx) :
        sub_id = self.df_label.iloc[lbl_idx].subject_id
        
        if self.flg_mAcc != int(sub_id) :
            if self.is_test :   
                self.df_mAcc = self.df_mAcces[int(sub_id - 5)]
                self.flg_mAcc = sub_id
            else :
                self.df_mAcc = self.df_mAcces[int(sub_id - 1)]
                self.flg_mAcc = sub_id
            
        lbl_date = self.df_label.iloc[lbl_idx].date
        if self.question_type == 'q' :
            date_start = lbl_date + timedelta(days = -1, hours = 9)
            date_end = lbl_date + timedelta(hours = 9)
            #date_start = lbl_date
            #date_end = lbl_date + timedelta(days=1)            
        elif self.question_type == 's' :
            date_start = lbl_date + timedelta(hours = -3)
            date_end = lbl_date + timedelta(hours=9)
            #date_start = lbl_date
            #date_end = lbl_date + timedelta(hours=12)
        else :
            return
        date_start = date_start.to_pydatetime()
        date_end = date_end.to_pydatetime()
        df_user_mAcc, df_user_mLight, df_user_wHr, df_user_wLight = self.get_user_dataframe(sub_id, date_start, date_end)

        #
        return df_user_mAcc.x.values, df_user_mAcc.y.values, df_user_mAcc.z.values, df_user_mLight.m_light.values, df_user_wHr.heart_rate.values, df_user_wLight.w_light.values

            
    def get_user_dataframe(self, subject_id, date_start, date_end) :
        df_user_mAcc = self.df_mAcc[self.df_mAcc['subject_id'] == subject_id]
        df_user_mLight = self.df_mLight[self.df_mLight['subject_id'] == subject_id]
        df_user_wHr = self.df_wHr[self.df_wHr['subject_id'] == subject_id]
        df_user_wLight = self.df_wLight[self.df_wLight['subject_id'] == subject_id]

        df_user_mAcc = df_user_mAcc[df_user_mAcc["timestamp"].between(date_start,date_end)]
        df_user_mLight = df_user_mLight[df_user_mLight["timestamp"].between(date_start,date_end)]
        df_user_wHr = df_user_wHr[df_user_wHr["timestamp"].between(date_start,date_end)]
        df_user_wLight = df_user_wLight[df_user_wLight["timestamp"].between(date_start,date_end)]

        return df_user_mAcc, df_user_mLight, df_user_wHr, df_user_wLight
    
    def __len__(self) :
        return len(self.df_label)
