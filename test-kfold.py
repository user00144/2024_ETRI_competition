from models.models import Q_Model, S_Model
from models.S_models_ import S_1_Model, S_2_Model, S_3_Model, S_4_Model
from datas.commondataset import ETRI_Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
import os
import pandas as pd
from tqdm import tqdm
import numpy as np
import random

seed = 21
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  
os.environ["CUDA_VISIBLE_DEVICES"]="0"

DATA_PATH = '/workspace/jupyter_workspace/SHARED_FILES/docker_shared/'
test_path = os.path.join(DATA_PATH, 'test_dataset')

test_df_mLight = pd.read_parquet(os.path.join(test_path, 'ch2024_test_m_light.parquet.gzip'))
test_df_wHr = pd.read_parquet(os.path.join(test_path, 'ch2024_test_w_heart_rate.parquet.gzip'))
test_df_wLight = pd.read_parquet(os.path.join(test_path, 'ch2024_test_w_light.parquet.gzip'))

answer_path = os.path.join(DATA_PATH, 'answer_sample.csv')

answer_df = pd.read_csv(answer_path)


Q_Test_Dataset = ETRI_Dataset(answer_path, test_df_mLight, test_df_wHr, test_df_wLight, 'q', 'zero')
Q_Test_DataLoader = DataLoader(Q_Test_Dataset, batch_size = 1, shuffle = False, num_workers = 4)
S_Test_Dataset = ETRI_Dataset(answer_path, test_df_mLight, test_df_wHr, test_df_wLight, 's', 'zero')
S_Test_DataLoader = DataLoader(S_Test_Dataset, batch_size = 1, shuffle = False, num_workers = 4)


FOLD = 10

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def Q_test(models) :
    for Q_num in range(3) :
        model = models[Q_num]
        all_predictions = []
        answer_np = np.zeros(115)
        for f in range(FOLD) :
            model.load_state_dict(torch.load(f'./archive/Q_{Q_num+1}_{f}_best.pt'))
            model.to(device)
            model.eval()
            predictions = []
            with torch.no_grad() :
                for i, batch in enumerate(tqdm(Q_Test_DataLoader)) :
                    mLight, wHr, wLight, labels = batch
                    mLight, wHr, wLight = mLight.to(device), wHr.to(device), wLight.to(device)
                    prediction = model(mLight, wHr, wLight)
                    pred_lbl = prediction.cpu().item()
                    predictions.append(pred_lbl)
            all_predictions.append(predictions)
        for pred in all_predictions :
            answer_np += np.array(pred)
        answer_np /= FOLD
        print(answer_np)
            
        answer_df[f"Q{Q_num+1}"] = np.round(answer_np).astype(np.int64)

def S_test(models) :
    for Q_num in range(4) :
        model = models[Q_num]
        all_predictions = []
        answer_np = np.zeros(115)
        for f in range(FOLD) :
            model.load_state_dict(torch.load(f'./archive/S_{Q_num+1}_{f}_best.pt'))
            model.to(device)
            model.eval()
            predictions = []
            with torch.no_grad() :
                for i, batch in enumerate(tqdm(S_Test_DataLoader)) :
                    mLight, wHr, wLight, labels = batch
                    mLight, wHr, wLight = mLight.to(device), wHr.to(device), wLight.to(device)
                    prediction = model(mLight, wHr, wLight)
                    pred_lbl = prediction.cpu().item()
                    predictions.append(pred_lbl)
            all_predictions.append(predictions)
        for pred in all_predictions :
            answer_np += np.array(pred)
        answer_np /= FOLD
        print(answer_np)
            
        answer_df[f"S{Q_num+1}"] = np.round(answer_np).astype(np.int64)


Q_models = [Q_Model(), Q_Model(), Q_Model()]
S_models = [S_Model(), S_Model(), S_Model(), S_Model()]

Q_test(Q_models)
S_test(S_models)

print(answer_df)

answer_df.to_csv('./pred.csv',index=False)
