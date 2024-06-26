from models.models import Q_Model, S_Model
from models.S_models_ import S_1_Model, S_2_Model, S_3_Model, S_4_Model
from datas.commondataset import ETRI_Dataset
from torch.utils.data import DataLoader, Subset
from transformers.optimization import get_cosine_schedule_with_warmup
from sklearn.model_selection import StratifiedKFold, KFold
from eval_func import eval_func
from makeplot import make_plot_and_save
import torch.nn as nn
import torch
from tqdm import tqdm
from torchinfo import summary
import os
import pandas as pd
import numpy as np
import random
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  
os.environ["CUDA_VISIBLE_DEVICES"]="0"

DATA_PATH = '/workspace/jupyter_workspace/SHARED_FILES/docker_shared/'

val_path = os.path.join(DATA_PATH, 'val_dataset')
df_mLight = pd.read_parquet(os.path.join(val_path, 'ch2024_val__m_light.parquet.gzip'))
df_wHr = pd.read_parquet(os.path.join(val_path, 'ch2024_val__w_heart_rate.parquet.gzip'))
df_wLight = pd.read_parquet(os.path.join(val_path, 'ch2024_val__w_light.parquet.gzip'))

label_path = os.path.join(DATA_PATH, 'val_label.csv')


Q_dataset = ETRI_Dataset(label_path, df_mLight, df_wHr, df_wLight, 'q', 'zero')
#Q_dataloader = DataLoader(Q_dataset, batch_size = 1, shuffle = True, num_workers = 4)

S_dataset = ETRI_Dataset(label_path, df_mLight, df_wHr, df_wLight, 's', 'zero')
#S_dataloader = DataLoader(S_dataset, batch_size = 1, shuffle = True, num_workers = 4)


# Fold로 학습

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPOCHS = 50
FOLD = 10

seed = 21
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


fold = KFold(n_splits = FOLD, shuffle = True, random_state = seed)

def Q_train (models) :
    for Q_num in range(3) :
        print(f"==== Training Question Q{Q_num+1} Model ====")
        model = models[Q_num]
        model = model.to(device)
        loss_fn = nn.BCELoss()
        optimizer = torch.optim.AdamW(model.parameters(),lr= 0.00001,weight_decay = 0.0005)
        warmup_ratio = 0.1
        t_total = len(Q_dataset) * EPOCHS * FOLD
        warmup_step = int(t_total * warmup_ratio)
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps = warmup_step, num_training_steps = t_total)
        
        for f, (train_idx, val_idx) in enumerate(fold.split(Q_dataset)) :
            best_loss = 10
            train_ds = Subset(Q_dataset, train_idx)
            val_ds = Subset(Q_dataset, val_idx)

            train_dl = DataLoader(train_ds, batch_size = 1 , shuffle = True, num_workers = 4)
            val_dl = DataLoader(val_ds, batch_size = 1, shuffle = False, num_workers = 4)

            FOLD_LOSSES_T = []
            FOLD_LOSSES_V = []

            prev_val_loss = 0
            step = 1

            min_val_loss = 0.2
            
            for epoch in range(EPOCHS) :
                epoch_loss = 0
    
                model = model.train()
                for i, batch in enumerate(tqdm(train_dl)) :
                    mLight, wHr, wLight, labels = batch
                    mLight, wHr, wLight = mLight.to(device), wHr.to(device), wLight.to(device)
                    prediction = model(mLight, wHr, wLight)
                    loss = loss_fn(prediction[0], labels[0][Q_num].to(device)) + prev_val_loss * 0.5
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    scheduler.step()
                
                    epoch_loss += loss.item()

                val_loss = eval_func(model, val_dl, loss_fn, Q_num)
                epoch_loss = epoch_loss / len(train_dl)

                if epoch % step == 0 :
                    prev_val_loss = val_loss
                

                FOLD_LOSSES_T.append(epoch_loss)
                FOLD_LOSSES_V.append(val_loss)
                
                if (best_loss > val_loss) & (val_loss > min_val_loss) :
                    torch.save(model.state_dict(), f'./archive/Q_{Q_num+1}_{f}_best.pt') 
                    best_loss = val_loss
                print(f"FOLD {f} : EPOCH {epoch} - Loss : {epoch_loss}  \n val_Loss : {val_loss}\n")
            make_plot_and_save(FOLD_LOSSES_T, FOLD_LOSSES_V, EPOCHS, f, Q_num, 'q')


def S_train (models) :
    for Q_num in range(4) :
        print(f"==== Training Question S{Q_num+1} Model ====")
        model = models[Q_num]
        model = model.to(device)
        loss_fn = nn.BCELoss()
        optimizer = torch.optim.AdamW(model.parameters(),lr= 0.00001,weight_decay = 0.0005)
        warmup_ratio = 0.1
        t_total = len(S_dataset) * EPOCHS * FOLD
        warmup_step = int(t_total * warmup_ratio)
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps = warmup_step, num_training_steps = t_total)


        for f, (train_idx, val_idx) in enumerate(fold.split(S_dataset)) :
        
            best_loss = 10
            train_ds = Subset(S_dataset, train_idx)
            val_ds = Subset(S_dataset, val_idx)

            train_dl = DataLoader(train_ds, batch_size = 1 , shuffle = True, num_workers = 4)
            val_dl = DataLoader(val_ds, batch_size = 1, shuffle = False, num_workers = 4)

            FOLD_LOSSES_T = []
            FOLD_LOSSES_V = []

            min_val_loss = 0.2
            
            for epoch in range(EPOCHS) :
                epoch_loss = 0
    
                model = model.train()
                for i, batch in enumerate(tqdm(train_dl)) :
                    mLight, wHr, wLight, labels = batch
                    mLight, wHr, wLight = mLight.to(device), wHr.to(device), wLight.to(device)
                    prediction = model(mLight, wHr, wLight)
                    loss = loss_fn(prediction[0], labels[0][Q_num].to(device))
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    scheduler.step()
                
                    epoch_loss += loss.item()
                
                epoch_loss = epoch_loss / len(train_dl)
                val_loss = eval_func(model, val_dl, loss_fn, Q_num)

                
                FOLD_LOSSES_T.append(epoch_loss)
                FOLD_LOSSES_V.append(val_loss)

                if (best_loss > val_loss) & (val_loss > min_val_loss) :
                    torch.save(model.state_dict(), f'./archive/S_{Q_num+1}_{f}_best.pt') 
                    best_loss = val_loss
                print(f"FOLD {f} : EPOCH {epoch} - Loss : {epoch_loss} \n val_Loss : {val_loss}\n")
            make_plot_and_save(FOLD_LOSSES_T, FOLD_LOSSES_V, EPOCHS, f, Q_num, 's')

Q_models = [Q_Model(), Q_Model(), Q_Model()]
S_models = [S_Model(), S_Model(), S_Model(), S_Model()]
summary(Q_models[0])
summary(S_models[0])

Q_train(Q_models)
S_train(S_models)


