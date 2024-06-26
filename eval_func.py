import torch
import torch.nn
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



def eval_func(model, dataloader, loss_fn, Q_num) :
    e_loss = 0
    model.eval()
    with torch.no_grad() :
         for i, batch in enumerate(tqdm(dataloader)) :
                mLight, wHr, wLight, labels = batch
                mLight, wHr, wLight = mLight.to(device), wHr.to(device), wLight.to(device)
                prediction = model(mLight, wHr, wLight)
                loss = loss_fn(prediction[0], labels[0][Q_num].to(device))
                e_loss += loss.item()
    e_loss /= len(dataloader)

    return e_loss

def eval_mAcc_func(model, dataloader, loss_fn, Q_num) :
    e_loss = 0
    model.eval()
    with torch.no_grad() :
         for i, batch in enumerate(tqdm(dataloader)) :
                mAcc ,mLight, wHr, wLight, labels = batch
                mAcc, mLight, wHr, wLight = mAcc.to(device), mLight.to(device), wHr.to(device), wLight.to(device)
                prediction = model(mAcc, mLight, wHr, wLight)
                loss = loss_fn(prediction[0], labels[0][Q_num].to(device))
                e_loss += loss.item()
    e_loss /= len(dataloader)

    return e_loss