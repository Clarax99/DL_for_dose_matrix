from dataset import niiDataset, NiiFeatureDataset
from model import FirstNet, CNN, Loop, CNN_tho, NewNet
from loss import LDAMHingeLoss
from os.path import join
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms as tf
import os
import sys
os.environ['KMP_DUPLICATE_LIB_OK']='True'


if __name__ == "__main__":

    ### WHO IS RUNNING THE CODE
    on_ruche = False
    user = "cousteixc"

    ### CONFIG
    if on_ruche:
        path_to_dir = f"/gpfs/workdir/{user}/dose_matrices"
        csv_name = "labels_ruche.csv"
    else : 
        csv_name = "labels.csv"
        if user=="cousteixc":
            path_to_dir = "D:\data\dose_matrices_updated_25_01_2023"
        elif user=="menardth":
            path_to_dir = "G:\data\dose_matrices_updated_25_01_2023"

    min_card_age = 50  #int(sys.argv[1])
    weights = torch.tensor([1,(1378-282)/282]) if min_card_age==40 else torch.tensor([1, (580-282)/282])
    batch_size = 4
    epochs = 50
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    ### DATALOADER
    data_transform = tf.Compose([tf.Normalize(mean=1.3, std=4.9)])

    training_data = NiiFeatureDataset(csv_name, "clinical_features.csv", path_to_dir, min_card_age, training = True)
    val_data = NiiFeatureDataset(csv_name, "clinical_features.csv", path_to_dir, min_card_age, validation = True)

    #training_data = niiDataset(csv_name, path_to_dir, min_card_age, training = True)
    #val_data = niiDataset(csv_name, path_to_dir, min_card_age, validation = True)

    print(f"Train set : {len(training_data)}, Validation set : {len(val_data)}")

    train_dataloader = DataLoader(training_data, batch_size, shuffle=True, drop_last=False)
    val_dataloader = DataLoader(val_data, batch_size, shuffle=True, drop_last=False)

    ### NET AND LOSS
    name_exp = f'newnet_noreg_{min_card_age}_allfeat_weighted_loss_3'
    net = NewNet(70, 15, 10).to(DEVICE)
    loss = nn.CrossEntropyLoss(weights)
    optimizer = torch.optim.Adam(params=net.parameters(), lr=0.001, weight_decay=0) #int(sys.argv[2])
    model = Loop(train_dataloader, val_dataloader, net, loss, optimizer, DEVICE, path_to_dir, f"{name_exp}.pth")
    print(f"Training {net.__class__.__name__} for {epochs} epochs on {DEVICE}")

    ### SAVING PARAMETERS
    dic = {"name":name_exp, "min_card_age":min_card_age, "net": net.__class__.__name__, "loss": loss.__class__.__name__, "weighted": True if loss.weight!=None else False,
           "optimizer":optimizer.__class__.__name__, "lr":optimizer.param_groups[0]['lr'], "weight_decay":optimizer.param_groups[0]['weight_decay']}
    
    with open(join(path_to_dir, "saved_models", f'{name_exp}.txt'), 'w') as f:
        for key in dic:
            f.write(f'{key} : {dic[key]}\n')

    ### TRAINING
    for epoch in range(2):
        print(f"-------------------------\nEpoch {epoch+1}")
        model.train_loop(epoch)
        model.validation_loop(epoch)
    model.plot_loss()
    model.plot_acc()
    print("Done!")    

