from dataset import niiDataset, NiiFeatureDataset
from model import FirstNet, CNN, Loop, CNN_tho, NewNet
from os.path import join
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms as tf
import os
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

    min_card_age = 40
    batch_size = 4
    epochs = 50
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    feature_list = ["do_ANTHRA", "do_ALKYL", "do_VINCA"]
    
    ### DATALOADER
    data_transform = tf.Compose([
                    tf.Normalize(mean=1.3, std=4.9)
                    ])


    training_data = NiiFeatureDataset(csv_name, "clinical_features.csv", path_to_dir, min_card_age, training = True)
    val_data = NiiFeatureDataset(csv_name, "clinical_features.csv", path_to_dir, min_card_age, validation = True)

    #training_data = niiDataset(csv_name, path_to_dir, min_card_age, training = True)
    #val_data = niiDataset(csv_name, path_to_dir, min_card_age, validation = True)

    print(f"Train set : {len(training_data)}, Validation set : {len(val_data)}")

    train_dataloader = DataLoader(training_data, batch_size, shuffle=True, drop_last=False)
    val_dataloader = DataLoader(val_data, batch_size, shuffle=True, drop_last=False)

    def mean_std(loader):
        mean, std = [], []
        for images, labels in loader:
            # shape of images = [b,c,w,h]
            mean.append(images.mean([0, 2,3, 4]))
            std.append(images.std([0, 2,3, 4]))
        return sum(mean)/len(loader), sum(std)/len(loader)
    

    ### NET AND LOSS
    net = NewNet(70, 15, 10).to(DEVICE)
    #net =  CNN(16).to(DEVICE)
    loss = nn.CrossEntropyLoss()
    optimizer_adam = torch.optim.Adam(params=net.parameters(), lr=0.001)
    model = Loop(train_dataloader, val_dataloader, net, loss, optimizer_adam, DEVICE, join(path_to_dir, "saved_models", "newnet_withTF_40.pth"))
    print(f"Training {net.__class__.__name__} for {epochs} epochs on {DEVICE}")

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}\n-------------------------------")
        model.train_loop(epoch)
        model.validation_loop(epoch)
    print("Done!")    

