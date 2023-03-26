from dataset import niiDataset
from model import FirstNet, CNN, Loop, NewNet
from os.path import join
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


if __name__ == "__main__":

    ### CONFIG


    path_to_dir = "/gpfs/workdir/cousteixc/dose_matrices"
    min_card_age = 50
    batch_size = 32
    epochs = 50
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    ### DATALOADER
    training_data = niiDataset("labels_ruche.csv", path_to_dir, min_card_age, training = True)
    val_data = niiDataset("labels_ruche.csv", path_to_dir, min_card_age, validation = True)

    print(len(training_data), len(val_data))

    train_dataloader = DataLoader(training_data, batch_size, shuffle=True, drop_last=True) #drop last à changer !
    val_dataloader = DataLoader(val_data, batch_size, shuffle=True, drop_last=True)

    ### NET AND LOSS
    net = NewNet(15, 10)
    net = CNN(16).to(DEVICE)
    loss = nn.CrossEntropyLoss()
    optimizer_adam = torch.optim.Adam(params=net.parameters(), lr=0.001)
    model = Loop(train_dataloader, val_dataloader, net, loss, optimizer_adam, DEVICE)
    print(f"Training for {epochs} epochs on {DEVICE}")

    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        model.train_loop()
        model.validation_loop()
    print("Done!")    

