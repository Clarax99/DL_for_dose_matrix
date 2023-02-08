from dataset import niiDataset
from model import FirstNet, Loop
from os.path import join
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


if __name__ == "__main__":

    ### CONFIG

    path_to_dir = "D:\data\dose_matrices_updated_25_01_2023"
    min_card_age = 50
    batch_size = 8
    epochs = 10


    ### DATALOADER
    training_data = niiDataset("labels.csv", path_to_dir, min_card_age, training = True)
    val_data = niiDataset("labels.csv", path_to_dir, min_card_age, validation = True)

    print(len(training_data), len(val_data))

    train_dataloader = DataLoader(training_data, batch_size, shuffle=True, drop_last=True)
    val_dataloader = DataLoader(val_data, batch_size, shuffle=True, drop_last=True)

    ### NET AND LOSS
    net = FirstNet()
    loss = nn.CrossEntropyLoss()
    optimizer_adam = torch.optim.Adam(params=net.parameters(), lr=0.001)
    model = Loop(train_dataloader, val_dataloader, net, loss, optimizer_adam)

    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        model.train_loop()
        model.validation_loop()
    print("Done!")    

