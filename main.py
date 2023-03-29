from dataset import niiDataset
from model import FirstNet, CNN, Loop, CNN_tho, NewNet
from os.path import join
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
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

    min_card_age = 50
    batch_size = 4
    epochs = 50
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    ### DATALOADER
    training_data = niiDataset(csv_name, path_to_dir, min_card_age, training = True)
    val_data = niiDataset(csv_name, path_to_dir, min_card_age, validation = True)

    print(len(training_data), len(val_data))

    train_dataloader = DataLoader(training_data, batch_size, shuffle=True, drop_last=True) #drop last à changer !
    val_dataloader = DataLoader(val_data, batch_size, shuffle=True, drop_last=True)

    ### NET AND LOSS
    net = NewNet(15, 10).to(DEVICE)
    loss = nn.CrossEntropyLoss()
    optimizer_adam = torch.optim.Adam(params=net.parameters(), lr=0.001)
    model = Loop(train_dataloader, val_dataloader, net, loss, optimizer_adam, DEVICE, join(path_to_dir, "saved_models", "new_net_0.pth"))
    print(f"Training for {epochs} epochs on {DEVICE}")

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}\n-------------------------------")
        model.train_loop(epoch)
        model.validation_loop(epoch)
    print("Done!")    

