import torch.nn as nn
import torch
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score, confusion_matrix,f1_score, ConfusionMatrixDisplay
import os



class FirstNet(nn.Module):
    def __init__(self,  hidden_units=200):
        super(FirstNet, self).__init__()

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64*64*64, hidden_units)
        self.fc2 = nn.Linear(hidden_units, hidden_units)
        self.fc3 = nn.Linear(hidden_units, 2)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = nn.ReLU()(x)
        x = self.fc2(x)
        x = nn.ReLU()(x)
        x = self.fc3(x)
        return x.float()
    
class CNN(nn.Module):
    def __init__(self, out_channels: int, hidden_units : int =200):
        super(CNN, self).__init__()
        self.out_channels : int = out_channels
        self.conv1 = nn.Sequential(
            nn.Conv3d(1, out_channels, kernel_size=3, stride = 1, padding =1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(),
            )
        self.conv2 = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, kernel_size=3, stride =1, padding = 1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(),
        )

        self.fc1 = nn.Linear(out_channels*64*64*64, hidden_units)
        self.fc2 = nn.Linear(hidden_units, hidden_units)
        self.fc3 = nn.Linear(hidden_units, 2)


    def forward(self, x):
        x = x.float()
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(-1, self.out_channels*64*64*64)
        x = self.fc1(x)
        x = nn.ReLU()(x)
        x = self.fc2(x)
        x = nn.ReLU()(x)
        x = self.fc3(x)
        return x


class NewNet():
    super(NewNet, self).__init__()
        self.out_channels_1 : int = out_channels_1
        self.out_channels_2 : int = out_channels_2

        self.conv1 = nn.Sequential(
            nn.Conv3d(1, self.out_channels_1, kernel_size=3, stride = 1, padding = 'valid'),
            nn.ReLU()
            nn.Dropout(),
            nn.MaxPool3d(kernel_size=3)
            )
        self.conv2 = nn.Sequential(
            nn.Conv3d(self.out_channels_1, self.out_channels_2, kernel_size=2, stride = 1, padding = 'valid'),
            nn.ReLU()
            nn.Dropout(),
            nn.MaxPool3d(kernel_size=3)
            )
        self.conv3 = nn.Sequential(
            nn.Conv3d(self.out_channels_2, self.out_channels_1, kernel_size=2, stride = 1, padding = 'valid'),
            nn.ReLU()
            nn.Dropout(),
            nn.MaxPool3d(kernel_size=3)
            )

        self.fc1 = nn.Linear(self.out_channels_2*64*64*64, hidden_units)
        self.fc2 = nn.Linear(hidden_units, hidden_units)
        self.fc3 = nn.Linear(hidden_units, 2)


    def forward(self, x):
        x = x.float()
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        print(x.shape)
        x = x.view(-1, self.out_channels*64*64*64)
        x = self.fc1(x)
        x = nn.ReLU()(x)
        x = self.fc2(x)
        x = nn.ReLU()(x)
        x = self.fc3(x)
        return x

class CNN_tho(nn.Module):
    def __init__(self, out_channels: int, hidden_units : int =200):
        super(CNN, self).__init__()
        self.out_channels : int = out_channels
        self.conv1 = nn.Sequential(
            nn.Conv3d(1, out_channels, kernel_size=3, stride = 1, padding =1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(),
            )
        self.conv2 = nn.Sequential(
            nn.Conv3d(out_channels, out_channels*2, kernel_size=3, stride =1, padding = 1),
            nn.BatchNorm3d(out_channels*2),
            nn.ReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv3d(out_channels*2, out_channels*3, kernel_size=3, stride = 1, padding =1),
            nn.BatchNorm3d(out_channels*3),
            nn.ReLU(),
            torch.nn.MaxPool3d(kernel_size=2, stride=2),
            )
        self.conv4 = nn.Sequential(
            nn.Conv3d(out_channels*3, out_channels*4, kernel_size=3, stride =1, padding = 1),
            nn.BatchNorm3d(out_channels*4),
            nn.ReLU(),
        )
        self.conv5 = nn.Sequential(
            nn.Conv3d(out_channels*4, out_channels*5, kernel_size=3, stride = 1, padding =1),
            nn.BatchNorm3d(out_channels*5),
            nn.ReLU(),
            )
        self.conv6 = nn.Sequential(
            nn.Conv3d(out_channels*5, out_channels*6, kernel_size=3, stride =1, padding = 1),
            nn.BatchNorm3d(out_channels*6),
            nn.ReLU(),
        )

        self.fc1 = nn.Linear(out_channels*6*32*32*32, hidden_units)
        self.fc2 = nn.Linear(hidden_units, hidden_units)
        self.fc3 = nn.Linear(hidden_units, 2)


    def forward(self, x):
        x = x.float()
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        #print(x.shape)
        x = x.view(-1, self.out_channels*6*32*32*32)
        x = self.fc1(x)
        x = nn.ReLU()(x)
        x = self.fc2(x)
        x = nn.ReLU()(x)
        x = self.fc3(x)
        return x

class Loop():

    def __init__(self, train_dataloader, val_dataloader, net, loss_fct, optimizer, device):
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.net = net
        self.loss_fct = loss_fct
        self.optimizer = optimizer
        self.device= device
        self.save_best_model = SaveBestModel()

    def train_loop(self):
        size = len(self.train_dataloader.dataset)
        num_batches = len(self.val_dataloader)
        train_loss, train_acc = 0, 0

        for batch, (X, y) in enumerate(self.train_dataloader):
            # Compute prediction and loss
            pred = self.net(X.float().to(self.device))   #pas sûre que ça fasse un forward pass
            loss = self.loss_fct(pred, y.to(self.device)) #pred.float() ?

            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            loss += loss.item()/num_batches
            y_pred_class = pred.argmax()
            train_acc += (y_pred_class.to(self.device) == y.to(self.device)).sum()/num_batches

        print(f"Train Error: \n Accuracy: {100*train_acc}%, Avg loss: {loss:>8f} \n")

    def validation_loop(self):
        size = len(self.val_dataloader.dataset)
        num_batches = len(self.val_dataloader)
        test_loss, correct = 0, 0
        y_val, y_pred = [], []

        with torch.no_grad():
            for X, y in self.val_dataloader:
                y_val.extend(y.tolist())
                pred = self.net(X.float().to(self.device))
                y_pred.extend(pred.argmax(1).tolist())
                test_loss += self.loss_fct(pred, y.to(self.device)).item()
                correct += (pred.argmax(1).to(self.device) == y.to(self.device)).type(torch.float).sum().item()

        #print(f"pred = {y_pred}")
        #print(f"true = {y_val}")


        test_loss /= num_batches
        correct /= size
        print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
        scores = {'balanced_accuracy': balanced_accuracy_score(y_val, y_pred),
                'cohen_kappa': cohen_kappa_score(y_val, y_pred),
                'macro_f1': f1_score(y_val, y_pred,average ='macro')}
        print(scores)

        self.save_best_model(test_loss, self.net, "model_cnn_16.pth")

        #cm = confusion_matrix(y_true=y_val, y_pred=y_pred)
        #disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        #disp.plot()


class SaveBestModel:
    """
    Class to save tahe best model while training. If the current epoch's 
    validation loss is less than the previous least less, then save the
    model state.
    """
    def __init__(
        self, best_valid_loss=float('inf')
    ):
        self.best_valid_loss = best_valid_loss
        
    def __call__(self, current_valid_loss, model, name_model):
        if current_valid_loss < self.best_valid_loss:
            self.best_valid_loss = current_valid_loss
            print(f"\nBest validation loss: {self.best_valid_loss}")
            print(f"\nSaving best model\n")
            torch.save(model.state_dict(), os.path.join('/gpfs/workdir/cousteixc/dose_matrices/saved_models', name_model))