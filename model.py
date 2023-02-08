import torch.nn as nn
import torch
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score, confusion_matrix,f1_score, ConfusionMatrixDisplay



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


class Loop():

    def __init__(self, train_dataloader, val_dataloader, net, loss_fct, optimizer):
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.net = net
        self.loss_fct = loss_fct
        self.optimizer = optimizer


    def train_loop(self):
        size = len(self.train_dataloader.dataset)
        for batch, (X, y) in enumerate(self.train_dataloader):
            # Compute prediction and loss
            pred = self.net(X.float())   #pas sûre que ça fasse un forward pass
            loss = self.loss_fct(pred, y) #pred.float() ?

            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if batch % 10 == 0:
                loss, current = loss.item(), batch * len(X)
                y_pred_class = pred.argmax()
                train_acc = (y_pred_class == y).sum()
                print(f"loss: {loss:>7f}, accuracy: {train_acc}  [{current:>5d}/{size:>5d}]")


    def validation_loop(self):
        size = len(self.val_dataloader.dataset)
        num_batches = len(self.val_dataloader)
        test_loss, correct = 0, 0
        y_val, y_pred = [], []

        with torch.no_grad():
            for X, y in self.val_dataloader:
                y_val.extend(y.tolist())
                pred = self.net(X.float())
                y_pred.extend(pred.argmax(1).tolist())
                test_loss += self.loss_fct(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        print(f"pred = {y_pred}")
        print(f"true = {y_val}")


        test_loss /= num_batches
        correct /= size
        print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
        scores = {'balanced_accuracy': balanced_accuracy_score(y_val, y_pred),
                'cohen_kappa': cohen_kappa_score(y_val, y_pred),
                'macro_f1': f1_score(y_val, y_pred,average ='macro')}
        print(scores)

        #cm = confusion_matrix(y_true=y_val, y_pred=y_pred)
        #disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        #disp.plot()