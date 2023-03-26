import os
import pandas as pd
import torch
from model import CNN
from dataset import TestDataset
from torch.utils.data import DataLoader
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score, confusion_matrix,f1_score, ConfusionMatrixDisplay


path_to_dir = "/gpfs/workdir/cousteixc/dose_matrices"
min_card_age = 50
batch_size = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


test_data = TestDataset("labels_ruche.csv", path_to_dir, min_card_age, testing = True)
test_dataloader = DataLoader(test_data, batch_size, shuffle=False)

def predict(model, test_loader):
    model.eval()
    preds = []
    y_true = []
    for batch_idx, (data, labels) in enumerate(test_loader):
        #print(f'Predicting batch {batch_idx+1}/{len(test_loader)}')
        y_true.extend(labels)
        model = model.to(DEVICE)
        labels = labels.to(DEVICE)
        data = data.to(DEVICE)
        output = model(data)
        pred = output.argmax(dim=1, keepdim=True)
        preds += pred.tolist()

    return y_true, preds

net = CNN(16)
net.load_state_dict(torch.load("/gpfs/workdir/cousteixc/dose_matrices/saved_models/model_cnn_16.pth"))
y_true, y_pred = predict(net, test_dataloader)
print(y_pred)

scores = {'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
                'cohen_kappa': cohen_kappa_score(y_true, y_pred),
                'macro_f1': f1_score(y_true, y_pred,average ='macro')}
print(scores)



