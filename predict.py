import os
import pandas as pd
import torch
from model import CNN, FirstNet, CNN_tho, NewNet
from dataset import TestDataset, FeatTestDataset
from torch.utils.data import DataLoader
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, accuracy_score, ConfusionMatrixDisplay, recall_score, roc_auc_score
import matplotlib.pyplot as plt
from os.path import join

path_to_dir = "/gpfs/workdir/cousteixc/dose_matrices"
path_to_models = "/gpfs/workdir/cousteixc/dose_matrices/saved_models"
batch_size = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def predict(model, test_loader):
    model.eval()
    preds = []
    y_true = []
    scores = []
    for batch_idx, (data, labels) in enumerate(test_loader):
        y_true.extend(labels)
        model = model.to(DEVICE)
        labels = labels.to(DEVICE)
        if model.__class__.__name__ == "NewNet":
            output = model(data[0].to(DEVICE), data[1].to(DEVICE))
        else : 
            data = data.to(DEVICE)
            output = model(data.float())
        pred = output.argmax(dim=1, keepdim=True)
        scores.extend(output.cpu().detach().numpy()[:, 1])
        preds += pred.tolist()

    return y_true, preds, scores

for model in os.listdir(path_to_models):
    if str(model)[-3:] == "pth":
        print(model)
        model_split = model.split("_")
        min_card_age = int(model_split[3])

        test_data = TestDataset("labels_ruche.csv", path_to_dir, min_card_age, testing = True)

        if str(model)[0] == '0':
            net = NewNet(3, 15, 10)
            test_data = FeatTestDataset("labels_ruche.csv", path_to_dir, min_card_age, testing = True)
        elif str(model)[0] == '1':
            net = CNN(16)
        elif str(model)[0] == '2':
            net = CNN_tho(16)
        elif str(model)[0] == '3':
            net = FirstNet()

        test_dataloader = DataLoader(test_data, batch_size, shuffle=False)
        net = net.to(DEVICE)
        net.load_state_dict(torch.load(os.path.join(path_to_models, model)))
        y_true, y_pred, scores = predict(net, test_dataloader)

        cm = confusion_matrix(y_true, y_pred, labels =[0,1])
        disp = ConfusionMatrixDisplay(cm)
        disp.plot()
        plt.savefig(join(path_to_models, "saved_plots_test", f"{str(model)[:-4]}_cm.jpg"))

        lines = [f'Saving results for model {net.__class__.__name__} :', f"Parameters : {str(model)[:-4]}",
                f'acc : {(100*accuracy_score(y_true, y_pred)):>0.1f}',
                f'balanced accuracy : {(100*balanced_accuracy_score(y_true, y_pred)):>0.1f}',
                f'recall : {(100*recall_score(y_true, y_pred)):>0.1f}',
                f'AUC :  {roc_auc_score(y_true, scores)}',
                f'cm : {confusion_matrix(y_true, y_pred)}']

        with open(join(path_to_models, "saved_plots_test", f"{model[:-4]}.txt"), 'a') as f:
            f.write('\n'.join(lines))

        print(f"{model} fini")

