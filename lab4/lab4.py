import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import TensorDataset, DataLoader, Subset
import matplotlib.pyplot as plt
import time
import os
import copy
from dataloader import *
import argparse
from tqdm import tqdm
from PIL import ImageFile
from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sns
ImageFile.LOAD_TRUNCATED_IMAGES=True
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def plot_confusion_matrix(cf_matrix,name):
    class_names = ['no DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR']
    df_cm = pd.DataFrame(cf_matrix, class_names, class_names) 
    sns.heatmap(df_cm, annot=True, cmap='Oranges')
    plt.title(name)
    plt.xlabel("prediction")
    plt.ylabel("laTbel (ground truth)")
    plt.savefig('cf_matrix_{}.png'.format(name))

def makeconfusionmatrix(model, test_dataset):
    y_pred = []
    y_true = []
    model.to(device)
    model.eval()
    test_loader = DataLoader(test_dataset, batch_size=Batch_size, shuffle=True)

    with torch.no_grad():
        for i, (images, target) in enumerate(test_loader):
            images=images.to(device)
            target=target.to(device)
            output = model(images)
            _, preds = torch.max(output, 1) 
            y_pred.extend(preds.view(-1).detach().cpu().numpy())       
            y_true.extend(target.view(-1).detach().cpu().numpy())
            print(i,'/',len(test_loader))
    cf_matrix = confusion_matrix(y_true, y_pred)
    cf_matrix_normalize=confusion_matrix(y_true,y_pred,normalize='true')
    return cf_matrix,cf_matrix_normalize



def evaluate(model, Loader):
    test_accuracy = 0
    model.to(device)
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(tqdm(Loader)):
            inputs, labels = batch
            inputs = inputs.to(device)
            labels = labels.to(device).long()
            output = model(inputs)
            # print("output: {}".format(output))
            _,Predicted=torch.max(output,1)
            # print("labels: {}".format(labels))
            test_accuracy+=(Predicted==labels).sum().item()
    # results = accuracy_score(labels, result)
    return test_accuracy

def train(Epochs, model, criterion, train_dataset, test_dataset, Batch_size, lr, weight_decay):
    test_accuracy_list = []
    train_accuracy_list = []
    model = model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    for epoch in range(Epochs):
        Train_loader = DataLoader(train_dataset, batch_size=Batch_size, shuffle=True)
        Test_loader = DataLoader(test_dataset, batch_size=Batch_size, shuffle=True)

        train_accuracy = 0
        for i, batch in enumerate(tqdm(Train_loader, desc='Epoch %d' % (epoch+1))):
            model.train()
            inputs, labels = batch
            labels = labels.to(device)
            inputs = inputs.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()

            outputs = model(inputs)

            loss = criterion(outputs, torch.squeeze(labels.long()))  # CrossEntropyLoss' target need to be long
            loss.backward()

            optimizer.step()

            _,Predicted=torch.max(outputs,1)
            train_accuracy+=(Predicted==labels).sum().item()

        train_accuracy_list.append(train_accuracy/28100)  # record the accuracy of each epoch, need to divided by 28100


        test_accuracy = evaluate(model, Test_loader)  # record the accuracy of test dataset, need to divided by 7026
        test_accuracy_list.append(test_accuracy/7026)

        print("accuracy: {}".format(test_accuracy))
    print("test_accuracy: {}".format(test_accuracy_list))
    return train_accuracy_list, test_accuracy_list

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained", type=bool, default=True, help="please input False if you want to train from scratch")
    parser.add_argument("--model", default="resnet18", help="please input resnet50 if you want")
    parser.add_argument("--load", type=bool, default=False, help="bool, determine load model or not")
    args = parser.parse_args()
    root = "/home/pp037/deep_learning/lab4/dataset/"
    Batch_size = 8
    learning_rate = 1e-3
    Epochs = 10
    Momentum = 0.9
    Weight_Decay = 5e-4
    torch.cuda.empty_cache()



    Train_dataset = RetinopathyLoader(root=root, mode="train_resize", transform=[
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=10)
    ])
    Test_dataset = RetinopathyLoader(root=root, mode="test_resize", transform=[])

    Sub_Train_dataset = Subset(Train_dataset, indices=np.arange(100))
    Sub_Test_dataset = Subset(Test_dataset, indices=np.arange(100))

    
    criterion = nn.CrossEntropyLoss()
    # if args.model == "resnet18":
    #     model_pre = models.resnet18(pretrained = True)  # pre = pretrained
    #     model_pre.fc = nn.Linear(512, 5)
    #     model_scr = models.resnet18(pretrained = False)  # scr = scratch
    #     model_scr.fc = nn.Linear(512, 5)
    # else:
    #     model_pre = models.resnet50(pretrained = True)  # pre = pretrained
    #     model_pre.fc = nn.Linear(2048, 5)
    #     model_scr = models.resnet50(pretrained = False)  # scr = scratch
    #     model_scr.fc = nn.Linear(2048, 5)

    if args.load is True:
        print("loading model")
        if args.model == "resnet18":
            model_pre = torch.load("weight/resnet18_pretrain.pt")
            model_scr = torch.load("weight/resnet18_scratch.pt")
        else:
            model_pre = torch.load("weight/resnet50_pretrain.pt")
            model_scr = torch.load("weight/resnet50_scratch.pt")
    else:
        if args.model == "resnet18":
            model_pre = models.resnet18(pretrained = True)  # pre = pretrained
            model_pre.fc = nn.Linear(512, 5)
            model_scr = models.resnet18(pretrained = False)  # scr = scratch
            model_scr.fc = nn.Linear(512, 5)
        else:
            model_pre = models.resnet50(pretrained = True)  # pre = pretrained
            model_pre.fc = nn.Linear(2048, 5)
            model_scr = models.resnet50(pretrained = False)  # scr = scratch
            model_scr.fc = nn.Linear(2048, 5)

    Train_pre_accuracy_list, Test_pre_accuracy_list = train(Epochs=Epochs, 
                                                        model=model_pre,
                                                        criterion=criterion,
                                                        train_dataset=Train_dataset,
                                                        test_dataset=Test_dataset,
                                                        Batch_size=Batch_size,
                                                        lr=learning_rate,
                                                        weight_decay=Weight_Decay)
    
    confuse_pre_matrix, conffuse_pre_matrix_nor = makeconfusionmatrix(model_pre, Test_dataset)
    plot_confusion_matrix(conffuse_pre_matrix_nor, args.model + "pretrain")
    plt.clf()

    torch.save(model_pre, "./weight/{}_pretrain.pt".format(args.model))
    torch.cuda.empty_cache()


    Train_scr_accuracy_list, Test_scr_accuracy_list = train(Epochs=Epochs,
                                                        model=model_scr,
                                                        criterion=criterion,
                                                        train_dataset=Train_dataset,
                                                        test_dataset=Test_dataset,
                                                        Batch_size=Batch_size,
                                                        lr=learning_rate,
                                                        weight_decay=Weight_Decay)    
    

    confuse_scr_matrix, conffuse_scr_matrix_nor = makeconfusionmatrix(model_scr, Test_dataset)
    plot_confusion_matrix(conffuse_scr_matrix_nor, args.model + "scratch")
    plt.clf()
    torch.save(model_scr, "./weight/{}_scratch.pt".format(args.model))

    plt.plot(np.arange(len(Train_pre_accuracy_list)), Train_pre_accuracy_list, color='red', linestyle="-", markersize="16", label="Train_pre_accuracy")
    plt.plot(np.arange(len(Test_pre_accuracy_list)), Test_pre_accuracy_list, color='blue', linestyle="-", markersize="16", label="Test_pre_accuracy")
    plt.plot(np.arange(len(Train_scr_accuracy_list)), Train_scr_accuracy_list, color='brown', linestyle="-", markersize="16", label="Train_scr_accuracy")
    plt.plot(np.arange(len(Test_scr_accuracy_list)), Test_scr_accuracy_list, color='green', linestyle="-", markersize="16", label="Test_scr_accuracy")

    plt.xlabel('epoches')
    plt.ylabel('accuracy')
    plt.title('Result comparison({})'.format(args.model))
    plt.legend()
    plt.savefig('model_{}.png'.format(args.model))
    