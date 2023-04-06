import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.autograd import Variable
from dataloader import read_bci_data
import numpy as np
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt


class DeepConvNet(nn.Module):
    def __init__(self, activation='elu'):
        super(DeepConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 25, kernel_size=(1, 5), stride=(1, 1), bias=True)
        self.conv2 = nn.Conv2d(25, 25, kernel_size=(2, 1), stride=(1, 1), bias=True)
        self.batchnorm1 = nn.BatchNorm2d(25, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.dropout1 = nn.Dropout(p=0.5)
        self.conv3 = nn.Conv2d(25, 50, kernel_size=(1, 5), stride=(1, 1), bias=True)
        self.batchnorm2 = nn.BatchNorm2d(50, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.dropout2 = nn.Dropout(p=0.5)
        self.conv4 = nn.Conv2d(50, 100, kernel_size=(1, 5), stride=(1, 1), bias=True)
        self.batchnorm3 = nn.BatchNorm2d(100, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.dropout3 = nn.Dropout(p=0.5)
        self.conv5 = nn.Conv2d(100, 200, kernel_size=(1, 5), stride=(1, 1), bias=True)
        self.batchnorm4 = nn.BatchNorm2d(200, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.dropout4 = nn.Dropout(p=0.5)
        self.batchnorm5 = nn.BatchNorm1d(200, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.dropout5 = nn.Dropout(p=0.5)
        self.fc = nn.Linear(8600, 2, bias=True)

        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'elu':
            self.activation = nn.ELU(alpha=1.0)
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(negative_slope=0.01)
        else:
            raise ValueError('Invalid activation function')

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.batchnorm1(x)
        x = self.activation(x)
        x = F.max_pool2d(x, kernel_size=(1, 2), stride=(1, 2))
        x = self.dropout1(x)
        x = self.conv3(x)
        x = self.batchnorm2(x)
        x = self.activation(x)
        x = F.max_pool2d(x, kernel_size=(1, 2), stride=(1, 2))
        x = self.dropout2(x)
        x = self.conv4(x)
        x = self.batchnorm3(x)
        x = self.activation(x)
        x = F.max_pool2d(x, kernel_size=(1, 2), stride=(1, 2))
        x = self.dropout3(x)
        x = self.conv5(x)
        x = self.batchnorm4(x)
        x = self.activation(x)
        x = F.max_pool2d(x, kernel_size=(1, 2), stride=(1, 2))
        x = self.dropout4(x)
        x = x.view(x.size(0), -1)
        # print(x.size())
        x = self.fc(x)

        return x


class EEGNet(nn.Module):
    def __init__(self, activation='elu'):
        super(EEGNet, self).__init__()

        # Layer 1
        self.conv1 = nn.Conv2d(1, 16, kernel_size=(1, 51), stride=(1, 1), padding=(0, 25), bias=False)
        self.batchnorm1 = nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        # Layer 2
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(2, 1), stride=(1, 1), groups=16, bias=False)
        self.batchnorm2 = nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.pooling2 = nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4), padding=0)

        # Layer 3
        self.conv3 = nn.Conv2d(32, 32, kernel_size=(1, 15), stride=(1, 1), padding=(0, 7), bias=False)
        self.batchnorm3 = nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.pooling3 = nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8), padding=0)

        # Activation function
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'elu':
            self.activation = nn.ELU(alpha=1.0)
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(negative_slope=0.01)
        else:
            raise ValueError('Invalid activation function')

        # Fully connected layer
        self.fc = nn.Linear(736, 2, bias=True)

    def forward(self, x):
        # Layer 1
        x = self.conv1(x)
        x = self.batchnorm1(x)

        # Layer 2
        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = self.activation(x)
        x = self.pooling2(x)
        x = nn.functional.dropout(x, p=0.25)

        # Layer 3
        x = self.conv3(x)
        x = self.batchnorm3(x)
        x = self.activation(x)
        x = self.pooling3(x)
        x = nn.functional.dropout(x, p=0.25)

        # Flatten
        x = x.view(x.size(0), -1)

        # Fully connected layer
        x = self.fc(x)

        return x


def evaluate(model, Loader):

    for i, batch in enumerate(Loader):
        inputs, labels = batch
        inputs = inputs.to("cuda")
        labels = np.asarray(labels)
        predicted = model(inputs)

        mask = predicted[:, 0] < predicted[:, 1]
        result = np.where(mask.cpu(), 1, 0)
        labels = labels.astype(np.int32)

        results = accuracy_score(labels, result)

        return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model", default="eegnet", help="please input your model name, eegnet or deepconvnet")
    args = parser.parse_args()

    if args.model == "eegnet":
        net_elu = EEGNet(activation="elu").cuda(0)
        net_relu = EEGNet(activation="relu").cuda(0)
        net_lrelu = EEGNet(activation="leaky_relu").cuda(0)
    elif args.model == "deepconvnet":
        net_elu = DeepConvNet(activation="elu").cuda(0)
        net_relu = DeepConvNet(activation="relu").cuda(0)
        net_lrelu = DeepConvNet(activation="leaky_relu").cuda(0)
    criterion = nn.CrossEntropyLoss()  # DOn't forget to add softmax layer if change to another loss function
    Batch_size = 64
    Learning_rate = 1e-2
    Epochs = 200
    optimizer_elu = optim.Adam(net_elu.parameters(), lr=Learning_rate)
    optimizer_relu = optim.Adam(net_relu.parameters(), lr=Learning_rate)
    optimizer_lrelu = optim.Adam(net_lrelu.parameters(), lr=Learning_rate)

    train_X, train_Y, test_X, test_Y = read_bci_data()  # load data from dataloader
    train_dataset = TensorDataset(torch.Tensor(train_X), torch.Tensor(train_Y))
    test_dataset = TensorDataset(torch.Tensor(test_X), torch.Tensor(test_Y))

    train_data = DataLoader(train_dataset, batch_size=Batch_size, shuffle=True)
    test_data = DataLoader(test_dataset, batch_size=1080)

    train_accuracy_elu = []
    test_accuracy_elu = []
    for epoch in range(Epochs):

        running_loss = 0.0
        # epoch_bar = tqdm(range(len(train_X)//Batch_size + int(bool(len(train_X) % Batch_size))), desc='Epoch %d' % (epoch+1))  # deal with the last one batch
        train_accuracy_batch = []
        test_accuracy_batch = []
        for i, batch in enumerate(tqdm(train_data, desc='Epoch %d' % (epoch+1))):
            inputs, labels = batch
            labels = labels.to("cuda")
            inputs = inputs.to("cuda")
            # zero the parameter gradients
            optimizer_elu.zero_grad()

            # forward + backward + optimize
            outputs = net_elu(inputs)

            loss = criterion(outputs, torch.squeeze(labels.long()))  # CrossEntropyLoss' target need to be long
            loss.backward()

            optimizer_elu.step()

            mask = outputs[:, 0] < outputs[:, 1]
            result = np.where(mask.cpu(), 1, 0)
            labels = np.asanyarray(labels.cpu(), dtype=np.int32)
            train_accuracy_batch.append(accuracy_score(labels, result))

            # print("loss: {}\n".format(loss.data))
            running_loss += loss.data.item()

            test_accuracy_batch.append(evaluate(net_elu, test_data))

        tqdm.write(f'Epoch {epoch+1}, Loss: {running_loss:.4f}')
        train_accuracy_elu.append(np.mean(train_accuracy_batch))
        test_accuracy_elu.append(np.mean(test_accuracy_batch))

    print("train_accuracy_elu: {}".format(train_accuracy_elu))
    print("test_accuracy_elu: {}".format(test_accuracy_elu))
    plt.plot(np.arange(len(train_accuracy_elu)), train_accuracy_elu, color='red', linestyle="-", markersize="16", label="train_elu")
    plt.plot(np.arange(len(test_accuracy_elu)), test_accuracy_elu, color='blue', linestyle="-", markersize="16", label="test_elu")

    train_accuracy_relu = []
    test_accuracy_relu = []
    for epoch in range(Epochs):

        running_loss = 0.0
        # epoch_bar = tqdm(range(len(train_X)//Batch_size + int(bool(len(train_X) % Batch_size))), desc='Epoch %d' % (epoch+1))  # deal with the last one batch
        train_accuracy_batch = []
        test_accuracy_batch = []
        for i, batch in enumerate(tqdm(train_data, desc='Epoch %d' % (epoch+1))):
            inputs, labels = batch
            labels = labels.to("cuda")
            inputs = inputs.to("cuda")
            # zero the parameter gradients
            optimizer_relu.zero_grad()

            # forward + backward + optimize
            outputs = net_relu(inputs)

            loss = criterion(outputs, torch.squeeze(labels.long()))  # CrossEntropyLoss' target need to be long
            loss.backward()

            optimizer_relu.step()

            mask = outputs[:, 0] < outputs[:, 1]
            result = np.where(mask.cpu(), 1, 0)
            labels = np.asanyarray(labels.cpu(), dtype=np.int32)
            train_accuracy_batch.append(accuracy_score(labels, result))

            # print("loss: {}\n".format(loss.data))
            running_loss += loss.data.item()

            test_accuracy_batch.append(evaluate(net_relu, test_data))
        tqdm.write(f'Epoch {epoch+1}, Loss: {running_loss:.4f}')
        train_accuracy_relu.append(np.mean(train_accuracy_batch))
        test_accuracy_relu.append(np.mean(test_accuracy_batch))

    print("train_accuracy_relu: {}".format(train_accuracy_relu))
    print("test_accuracy_relu: {}".format(test_accuracy_relu))
    plt.plot(np.arange(len(train_accuracy_relu)), train_accuracy_relu, color='brown', linestyle="-", markersize="16", label="train_relu")
    plt.plot(np.arange(len(test_accuracy_relu)), test_accuracy_relu, color='pink', linestyle="-", markersize="16", label="test_relu")

    train_accuracy_lrelu = []
    test_accuracy_lrelu = []
    for epoch in range(Epochs):

        running_loss = 0.0
        # epoch_bar = tqdm(range(len(train_X)//Batch_size + int(bool(len(train_X) % Batch_size))), desc='Epoch %d' % (epoch+1))  # deal with the last one batch
        train_accuracy_batch = []
        test_accuracy_batch = []
        for i, batch in enumerate(tqdm(train_data, desc='Epoch %d' % (epoch+1))):
            inputs, labels = batch
            labels = labels.to("cuda")
            inputs = inputs.to("cuda")
            # zero the parameter gradients
            optimizer_lrelu.zero_grad()

            # forward + backward + optimize
            outputs = net_lrelu(inputs)

            loss = criterion(outputs, torch.squeeze(labels.long()))  # CrossEntropyLoss' target need to be long
            loss.backward()

            optimizer_lrelu.step()

            mask = outputs[:, 0] < outputs[:, 1]
            result = np.where(mask.cpu(), 1, 0)
            labels = np.asanyarray(labels.cpu(), dtype=np.int32)
            train_accuracy_batch.append(accuracy_score(labels, result))

            # print("loss: {}\n".format(loss.data))
            running_loss += loss.data.item()

            test_accuracy_batch.append(evaluate(net_lrelu, test_data))
        tqdm.write(f'Epoch {epoch+1}, Loss: {running_loss:.4f}')
        train_accuracy_lrelu.append(np.mean(train_accuracy_batch))
        test_accuracy_lrelu.append(np.mean(test_accuracy_batch))

    print("train_accuracy_lrelu: {}".format(train_accuracy_lrelu))
    print("test_accuracy_lrelu: {}".format(test_accuracy_lrelu))
    plt.plot(np.arange(len(train_accuracy_lrelu)), train_accuracy_lrelu, color='olive', linestyle="-", markersize="16", label="train_lrelu")
    plt.plot(np.arange(len(test_accuracy_lrelu)), test_accuracy_lrelu, color='cyan', linestyle="-", markersize="16", label="test_lrelu")
    # plt.plot(len(train_accuracy), train_accuracy, color='red', linestyle="-", marker=".", label="train_accuracy")
    # plt.plot(len(test_accuracy), test_accuracy, color='blue', linestyle="-", marker=".", label="test_accuracy")
    plt.xlabel('epoches')
    plt.ylabel('accuracy')
    plt.title('comparison')
    plt.legend()
    plt.savefig('model_{}.png'.format(args.model))
    # plt.legend()
    # plt.show()
