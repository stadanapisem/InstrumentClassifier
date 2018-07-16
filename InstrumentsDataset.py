from math import floor
from pathlib import Path

import dill as pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from tqdm import tqdm
import torch


def save_obj(obj, name):
    with open(SAVE_PATH / name, 'wb') as f:
        pickle.dump(obj, f, protocol=pickle.DEFAULT_PROTOCOL)


def load_obj(name):
    with open(SAVE_PATH / name, 'rb') as f:
        return pickle.load(f)


DATA_PATH = Path("../../dataset")
SAVE_PATH = Path("/opt/project")
DATA_FILE = "data_small.pickle"
LAB_IDX_FILE = "to_idx.pickle"
IDX_LAB_FILE = "to_lab.pickle"

torch.set_default_tensor_type('torch.cuda.DoubleTensor')


class InstrumentData(Dataset):
    data = []
    labels = []

    def __init__(self, name):
        tmp_data = load_obj(name)
        label_to_idx = load_obj(LAB_IDX_FILE)

        for key, val in tmp_data.items():
            for unwrap in val:
                for value in unwrap:
                    self.labels.append(label_to_idx[key])
                    self.data.append(value)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        ret_data = torch.from_numpy(np.asarray(self.data[item]))
        # label_list = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        label_list = [0, 0]
        label_list[self.labels[item]] = 1
        ret_label = torch.from_numpy(np.asarray(label_list))
        return ret_data, ret_label


class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()

        self.conv1 = nn.Conv2d(1, 4, kernel_size=(4, 2), stride=(2, 1))  # (1x438x13) -> (4x240x12)
        self.avgpool1 = nn.AvgPool2d(3, padding=(1, 1), stride=1)  # (4x240x12) -> (4x240x12)
        self.conv2 = nn.Conv2d(4, 8, kernel_size=(4, 3), padding=(1, 1),
                               stride=(2, 1))  # (4x240x12) -> (8x120x12)
        self.avgpool2 = nn.AvgPool2d(3, padding=(1, 1), stride=1)  # (8x120x12) -> (8x120x12)
        self.conv3 = nn.Conv2d(8, 16, kernel_size=(4, 3), padding=(1, 1),
                               stride=(2, 1))  # (8x120x12) -> (16x60x12)
        self.avgpool3 = nn.AvgPool2d(3, padding=(1, 1), stride=1)  # (16x60x12) -> (16x60x12)

        # nn.init.xavier_normal_(self.conv1.weight)
        # nn.init.xavier_normal_(self.conv2.weight)
        # nn.init.xavier_normal_(self.conv3.weight)

        self.linear1 = nn.Linear(16 * 60 * 12, 60 * 12)
        self.linear2 = nn.Linear(60 * 12, 12)
        self.linear3 = nn.Linear(12, 2)

        # self.batchnorm1 = nn.BatchNorm2d(64)
        self.batchnorm2 = nn.BatchNorm2d(8)

    def forward(self, x):
        x.unsqueeze_(1)

        x = self.conv1(x)
        # x = self.batchnorm1(x)
        # x = F.elu(x)
        x = torch.tanh(x)
        x = self.avgpool1(x)

        x = self.conv2(x)
        # x = F.elu(x)
        x = torch.tanh(x)
        x = self.batchnorm2(x)
        x = F.dropout(x, training=self.training, p=0.5)
        x = self.avgpool2(x)

        x = self.conv3(x)
        x = torch.tanh(x)
        x = F.dropout(x, training=self.training, p=0.5)
        # x = self.avgpool3(x) not needed before a fully-conected layer

        x = x.view(x.size(0), -1)
        x = self.linear1(x)
        x = torch.tanh(x)
        x = self.linear2(x)
        x = self.linear3(x)

        # if not self.training:
        #    x = torch.softmax(x, dim=0)

        return x


def dataset_split(dataset, test_size=0.3, shuffle=False):
    length = dataset.__len__()
    idx = list(range(1, length))

    if shuffle:
        np.random.seed()
        idx = np.random.permutation(len(idx))

    split = floor(test_size * length)

    return idx[split:], idx[:split]


def train(epoch, data_loader, model, loss_func, optimizer):
    model.train()

    for batch, (data, target) in enumerate(data_loader):
        data, target = data.cuda(async=True), target.cuda(async=True)
        data, target = Variable(data), Variable(target).type(torch.cuda.DoubleTensor)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_func(output, torch.max(target, 1)[1]).cuda()
        loss.backward()
        optimizer.step()

        if batch % 20 == 0:
            print('Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch * len(data), len(data_loader.dataset),
                                                                     100. * batch / len(data_loader), loss.item()))


def validate(epoch, data_loader, model, loss_func):
    model.eval()

    total_loss = 0
    predictions = []
    true_labels = []

    for batch, (data, target) in enumerate(tqdm(data_loader)):
        true_labels.append(target.cpu().numpy())

        data, target = data.cuda(async=True), target.cuda(async=True)
        data, target = Variable(data), Variable(target).type(
            torch.cuda.DoubleTensor)  # can add volatile = True for less memory usage

        pred = model(data)
        predictions.append(F.softmax(pred, dim=1).data.cpu().numpy())
        loss = loss_func(pred, torch.max(target, 1)[1]).cuda()
        total_loss += loss.item()

    avg_loss = total_loss / len(data_loader)

    predictions = np.vstack(predictions)
    true_labels = np.vstack(true_labels)

    score = 0  # should use a better metric - F2 ?
    for i in range(len(predictions)):
        if np.argmax(predictions[i]).data[0] == np.argmax(true_labels[i]).data[0]:
            score += 1

    print('\t\tValidation - Avg Loss: {:.4f}\tAcc: {:.4f}%'.format(avg_loss, 100. * score / len(predictions)))


BATCH_SIZE = 64

train_data_set = InstrumentData(DATA_FILE)
validation_data_set = InstrumentData(DATA_FILE)

train_idx, validation_idx = dataset_split(train_data_set, shuffle=True)
train_sampler = RandomSampler(train_idx)
validation_sampler = SequentialSampler(validation_idx)

train_loader = DataLoader(train_data_set, batch_size=BATCH_SIZE, sampler=train_sampler, num_workers=1, pin_memory=True)
validation_loader = DataLoader(validation_data_set, batch_size=BATCH_SIZE, sampler=validation_sampler, num_workers=1,
                               pin_memory=True)

model = NeuralNet().cuda()
optimizer = optim.SGD(model.parameters(), lr=0.0006, momentum=0.9, weight_decay=0.05)
loss_function = nn.CrossEntropyLoss()

for epoch in range(3):
    train(epoch, train_loader, model, loss_function, optimizer)
    torch.save(model, 'model1.pytorch')
    # model = torch.load("model1.pytorch")
    validate(0, validation_loader, model, loss_function)
