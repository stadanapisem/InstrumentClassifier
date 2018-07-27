from math import floor
from pathlib import Path

import dill as pickle
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import RandomSampler, SequentialSampler
import torch
import time


def save_obj(obj, name):
    with open(SAVE_PATH / name, 'wb') as f:
        pickle.dump(obj, f, protocol=pickle.DEFAULT_PROTOCOL)


def load_obj(name):
    with open(SAVE_PATH / name, 'rb') as f:
        return pickle.load(f)


DATA_PATH = Path("../../dataset")
SAVE_PATH = Path("/opt/project")
DATA_FILE = "data.pickle"
LAB_IDX_FILE = "to_idx.pickle"
IDX_LAB_FILE = "to_lab.pickle"

torch.set_default_tensor_type('torch.cuda.DoubleTensor')

training_loss = []
training_acc = []
training_f1 = []
training_time = []

validation_loss = []
validation_acc = []
validation_prec = []
validation_rec = []
validation_f1 = []
validation_time = []
conf_matrix_best = []


def metrics(predictions, targets, thresh=0.5):
    true_positives = 0
    true_negatives = 0
    false_positives = 0
    false_negatives = 0
    conf_matrix = np.zeros((train_data_set.label_size, train_data_set.label_size), dtype=int)

    for i in range(len(predictions)):
        conf_matrix[np.argmax(targets[i]).data[0]][np.argmax(predictions[i]).data[0]] += 1
        if np.max(predictions[i]) > thresh:
            if np.argmax(predictions[i]).data[0] == np.argmax(targets[i]).data[0]:
                true_positives += 1
            else:
                false_positives += 1
        else:
            if np.argmax(predictions[i]).data[0] == np.argmax(targets[i]).data[0]:
                true_negatives += 1
            else:
                false_negatives += 1

    acc = (true_positives + true_negatives) / (true_positives + true_negatives + false_negatives + false_positives)
    try:
        precision = true_positives / (true_positives + false_positives)
    except ZeroDivisionError:
        precision = 0

    try:
        recall = true_positives / (true_positives + false_negatives)
    except ZeroDivisionError:
        recall = 0

    try:
        f1 = 2 * precision * recall / (precision + recall)
    except ZeroDivisionError:
        f1 = 0

    return acc, precision, recall, f1, conf_matrix


class InstrumentData(Dataset):
    data = []
    labels = []
    label_size = 0

    def __init__(self, name):
        tmp_data = load_obj(name)
        label_to_idx = load_obj(LAB_IDX_FILE)
        self.label_size = len(label_to_idx)

        for key, val in tmp_data.items():
            for unwrap in val:
                for value in unwrap:
                    self.labels.append(label_to_idx[key])
                    self.data.append(value)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        ret_data = torch.from_numpy(np.asarray(self.data[item]))
        label_list = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        # label_list = [0, 0]
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
        self.linear2 = nn.Linear(60 * 12, 150)
        self.linear3 = nn.Linear(150, 15)

        # self.batchnorm1 = nn.BatchNorm2d(64)
        self.batchnorm2 = nn.BatchNorm2d(8)

    def forward(self, x):
        x.unsqueeze_(1)

        x = self.conv1(x)
        # x = self.batchnorm1(x)
        x = F.elu(x)
        # x = torch.tanh(x)
        x = self.avgpool1(x)

        x = self.conv2(x)
        x = F.elu(x)
        # x = torch.tanh(x)
        x = self.batchnorm2(x)
        x = F.dropout(x, training=self.training, p=0.5)
        x = self.avgpool2(x)

        x = self.conv3(x)
        x = F.elu(x)
        # x = torch.tanh(x)
        x = F.dropout(x, training=self.training, p=0.5)
        # x = self.avgpool3(x) not needed before a fully-conected layer

        x = x.view(x.size(0), -1)
        x = self.linear1(x)
        x = F.elu(x)
        x = self.linear2(x)
        x = F.elu(x)
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
        acc, _, _, f1, _ = metrics(np.vstack(output.cpu().detach().numpy()), np.vstack(target.cpu().detach().numpy()))

        training_loss.append((batch, loss.item()))
        training_acc.append((batch, acc))
        training_f1.append((batch, f1))

        if batch % 15 == 0:
            print('Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAcc: {:.6f}%\tF1: {:.6f}'.format(epoch, batch * len(data),
                                                                                               len(data_loader.dataset),
                                                                                               100. * batch / len(
                                                                                                   data_loader),
                                                                                               loss.item(), 100. * acc,
                                                                                               f1))


def validate(epoch, data_loader, model, loss_func):
    model.eval()

    total_loss = 0
    predictions = []
    true_labels = []

    for batch, (data, target) in enumerate(data_loader):
        true_labels.append(target.cpu().numpy())

        data, target = data.cuda(async=True), target.cuda(async=True)
        data, target = Variable(data), Variable(target).type(
            torch.cuda.DoubleTensor)  # can add volatile = True for less memory usage

        pred = model(data)
        predictions.append(F.softmax(pred, dim=1).data.cpu().numpy())

        loss = loss_func(pred, torch.max(target, 1)[1]).cuda()
        total_loss += loss.item()

        acc, p, r, f1, _ = metrics(np.vstack(F.softmax(pred, dim=1).data.cpu().numpy()), np.vstack(target.cpu().numpy()))

        validation_loss.append((batch, loss.item()))
        validation_acc.append((batch, acc))
        validation_f1.append((batch, f1))
        validation_prec.append((batch, p))
        validation_rec.append((batch, r))

    avg_loss = total_loss / len(data_loader)

    predictions = np.vstack(predictions)
    true_labels = np.vstack(true_labels)
    acc, prec, rec, f1, conf = metrics(predictions, true_labels)

    print('epoch: {}\t\tValidation - Avg Loss: {:.4f}\tAcc: {:.4f}%\tPrec: {:.4f} Rec: {:.4f} F1: {:.4f}'.
          format(epoch,
                 avg_loss,
                 100. * acc,
                 prec,
                 rec,
                 f1))
    return f1, conf


if __name__ == '__main__':
    BATCH_SIZE = 64

    train_data_set = InstrumentData(DATA_FILE)
    validation_data_set = InstrumentData(DATA_FILE)

    train_idx, validation_idx = dataset_split(train_data_set, shuffle=True)
    train_sampler = RandomSampler(train_idx)
    validation_sampler = SequentialSampler(validation_idx)

    train_loader = DataLoader(train_data_set, batch_size=BATCH_SIZE, sampler=train_sampler, num_workers=1,
                              pin_memory=True)
    validation_loader = DataLoader(validation_data_set, batch_size=BATCH_SIZE, sampler=validation_sampler,
                                   num_workers=1,
                                   pin_memory=True)

    model = NeuralNet().cuda()
    # model = torch.load("model1.pytorch")

    optimizer = optim.Adam(model.parameters())
    loss_function = nn.CrossEntropyLoss()
    max_f1 = -1

    for epoch in range(5):
        curr = time.time()
        train(epoch, train_loader, model, loss_function, optimizer)
        print('Train time: {:.6f}'.format(time.time() - curr))
        training_time.append((epoch, time.time() - curr))

        curr = time.time()
        f1, c = validate(0, validation_loader, model, loss_function)
        validation_time.append((epoch, time.time() - curr))

        if f1 > max_f1:
            conf_matrix_best = c
            torch.save(model, 'model_best.pytorch')

        save_obj(training_loss, 'train_loss.data')
        save_obj(training_acc, 'train_acc.data')
        save_obj(training_f1, 'train_f1.data')
        save_obj(training_time, 'train_time.data')

        save_obj(validation_loss, 'validation_loss.data')
        save_obj(validation_acc, 'validation_acc.data')
        save_obj(validation_prec, 'validation_prec.data')
        save_obj(validation_rec, 'validation_rec.data')
        save_obj(validation_f1, 'validation_f1.data')
        save_obj(validation_time, 'validation_time.data')

        save_obj(conf_matrix_best, 'confusion_matrix.data')
