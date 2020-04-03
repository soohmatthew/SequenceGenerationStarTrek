from __future__ import unicode_literals, print_function, division
from io import open
import glob
import os
import unicodedata
import string
import csv
import numpy as np
import torch
import time
import math
import random
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import sys

from tqdm import tqdm as tqdm

all_letters = string.ascii_letters + "0123456789 .,:!?'[]()/+-="
n_letters = len(all_letters) + 1 # Plus EOS marker

# Dataloader Class
class iteratefromdict():
    def __init__(self, cat_dict, train, batch_size, seed, test = False, percentage_training = 0.8, percentage_test = 0.1):
        self.train = train
        self.seed = seed
        self.cat_dict = cat_dict
        self.ct = 0
        self.batch_size = batch_size
        self.namlab = cat_dict

        np.random.seed(self.seed)
        np.random.shuffle(self.namlab)
        if train:
            self.namlab = self.namlab[:int(percentage_training * len(self.namlab))]
        elif test:
            self.namlab = self.namlab[int(percentage_training * len(self.namlab)):int((percentage_training + percentage_test) * len(self.namlab))]
        else:
            self.namlab = self.namlab[int((percentage_training + percentage_test) * len(self.namlab)):]
            
    def num(self):
        return len(self.namlab)
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.ct >= len(self.namlab):
            self.ct = 0
            raise StopIteration()
        else:
            self.ct += self.batch_size
            if self.ct >= len(self.namlab): # If overflow from list, then get from the start
                remainder = self.batch_size - len(self.namlab[self.ct-self.batch_size:])
                return self.namlab[self.ct-self.batch_size:] + self.namlab[:remainder]
            else:
                return self.namlab[self.ct - self.batch_size : self.ct]

    def __len__(self):
        return int(len(self.namlab)/self.batch_size)

# --Code below adapted and modified from Segmentation Models Pytorch on GitHub
class AverageValueMeter(object):
    def __init__(self):
        super(AverageValueMeter, self).__init__()
        self.reset()
        self.val = 0

    def add(self, value, n=1):
        self.val = value
        self.sum += value
        self.var += value * value
        self.n += n

        if self.n == 0:
            self.mean, self.std = np.nan, np.nan
        elif self.n == 1:
            self.mean = 0.0 + self.sum  # This is to force a copy in torch/numpy
            self.std = np.inf
            self.mean_old = self.mean
            self.m_s = 0.0
        else:
            self.mean = self.mean_old + (value - n * self.mean_old) / float(self.n)
            self.m_s += (value - self.mean_old) * (value - self.mean)
            self.mean_old = self.mean
            self.std = np.sqrt(self.m_s / (self.n - 1.0))

    def value(self):
        return self.mean, self.std

    def reset(self):
        self.n = 0
        self.sum = 0.0
        self.var = 0.0
        self.val = 0.0
        self.mean = np.nan
        self.mean_old = 0.0
        self.m_s = 0.0
        self.std = np.nan

class Epoch(object):
    def __init__(self, model, loss, stage_name, device='cpu', verbose=True, logger = None):
        self.model = model
        self.loss = loss
        self.stage_name = stage_name
        self.verbose = verbose
        self.device = device
        self.logger = logger

        self._to_device()

    def _to_device(self):
        self.model.to(self.device)
        self.loss.to(self.device)

    def _format_logs(self, logs):
        str_logs = ['{} - {:.4}'.format(k, v) for k, v in logs.items()]
        s = ', '.join(str_logs)
        return s

    def batch_update(self, x):
        raise NotImplementedError

    def on_epoch_start(self):
        pass
    
    @staticmethod
    def correct_counts(y_pred, y):
        y_pred, y = y_pred.cpu(), y.cpu()
        mask = (y > -1)
        _, indices = torch.max(y_pred, 1)
        return (indices[mask] == y[mask]).float()

    def run(self, dataloader):

        self.on_epoch_start()

        logs = {}
        loss_meter = AverageValueMeter()
        correct_count_meter = AverageValueMeter()

        with tqdm(dataloader, desc=self.stage_name, file=sys.stdout, disable=not (self.verbose)) as iterator:
            # Run for 1 epoch
            for x in iterator:
                loss, y_pred, y = self.batch_update(x)

                # update loss logs
                loss_value = loss.cpu().detach().numpy()
                loss_meter.add(loss_value)
                loss_logs = {'CrossEntropyLoss': loss_meter.mean}
                logs.update(loss_logs)

                # update metrics logs
                metric_value = self.correct_counts(y_pred, y).cpu().detach().numpy()
                correct_count_meter.add(metric_value.sum().item(),n = metric_value.shape[0])
                metrics_logs = {'Accuracy': correct_count_meter.mean}
                logs.update(metrics_logs)

                if self.verbose:
                    s = self._format_logs(logs)
                    iterator.set_postfix_str(s)

        cumulative_logs = {'Accuracy': correct_count_meter.sum/correct_count_meter.n}
        cumulative_logs['loss'] = loss_meter.sum/loss_meter.n
        log_print(" ".join([f"{k}:{v:.4f}" for k, v in cumulative_logs.items()]), self.logger, log_only = True)

        return cumulative_logs

class TrainEpoch(Epoch):
    def __init__(self, model, loss, optimizer, device='cpu', verbose=True, logger = None):
        super().__init__(
            model=model,
            loss=loss,
            stage_name='train',
            device=device,
            verbose=verbose,
            logger=logger
        )
        self.optimizer = optimizer

    def on_epoch_start(self):
        self.model.train()

    def batch_update(self, x):
        self.optimizer.zero_grad()
        prediction, ground_truth = self.model.forward(x)
        loss = self.loss(prediction, ground_truth)
        loss.backward()
        self.optimizer.step()
        return loss, prediction, ground_truth

class ValidEpoch(Epoch):
    def __init__(self, model, loss, device='cpu', verbose=True, logger = None):
        super().__init__(
            model=model,
            loss=loss,
            stage_name='valid',
            device=device,
            verbose=verbose,
            logger = logger
        )

    def on_epoch_start(self):
        self.model.eval()

    def batch_update(self, x):
        with torch.no_grad():
            prediction, ground_truth = self.model.forward(x)
            loss = self.loss(prediction, ground_truth)
        return loss, prediction, ground_truth

# Code from Prof Alex Binder
def get_data():
    category_lines = {}
    all_categories = ['st']
    category_lines['st']=[]
    filterwords=['NEXTEPISODE']
    with open('./star_trek_transcripts_all_episodes_f.csv', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        for row in reader:
            for el in row:
                if (el not in filterwords) and (len(el)>1):
                    v = el.strip().replace(';','').replace('\"','')
                    category_lines['st'].append(v)
    return category_lines,all_categories

# One-hot matrix of first to last letters (not including EOS) for input
def inputTensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li in range(len(line)):
        letter = line[li]
        tensor[li][0][all_letters.find(letter)] = 1
    return tensor.squeeze(1)

# LongTensor of second letter to end (EOS) for target
def targetTensor(line):
    letter_indexes = [all_letters.find(line[li]) for li in range(1, len(line))]
    letter_indexes.append(n_letters - 1) # EOS
    return torch.LongTensor(letter_indexes)

# def timeSince(since):
#     now = time.time()
#     s = now - since
#     m = math.floor(s / 60)
#     s -= m * 60
#     return '%dm %ds' % (m, s)

# Sample from a category and starting letter
# def sample(rnn, max_length = 200):
#     rnn.cpu()
#     caps_letters = 'ABCDEFGHIJKLMNOPRSTUVWZ'
#     start_letter = caps_letters[np.random.choice(len(caps_letters))]
#     with torch.no_grad():  # no need to track history in sampling
#         input = inputTensor(start_letter)
#         hidden = rnn.init_Hidden_n_Cell_sample()

#         output_name = start_letter

#         for _ in range(max_length):
#             output, hidden = rnn(input, hidden, sample  = True)
#             topi = np.random.choice(a = output.shape[1], size = output.shape[0], p = output.squeeze().numpy())[0]
#             if topi == n_letters - 1:
#                 break
#             else:
#                 letter = all_letters[topi]
#                 output_name += letter
#             input = inputTensor(letter)
#         return output_name

def plot(train_losses, val_losses, train_acc, val_acc):
    fig, ax = plt.subplots(1,2, figsize = (10,5))
    #fig.suptitle(f"Learning Rate: {optimizer.state_dict()['param_groups'][0]['lr']}, Max Epochs: {epochs} Batch size: {batch_size}, Hidden size: {hidden_size} n_layers: {n_layers} Model: {choice_of_model}")

    ax[0].set_title('Loss Value')
    ax[0].plot(train_losses, color = 'skyblue', label="Training Loss")
    ax[0].plot(val_losses, color = 'orange', label = "Validation Loss")
    ax[0].legend()

    ax[1].set_title('Measure Value')
    ax[1].plot(train_acc, color = 'skyblue', label="Training Measure")
    ax[1].plot(val_acc, color = 'orange', label="Validation Measure")
    ax[1].legend()
    cwd = os.getcwd()

    if not os.path.exists(os.path.join(cwd,'plots')):
        os.makedirs(os.path.join(cwd,'plots'))
    plt.savefig(os.path.join(cwd,'plots','nn_training' + str(time.ctime()).replace(':','').replace('  ',' ').replace(' ','_') + ".png"))
    plt.close()

def log_print(text, logger, log_only = False):
    if not log_only:
        print(text)
    if logger is not None:
        logger.info(text)