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

all_letters = string.ascii_letters + "0123456789 .,:!?'[]()/+-="
n_letters = len(all_letters) + 1 # Plus EOS marker

def get_data():
    category_lines = {}
    all_categories = ['st']
    category_lines['st']=[]
    filterwords=['NEXTEPISODE']
    with open('./star_trek_transcripts_all_episodes.csv', newline='') as csvfile:
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
    return tensor

# LongTensor of second letter to end (EOS) for target
def targetTensor(line):
    letter_indexes = [all_letters.find(line[li]) for li in range(1, len(line))]
    letter_indexes.append(n_letters - 1) # EOS
    return torch.LongTensor(letter_indexes)

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

# Sample from a category and starting letter
def sample(rnn, max_length = 200):
    caps_letters = 'ABCDEFGHIJKLMNOPRSTUVWZ'
    start_letter = caps_letters[np.random.choice(len(caps_letters))]
    with torch.no_grad():  # no need to track history in sampling
        input = inputTensor(start_letter)
        hidden = rnn.init_Hidden_n_Cell()

        output_name = start_letter

        for _ in range(max_length):
            output, hidden = rnn(input, hidden, sample  = True)
            topi = np.random.choice(a = output.shape[1], size = output.shape[0], p = output.squeeze().numpy())[0]
            if topi == n_letters - 1:
                break
            else:
                letter = all_letters[topi]
                output_name += letter
            input = inputTensor(letter)
        return output_name

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