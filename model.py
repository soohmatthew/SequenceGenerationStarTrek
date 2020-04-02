import torch
import torch.nn as nn
import numpy as np
from utils import *

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.batch_size = 1
        self.temperature = 0.5

        # Model architecture
        self.LSTM1 = nn.LSTM(input_size = self.input_size, 
                             hidden_size = self.hidden_size, 
                             num_layers = self.n_layers, 
                             batch_first = True, 
                             dropout = 0.1)
        self.fc = nn.Linear(self.hidden_size, self.output_size)
        self.logsoftmax = nn.LogSoftmax(dim = 1)
        self.softmax = nn.Softmax(dim = 1)
        
    def forward(self, input, hidden_n_cell, sample = False):
        output, hidden_n_cell = self.LSTM1(input, hidden_n_cell)
        hidden_state, cell_state = hidden_n_cell
        output = self.fc(hidden_state[-1])
        if sample:
            output /= self.temperature
            return self.softmax(output), hidden_n_cell

        output = self.logsoftmax(output)
        return output, hidden_n_cell

    def init_Hidden_n_Cell(self):
        hidden_state = torch.zeros(self.n_layers, self.batch_size, self.hidden_size)
        cell_state = torch.zeros(self.n_layers, self.batch_size, self.hidden_size)
        return (hidden_state, cell_state)

def train(input_line_tensor, target_line_tensor, criterion, device, optimizer, scheduler, rnn):
    hidden = rnn.init_Hidden_n_Cell()
    rnn.zero_grad()
    loss = 0
    running_correct = 0
    input_line_tensor.to(device)
    for i in range(input_line_tensor.size(0)):
        input_batched = input_line_tensor[i].unsqueeze(0)
        output, hidden = rnn(input_batched, hidden)
        target = target_line_tensor.long()[i].unsqueeze(0)
        _, indices = torch.max(output.cpu(), 1)
        if indices.item() == target.item():
            running_correct += 1
        l = criterion(output.cpu(), target)
        loss += l
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()
    return loss.item() / input_line_tensor.size(0), running_correct / input_line_tensor.size(0)

def evaluate(rnn, valid_category_line, device, criterion):
    rnn.eval()
    val_acc_per_epoch = []
    val_loss_per_epoch = []
    for val_line in valid_category_line:
        val_hidden = rnn.init_Hidden_n_Cell()
        val_running_loss = 0
        val_running_correct = 0
        val_input_line_tensor= inputTensor(val_line)
        val_target_line_tensor = targetTensor(val_line)
        val_input_line_tensor.to(device)
        for j in range(val_input_line_tensor.size(0)):
            val_input_batched = val_input_line_tensor[j].unsqueeze(0)
            val_output, val_hidden = rnn(val_input_batched, val_hidden)
            val_target = val_target_line_tensor.long()[j].unsqueeze(0)
            _, val_indices = torch.max(val_output.cpu(), 1)
            if val_indices.item() == val_target.item():
                val_running_correct += 1
            val_l = criterion(val_output.cpu(), val_target)
            val_running_loss += val_l
        val_loss_per_epoch.append(val_running_loss.item() / val_input_line_tensor.size(0))
        val_acc_per_epoch.append(val_running_correct / val_input_line_tensor.size(0))

    return np.mean(val_acc_per_epoch), np.mean(val_loss_per_epoch)