import torch
import torch.nn as nn
import numpy as np
import datetime as dt
from utils import *
from torch.nn.utils.rnn import pack_sequence, pad_packed_sequence

class StarTrekLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers, temperature = 0.5, device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
        super(StarTrekLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.temperature = temperature
        self.device = device

        # Model architecture
        self.lstm = nn.LSTM(input_size = self.input_size, 
                             hidden_size = self.hidden_size, 
                             num_layers = self.n_layers,
                             dropout = 0.1)
        self.fc = nn.Linear(self.hidden_size, self.output_size)
        self.softmax = nn.Softmax(dim = 1)

    def init_hidden(self, batch_size = 1):
        # Initialise hidden and cell states with specific batch_size
        # Do not call directly
        init_hidden_state = torch.zeros(self.n_layers, batch_size, self.hidden_size)
        init_cell_state = torch.zeros(self.n_layers, batch_size, self.hidden_size)

        init_hidden_state, init_cell_state = init_hidden_state.to(self.device), init_cell_state.to(self.device)

        return init_hidden_state, init_cell_state

    @staticmethod
    def character_embed(list_of_sentences):
        # Takes in a sentence, and converts it to the input X and the ground truth y

        # output: packed_input_tensor_batch: A packed sequence for the tensors in a batch. Each tensor in the batch has shape (sentence_length, #characters)
        # type: torch.nn.utils.rnn.PackedSequence

        # output: packed_target_tensor_batch: Tensor of shape (batch, max_sequence_length)
        # type: tensor

        input_tensor_batch = []
        target_tensor_batch = []
        for line in list_of_sentences:
            input_tensor_batch.append(inputTensor(line).squeeze(1))
            target_tensor_batch.append(targetTensor(line))

        packed_input_tensor_batch = pack_sequence(input_tensor_batch, enforce_sorted = False)
        padded_target_tensor_batch, sequence_lengths = pad_packed_sequence(pack_sequence(target_tensor_batch, enforce_sorted=False), padding_value=-1, batch_first=True)

        return packed_input_tensor_batch, sequence_lengths, padded_target_tensor_batch
    
    def sample_sentence(self, max_length = 200):
        # Generate sample sentence by feeding a capital letter into the LSTM continuously until the EOS is returned
        caps_letters = 'ABCDEFGHIJKLMNOPRSTUVWZ'
        start_letter = caps_letters[np.random.choice(len(caps_letters))]
        X = inputTensor(start_letter).unsqueeze(0).to(self.device)
        with torch.no_grad():  # no need to track history in sampling
            self.hidden = self.init_hidden(batch_size = 1) # Initialise hidden and cell states
            output_name = start_letter

            for _ in range(max_length):
                X = self._internal_forward(X, sample = True)
                topi = np.random.choice(a = X.shape[1], size = X.shape[0], p = X.squeeze().cpu().numpy())[0]
                if topi == n_letters - 1:
                    break
                else:
                    letter = all_letters[topi]
                    output_name += letter
                X = inputTensor(letter).unsqueeze(0).to(self.device)
            return output_name
    
    def forward(self, X):
        # Function to
        # param: X: list of sentences
        # type: list
        X, sequence_lengths, y = self.character_embed(X)
        X, y = X.to(self.device), y.to(self.device)
        self.batch_size = sequence_lengths.shape[0] # Get Batch Size
        self.hidden = self.init_hidden(batch_size = self.batch_size) # Initialise hidden and cell states

        X = self._internal_forward(X)

        y = y.view(-1)
        return X, y
    
    def _internal_forward(self, X, sample = False):
        X, self.hidden = self.lstm(X, self.hidden)
        self.lstm.flatten_parameters()

        if not sample:
            X, _ = pad_packed_sequence(X, padding_value=-1, batch_first=True)
            X = X.contiguous()

        X = X.view(-1, X.shape[2]) # Reshape to (batch_size * max_seq_length, hidden_size)

        X = self.fc(X) # Output should be shape (batch_size * max_seq_length, # classes)

        if sample:
            X /= self.temperature
            X = self.softmax(X)

        return X

def train_model(train_dataloader,
                validation_dataloader,
                model,
                loss,
                optimizer,
                scheduler = None,
                batch_size = 1,
                num_epochs = 10,
                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
                logger = None,
                verbose = True,
                model_save_path = os.path.join(os.getcwd(),'weights'),
                model_save_prefix = '',
                plots_save_path = os.path.join(os.getcwd(),'plots')
                ):

    if torch.cuda.is_available():
        log_print('Using GPU', logger)
    else:
        log_print('Using CPU', logger)
    
    # Define Epochs
    train_epoch = TrainEpoch(
        model = model,
        loss = loss, 
        optimizer = optimizer,
        device = device,
        verbose = verbose,
        logger = logger,
    )

    valid_epoch = ValidEpoch(
        model = model, 
        loss = loss, 
        device = device,
        verbose = verbose,
        logger = logger,
    )

    # Record for plotting
    losses = {'train':[],'val':[]}
    metric_values = {'train':{'Accuracy':[]},'val':{'Accuracy':[]}}

    # Run Epochs
    best_perfmeasure = 0
    best_epoch = -1
    start_time = dt.datetime.now()
    log_print('Training model...', logger)

    for epoch in range(num_epochs):
        log_print(f'\nEpoch: {epoch}', logger)

        train_logs = train_epoch.run(train_dataloader)
        losses['train'].append(train_logs['loss'])
        metric_values['train']['Accuracy'].append(train_logs['Accuracy'])

        valid_logs = valid_epoch.run(validation_dataloader)
        losses['val'].append(valid_logs['loss'])
        metric_values['val']['Accuracy'].append(valid_logs['Accuracy'])
        
        for _ in range(20):
            log_print(model.sample_sentence(), logger)

        if scheduler is not None:
            scheduler.step()
            log_print(f"Next Epoch Learning Rate: {optimizer.state_dict()['param_groups'][0]['lr']}", logger)

        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)

        if best_perfmeasure < valid_logs['Accuracy']: # Right now the metric to be chosen for best_perf_measure is always the first metric
            best_perfmeasure = valid_logs['Accuracy']
            best_epoch = epoch

            torch.save(model, os.path.join(model_save_path,model_save_prefix + 'best_model.pth'))
            log_print('Best Model Saved', logger)

        torch.save(model, os.path.join(model_save_path,model_save_prefix + 'current_model.pth'))
        log_print('Current Model Saved', logger)

    log_print(f'Best epoch: {best_epoch} Best Performance Measure: {best_perfmeasure:.5f}', logger)
    log_print(f'Time Taken to train: {dt.datetime.now()-start_time}', logger)

    # Implement plotting feature
    plot(losses['train'],losses['val'],metric_values['train']['Accuracy'],metric_values['val']['Accuracy'])
    log_print('Plot Saved', logger)