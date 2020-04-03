from utils import iteratefromdict, log_print, get_data
from model import StarTrekLSTM, train_model

import os
import torch
import logging
import string
import time

if __name__ == "__main__":
    # Build the category_lines dictionary, a list of lines per category
    category_lines,all_categories = get_data()
    starting_time = time.time()
    category_lines = category_lines['st']
    category_lines = [line for line in category_lines if len(line) < 400]
    
    training = True
    plotting = True 

    train_percent = 0.8
    valid_percent = 0.1
    batch_size = 6

    train_dataloader = iteratefromdict(category_lines, train = True, seed = 5, batch_size = batch_size)
    val_dataloader = iteratefromdict(category_lines, train = False, seed = 5, batch_size = batch_size)
    test_dataloader = iteratefromdict(category_lines, train= False, test = True, seed = 5, batch_size = batch_size)

    # Set up logging
    cwd = os.getcwd()
    if not os.path.exists(os.path.join(cwd,'logs')):
        os.makedirs(os.path.join(cwd,'logs'))
    logging.basicConfig(filename= "logs/logfile_rnn_" + str(time.ctime()).replace(':','').replace('  ',' ').replace(' ','_') + ".log",
                        format='%(message)s',
                        level=logging.INFO)

    # Set params
    all_letters = string.ascii_letters + "0123456789 .,:!?'[]()/+-="
    n_letters = len(all_letters) + 1 # Plus EOS marker
    n_layers = 3
    hidden_size = 300
    
    log_print(f'No. of layers: {n_layers}', logging)
    log_print(f'Hidden size: {hidden_size}', logging)

    stLSTM = StarTrekLSTM(input_size = n_letters, hidden_size = hidden_size, output_size = n_letters, n_layers = n_layers)
    # stLSTM = torch.load(os.path.join(os.getcwd(),'model','best_model.pth'))

    criterion = torch.nn.CrossEntropyLoss(ignore_index=-1) # Ignore the padding index -1
    learning_rate = 0.001
    optimizer = torch.optim.Adam(stLSTM.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 10, T_mult=1, eta_min=0)
    loss = torch.nn.CrossEntropyLoss(ignore_index=-1) # Ignore the padded class

    train_model(train_dataloader = train_dataloader,
                validation_dataloader = val_dataloader,
                model = stLSTM,
                loss = loss,
                optimizer = optimizer,
                scheduler = scheduler,
                batch_size = batch_size,
                num_epochs = 20,
                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
                logger = logging,
                verbose = True,
                model_save_path = os.path.join(os.getcwd(),'model'),
                plots_save_path = os.path.join(os.getcwd(),'plots')
                )


