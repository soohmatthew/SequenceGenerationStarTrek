from utils import *
from model import *
import torch
import logging

# def run(epochs, rnn, train_dataloader, validation_dataloader, criterion, device, optimizer, scheduler, print_every, sample_every):
#     start = time.time()
#     all_losses = []
#     all_acc = []
#     all_val_losses = []
#     all_val_acc = []
    
#     for epoch in range(1, epochs + 1):
#         print(f"Epoch: {epoch}")
#         count = 0
#         acc_per_epoch = []
#         loss_per_epoch = []

#         # Train on training set
#         train_start = time.time()
#         rnn.train()
#         print(f"Training ({epoch}/{epochs})")
#         for idx, batch_category_lines in enumerate(train_dataloader):
#             count += len(batch_category_lines)
#             input_tensor_batch = []
#             target_tensor_batch = []
#             for line in batch_category_lines:
#                 input_tensor_batch.append(inputTensor(line).squeeze(1))
#                 target_tensor_batch.append(targetTensor(line))

#             input_tensor_batch_padded = pad_packed_sequence(pack_sequence(input_tensor_batch, enforce_sorted=False))[0]
#             target_tensor_batch_padded = pad_packed_sequence(pack_sequence(target_tensor_batch, enforce_sorted=False))[0]

#             loss, acc = train(input_tensor_batch_padded, target_tensor_batch_padded, criterion, device, optimizer, scheduler, rnn)
#             acc_per_epoch.append(acc)
#             loss_per_epoch.append(loss)
#             if count % print_every == 0:
#                 print('%s (%d %d%%) %.4f' % (timeSince(start), count, count / train_dataloader.num() * 100, np.sum(loss_per_epoch)/count))
#                 # print(f'Training Acc: {np.sum(acc_per_epoch)/(count}')
#             if count % sample_every == 0:
#                 print(sample(rnn))
#                 logging.info(f'Trainined on {count} examples:')
#                 for i in range(1,11):
#                     logging.info(f'{i}. {sample(rnn)}')
#         print(f'Training Time taken: {timeSince(train_start)}')
        
#         val_start = time.time()
#         # Evaluate on validation set
#         print(f"Evaluating ({epoch}/{epochs})")
#         val_acc_per_epoch, val_loss_per_epoch = evaluate(rnn, validation_dataloader, device, criterion)
#         print(f'Validation Time taken: {timeSince(val_start)}')
#         print(f'Validation Accuracy: {val_acc_per_epoch}')
#         print(f'Validation Loss: {val_loss_per_epoch}')

#         logging.info(f'Epoch: {epoch}')
#         logging.info(f'Training Acc: {np.mean(acc_per_epoch)}')
#         logging.info(f'Training Loss: {np.mean(loss_per_epoch)}')
#         logging.info(f'Validation Acc: {val_acc_per_epoch}')
#         logging.info(f'Validation Loss: {val_loss_per_epoch}')

#         all_acc.append(np.mean(acc_per_epoch))
#         all_losses.append(np.mean(loss_per_epoch))
#         all_val_losses.append(val_loss_per_epoch)
#         all_val_acc.append(val_acc_per_epoch)

#     return all_losses, all_acc, all_val_losses, all_val_acc

if __name__ == "__main__":
    # Build the category_lines dictionary, a list of lines per category
    category_lines,all_categories = get_data()
    starting_time = time.time()
    category_lines = category_lines['st']
    
    training = True
    plotting = True 

    train_percent = 0.8
    valid_percent = 0.1
    batch_size = 16

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

    stLSTM = StarTrekLSTM(input_size = n_letters,
                        hidden_size = hidden_size,
                        output_size = n_letters,
                        n_layers = n_layers)
    # epochs = 1
    # print_every = 1000
    # plot_every = 1000
    # sample_every = 1000
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-1) # Ignore the padding index -1
    learning_rate = 0.005
    optimizer = torch.optim.Adam(stLSTM.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [3, 5, 7, 9], gamma=0.5)

    # # Run on training and validation set
    # if training:
    #     all_losses, all_acc, all_val_losses, all_val_acc = run(epochs, rnn, train_dataloader, val_dataloader, criterion, device, optimizer, scheduler, print_every, sample_every)

    # # Plot results
    # if plotting:
    #     plot(all_losses, all_val_losses, all_acc, all_val_acc)

    # logging.info(f'Total time taken: {timeSince(starting_time)}')

    # Define Loss and Accuracy Metric
    loss = torch.nn.CrossEntropyLoss(ignore_index=-1) # Ignore the padded class

    train_model(train_dataloader = train_dataloader,
                validation_dataloader = val_dataloader,
                model = stLSTM,
                loss = loss,
                optimizer = optimizer,
                scheduler = scheduler,
                batch_size = batch_size,
                num_epochs = 10,
                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
                logger = logging,
                verbose = True,
                model_save_path = os.path.join(os.getcwd(),'model'),
                plots_save_path = os.path.join(os.getcwd(),'plots')
                )


