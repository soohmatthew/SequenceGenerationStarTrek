from utils import *
from model import *
import logging

def run(epochs, rnn, train_category_lines, valid_category_line, criterion, device, optimizer, scheduler, print_every, sample_every):
    start = time.time()
    all_losses = []
    all_acc = []
    all_val_losses = []
    all_val_acc = []
    
    for epoch in range(1, epochs + 1):
        print(f"Epoch: {epoch}")
        count = 0
        acc_per_epoch = []
        loss_per_epoch = []

        # Train on training set
        train_start = time.time()
        rnn.train()
        print(f"Training ({epoch}/{epochs})")
        for line in train_category_lines:
            count += 1
            input_line_tensor= inputTensor(line)
            target_line_tensor = targetTensor(line)
            loss, acc = train(input_line_tensor, target_line_tensor, criterion, device, optimizer, scheduler, rnn)
            acc_per_epoch.append(acc)
            loss_per_epoch.append(loss)
            if count % print_every == 0:
                print('%s (%d %d%%) %.4f' % (timeSince(start), count, count / len(train_category_lines) * 100, np.mean(loss_per_epoch)))
                print(f'Training Acc: {np.mean(acc_per_epoch)}')
            if count % sample_every == 0:
                print(sample(rnn))
                logging.info(f'Trainined on {count} examples:')
                for i in range(1,11):
                    logging.info(f'{i}. {sample(rnn)}')
        print(f'Training Time taken: {timeSince(train_start)}')
        
        val_start = time.time()
        # Evaluate on validation set
        print(f"Evaluating ({epoch}/{epochs})")
        val_acc_per_epoch, val_loss_per_epoch = evaluate(rnn, valid_category_line, device, criterion)
        print(f'Validation Time taken: {timeSince(val_start)}')
        print(f'Validation Accuracy: {val_acc_per_epoch}')
        print(f'Validation Loss: {val_loss_per_epoch}')

        logging.info(f'Epoch: {epoch}')
        logging.info(f'Training Acc: {np.mean(acc_per_epoch)}')
        logging.info(f'Training Loss: {np.mean(loss_per_epoch)}')
        logging.info(f'Validation Acc: {val_acc_per_epoch}')
        logging.info(f'Validation Loss: {val_loss_per_epoch}')

        all_acc.append(np.mean(acc_per_epoch))
        all_losses.append(np.mean(loss_per_epoch))
        all_val_losses.append(val_loss_per_epoch)
        all_val_acc.append(val_acc_per_epoch)

    return all_losses, all_acc, all_val_losses, all_val_acc

if __name__ == "__main__":
    # Build the category_lines dictionary, a list of lines per category
    category_lines,all_categories = get_data()
    starting_time = time.time()
    
    training = True
    plotting = True 

    train_percent = 0.8
    valid_percent = 0.1

    train_category_lines = category_lines['st'][:int(train_percent * len(category_lines['st']))]
    valid_category_lines = category_lines['st'][int(train_percent * len(category_lines['st'])): int((train_percent + valid_percent) * len(category_lines['st']))]
    test_category_lines = category_lines['st'][int((train_percent + valid_percent) * len(category_lines['st'])):]

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
    criterion = nn.NLLLoss()

    logging.info(f'No. of layers: {n_layers}')
    logging.info(f'Hidden size: {hidden_size}')

    rnn = LSTM(n_letters, hidden_size, n_letters, n_layers)
    epochs = 10
    print_every = 1000
    plot_every =1000
    sample_every = 2000
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    learning_rate = 0.0005
    optimizer = torch.optim.Adam(rnn.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [1,2,3,4,5,6,7,8,9], gamma=0.5, last_epoch=-1)

    # Run on training and validation set
    if training:
        all_losses, all_acc, all_val_losses, all_val_acc = run(epochs, rnn, train_category_lines, valid_category_lines, criterion, device, optimizer, scheduler, print_every, sample_every)

    # Save model   
    cwd = os.getcwd()

    if not os.path.exists(os.path.join(cwd,'model')):
        os.makedirs(os.path.join(cwd,'model'))

    torch.save(rnn.state_dict(), f"model/model_lstm_{str(time.ctime()).replace(':','').replace('  ',' ').replace(' ','_') }.pth")

    # Plot results
    if plotting:
        plot(all_losses, all_val_losses, all_acc, all_val_acc)

    logging.info(f'Total time taken: {timeSince(starting_time)}')