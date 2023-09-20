import numpy as np
import gc
import torch
import torch.nn as nn
from tqdm import tqdm
import models
from multiome import MultiOmicDataset
from training_logger import NestedCVLogger
from utilities import calculate_metrics


def generate_random_params(seed = 1):
    '''
    Generate a set of parameters for random search (hyperparameter tuning).
    '''
    rnd = np.random.RandomState(seed)
    optimizer = 'sgd' # rnd.choice(['sgd', 'adam'])
    lr = round(rnd.uniform(low = 0.0001, high = 0.01), 4)
    weight_decay = rnd.choice([1e-4, 1e-5, 0])
    dropout = round(rnd.uniform(low = 0.1, high = 0.5), 2)
    return optimizer, lr, weight_decay, dropout


def encode_seq(seq):
    return torch.nn.functional.one_hot(torch.tensor(seq), 4).T


def make_dataloaders(train_indices, val_indices, data_dir, batch_size = 512, num_workers = 1):
    '''
    Prepare direct dataloaders for MultiOmicDataset.
    '''
    data = MultiOmicDataset(data_dir)
    seqs, atac, rna = data.fetch_samples(train_indices)
    train_data = torch.utils.data.TensorDataset(seqs.float(), atac.float(), rna.float())
    del seqs, atac, rna
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size = batch_size, shuffle = True, num_workers = num_workers)
    del train_data
    seqs, atac, rna = data.fetch_samples(val_indices)
    val_data = torch.utils.data.TensorDataset(seqs.float(), atac.float(), rna.float())
    del seqs, atac, rna
    val_dataloader = torch.utils.data.DataLoader(val_data, batch_size = batch_size, shuffle = True, num_workers = num_workers)
    del val_data
    del data
    gc.collect()
    return train_dataloader, val_dataloader


def train_with_early_stop(model, train_dl, val_dl, logger, opt, lr, wd, max_epochs, val_interval, device):
    '''
    Train a model using given parameters, and use the validation set for early stopping.
    '''
    loss_fn = nn.BCEWithLogitsLoss()
    if opt == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr = lr, weight_decay = wd)
    elif opt == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr = lr, weight_decay = wd)
    logger.reset_best_epoch()
    logger.reset_loss_curves()
    # find the number of arguments required for the model
    n_args = model.get_arg_count()
    # start training loop
    print('::: Training...')
    for epoch in tqdm(range(max_epochs), total = max_epochs):
        epoch_loss = 0.0  # for one full epoch
        model.train()
        for seq_data, atac_data, rna_data in train_dl:
            seq_data, atac_data, rna_data = seq_data.to(device), atac_data.to(device), rna_data.to(device)
            optimizer.zero_grad()
            if n_args == 1:
                output = model(seq_data)
            elif n_args == 2:
                output = model(seq_data, atac_data)
            train_loss = loss_fn(output.flatten(), rna_data)
            train_loss.backward()
            optimizer.step()
            epoch_loss += train_loss.item()
        # log training loss for epoch
        logger.append_to_train_loss(epoch, epoch_loss / len(train_dl))
        # validation
        if (epoch + 1) % val_interval == 0:
            model.eval()
            with torch.no_grad():
                val_loss = 0.0
                true_exp = torch.empty(0).to(device)
                pred_exp_probs = torch.empty(0).to(device)
                for seq_data, atac_data, rna_data in val_dl:
                    seq_data, atac_data, rna_data = seq_data.to(device), atac_data.to(device), rna_data.to(device)
                    if n_args == 1:
                        output = model(seq_data)
                    elif n_args == 2:
                        output = model(seq_data, atac_data)
                    loss = loss_fn(output.flatten(), rna_data)
                    val_loss += loss.item()
                    true_exp = torch.cat((true_exp, rna_data), 0)
                    pred_exp_probs = torch.cat((pred_exp_probs, torch.sigmoid(output)), 0)
                current_val_loss = val_loss / len(val_dl)
                logger.append_to_val_loss(epoch, current_val_loss)
                # if the validation loss improved, save a copy of the model parameters
                best_val_loss = logger.get_best_epoch()[-1]
                if current_val_loss < best_val_loss:
                    metrics = calculate_metrics(pred_exp_probs.cpu(), true_exp.cpu())
                    val_auroc = metrics['auroc']
                    val_aupr = metrics['aupr']
                    logger.set_best_epoch(epoch, current_val_loss, val_auroc, val_aupr)
                    torch.save(model.state_dict(), logger.get_model_save_path())
    # save loss curves
    logger.save_loss_curve_data()
    logger.save_val_results()


def search_params_kf(model_name, nested_cv_start_coords, train_indices, val_indices, data_dir, logger, max_trials = 10, max_epochs = 12, val_interval = 4, device = torch.device("cuda:0")):
    '''
    Train and evaluate a model with random parameters using k-fold cross-validation.
    '''
    outer_fold, start_trial_no, start_inner_fold = nested_cv_start_coords
    logger.set_outer_fold(outer_fold)
    model_class = getattr(models, model_name)
    for i in range(start_trial_no, max_trials):
        opt, lr, wd, drop = generate_random_params(i * 2)
        logger.set_search_space(i, opt, lr, wd, drop)
        # iterate over inner folds
        inner_k = len(train_indices)
        for inner_fold in range(start_inner_fold, inner_k):
            logger.set_inner_fold(inner_fold)
            # make dataloaders using inner cv indices
            train_dl, val_dl = make_dataloaders(train_indices[inner_fold], val_indices[inner_fold], data_dir)
            # initialize model with random parameters
            model = model_class(drop).to(device)
            # train and evaluate model with random hyperparameters
            train_with_early_stop(model, train_dl, val_dl, logger, opt, lr, wd, max_epochs, val_interval, device)


def load_best_model(model_name, outer_fold, logger):
    '''
    Load model using the best parameters saved to file.
    '''
    outer_fold, trial, med_inner_fold = logger.get_best_params(outer_fold)
    model_class = getattr(models, model_name)
    model = model_class()
    model.load_state_dict(torch.load(logger.get_path_to_model(outer_fold, trial, med_inner_fold)))
    return model
    

def test_model(model, test_indices, data_dir, logger, device = torch.device("cuda:0"), batch_size = 512, num_workers = 1):
    '''
    Test a trained model.
    '''
    print('::: Testing...')
    data = MultiOmicDataset(data_dir)
    seqs, atac, rna = data.fetch_samples(test_indices)
    test_data = torch.utils.data.TensorDataset(seqs.float(), atac.float(), rna.float())
    del seqs, atac, rna
    test_dl = torch.utils.data.DataLoader(test_data, batch_size = batch_size, shuffle = True, num_workers = num_workers)
    del data, test_data
    gc.collect()
    model.to(device)
    model.eval()
    true_exp = torch.empty(0).to(device)
    pred_exp_probs = torch.empty(0).to(device)
    # find the number of arguments required for the model
    n_args = model.get_arg_count()
    with torch.no_grad():
        for seq_data, atac_data, rna_data in test_dl:
            seq_data, atac_data, rna_data = seq_data.to(device), atac_data.to(device), rna_data.to(device)
            if n_args == 1:
                output = model(seq_data)
            elif n_args == 2:
                output = model(seq_data, atac_data)
            true_exp = torch.cat((true_exp, rna_data), 0)
            pred_exp_probs = torch.cat((pred_exp_probs, torch.sigmoid(output)), 0)
        # log results
        metrics = calculate_metrics(pred_exp_probs.cpu(), true_exp.cpu())
        logger.save_test_metrics(metrics)
        