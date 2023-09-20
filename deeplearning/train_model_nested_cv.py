import os, sys, getopt
import gc
import numpy as np
import torch
import torch.nn as nn
import utilities, model_training
from multiome import MultiOmicDataset
from training_logger import NestedCVLogger

'''
USAGE NOTES

Prerequisite
------------
make_cv_splits.py

Required Inputs
---------------
    -m (model name)
        the name of the model to train
    -i (input directory)
        the full path to the directory containing RNA, ATAC, and sequence data
    -j (subdirectory)
        name of the directory within the input directory containing train and test sets for 5-fold CV
    -o (output directory)
        the directory where the output should be written (will make this directory if it does not exist yet)
    -g (gpu id)
        a number from 0 to 3 (nvidia gpu)
    -r (output subdirectory prefix)
        if resuming a partially complete run, specify the output subdirectory prefix

Example
-------
nohup /bin/bash -c '{ cd ~/rimodl; /mnt/data/orsi/anaconda3/bin/python train_model_nested_cv.py -m CNNResPlus -i /mnt/data/orsi/datasets -j strict_0-1-2-3 -o /mnt/data/orsi/rimodl_output/strict_0-1-2-3 -g 4; }' &


'''


def main(argv):
    try:
        opts, _ = getopt.getopt(argv, 'm:i:j:o:g:r:')
    except getopt.GetoptError:
        print('\n::: Error: cannot parse command line inputs')
        sys.exit(2) 
    for opt, arg in opts:
        if opt == '-m':
            global model_name
            model_name = arg
        elif opt == '-i':
            global input_dir
            input_dir = arg
        elif opt == '-j':
            global subdir
            subdir = arg
        elif opt == '-o':
            global output_dir
            output_dir = arg
        elif opt == '-g':
            global gpu_id
            gpu_id = arg
        elif opt == '-r':
            global resume
            resume = True
            global resume_prefix
            resume_prefix = arg




if __name__ == "__main__":

    resume = False
    start_outer_fold = 0
    start_trial_no = 0
    start_inner_fold = 0

    main(sys.argv[1:])
    inner_folds = 3
    n_trials = 5
    loss_fn = nn.BCEWithLogitsLoss()
    device = torch.device('cuda:{}'.format(gpu_id))

    # check if model name is valid
    if model_name not in ['CNNRes', 'CNNResPlus']:
        print('\n::: Cannot run model: {}'.format(model_name))
        sys.exit(2)

    # check if input files and directories exist
    utilities.check_path(input_dir, dir = True)
    input_subdir = input_dir + '/' + subdir
    utilities.check_path(input_subdir, dir = True)
    split_type = subdir.split('_')[0]
    utilities.check_path(output_dir, dir = True)

    # get all training sets
    train_files = sorted([ i for i in os.listdir(input_subdir) if 'train' in i ])

    # initialize logger and set the start folds/trials
    outer_k = len(train_files)
    if resume == True:
        logger = NestedCVLogger(model_name, output_dir, subdir, outer_k, inner_folds, resume = True, resume_prefix = resume_prefix)
        
        # get the next inner fold
        prev_i = logger.get_inner_fold()
        prev_t = logger.get_trial_no()
        prev_o = logger.get_outer_fold()
        if prev_i == inner_folds - 1:
            start_inner_fold = 0
            # get the next trial
            if prev_t == n_trials - 1:
                start_trial_no = 0
                # get the next outer fold
                if prev_o == outer_k - 1:
                    print('Error: cannot resume if already finished')
                    sys.exit(2)
                else:
                    start_outer_fold = prev_o + 1
            else:
                start_trial_no = prev_t + 1
                start_outer_fold = prev_o
        else:
            start_inner_fold = prev_i + 1
            start_trial_no = prev_t
            start_outer_fold = prev_o

    else:
        logger = NestedCVLogger(model_name, output_dir, subdir, outer_k, inner_folds)
        logger.record_input_dir(input_subdir)
    

    # outer cv
    for outer_fold in range(start_outer_fold, outer_k):

        # load train and val indices
        indices = np.load(input_subdir + '/train_fold_{}.npy'.format(outer_fold))
        cell_indices = np.unique(indices[:, 0])
        gene_indices = np.unique(indices[:, 1])

        # use the split function that corresponds to the train-test split 
        split_fn = getattr(utilities, split_type + '_kf_split')
        train_indices, val_indices = split_fn(inner_folds, cell_indices, gene_indices, subset_size = 120000, seed = outer_fold + 1)
        print('::: Each training fold has {} samples'.format(len(train_indices[0])))
        del cell_indices, gene_indices
        gc.collect()

        # get the "nested cv start coordinates"
        if outer_fold == start_outer_fold:
            nested_cv_start_coords = (start_outer_fold, start_trial_no, start_inner_fold)
        else:
            nested_cv_start_coords = (outer_fold, 0, 0)

        # perform inner CV for hyperparameter tuning with random search
        model_training.search_params_kf(model_name, nested_cv_start_coords, train_indices, val_indices, input_dir, logger = logger, max_trials = n_trials, max_epochs = 500, val_interval = 4, device = device)
        
        # load test indices and take random subset
        indices = np.load(input_subdir + '/test_fold_{}.npy'.format(outer_fold))
        rnd = np.random.RandomState(3 * outer_fold)
        indices = indices[rnd.choice(indices.shape[0], 60000, replace = False), :]

        # test the selected model
        model = model_training.load_best_model(model_name, outer_fold, logger)
        model_training.test_model(model, indices, input_dir, logger)
        del model
        gc.collect()

    # finish
    logger.signal_completion()