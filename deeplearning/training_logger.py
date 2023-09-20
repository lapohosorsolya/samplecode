import os, sys
import pickle
import numpy as np
import pandas as pd
from datetime import datetime


class NestedCVLogger():
    '''
    Class to log progress and results of model training.
    '''
    def __init__(self, model_name, output_dir, input_dir, outer_k, inner_k, resume = False, resume_prefix = None):

        # private string attributes
        self.__model_name = model_name
        self.__output_dir = output_dir
        self.__train_data = input_dir
        self.__resuming = resume

        # private CV attributes for tracking; must have an odd number of folds
        if outer_k % 2 == 0 or inner_k % 2 == 0:
            print('Error: the number of fold should be an odd integer')
            sys.exit(2)
        else:
            self.__outer_k = outer_k
            self.__inner_k = inner_k

        # check if a previous run is being resumed
        if resume == True:
            if resume_prefix is not None:
                # use previous directories
                self.__save_str = self.__output_dir + '/' + resume_prefix
            else:
                print('Error: need a directory prefix to resume run')
                sys.exit(2)
        else:
            # initialize new directories with the current time
            self.__date_time = datetime.now()
            self.__save_str = self.__output_dir + '/' + self.__date_time.strftime('%y%m%d-%H%M%S') + '_' + self.__model_name + '_' + self.__train_data.split('/')[-1]

        # directories
        self.__logging_dir = self.__save_str + '_logs'
        self.__loss_data_dir = self.__save_str + '_loss'
        self.__metrics_data_dir = self.__save_str + '_metrics'
        self.__models_dir = self.__save_str + '_models'

        # log files
        self.__log_file = self.__logging_dir + '/.log'
        self.__search_spaces_file = self.__logging_dir + '/searchspaces.txt'
        self.__val_results_file = self.__logging_dir + '/valresults.txt'
        self.__best_params_file = self.__logging_dir + '/bestparams.txt'
        self.__test_results_file = self.__logging_dir + '/testresults.txt'

        self.__current_outer_fold = 0
        self.__current_inner_fold = 0
        self.__current_trial_no = 0
        
        self.__current_opt = None
        self.__current_lr = None
        self.__current_wd = None
        self.__current_drop = None
        self.__best_epoch = 0
        self.__best_val_loss = 100.0
        self.__best_val_auroc = 0
        self.__best_val_aupr = 0

        self.__val_loss_epochs = []
        self.__train_loss_epochs = []

        # make new directories and files if needed
        if resume == True:
            # read the previous output files to determine the current folds
            val_df = pd.read_csv(self.__val_results_file, sep = '\t')
            last_row = val_df.iloc[-1, :]
            self.__current_outer_fold = int(last_row.outer_fold)
            self.__current_inner_fold = int(last_row.inner_fold)
            self.__current_trial_no = int(last_row.trial)
            self.__signal_resumption()
        else:
            self.__make_files()


    def __make_files(self):
        # directory to collect loss data
        os.mkdir(self.__logging_dir)
        os.mkdir(self.__loss_data_dir)
        os.mkdir(self.__metrics_data_dir)
        os.mkdir(self.__models_dir)
        # start files that will be filled with data
        with open(self.__log_file, 'w') as f:
            f.write('\n'.join(['Log Start: ' + self.__date_time.strftime('%y-%m-%d %H:%M:%S'), 'Model Name: ' + self.__model_name, 'Training Data: ' + self.__train_data, '\n']))
        with open(self.__search_spaces_file, 'w') as f:
            f.write('\t'.join(['outer_fold', 'trial', 'optimizer', 'learning_rate', 'weight_decay', 'dropout']))
            f.write('\n')
        with open(self.__val_results_file, 'w') as f:
            f.write('\t'.join(['outer_fold', 'inner_fold', 'trial', 'val_loss', 'auroc', 'aupr', 'early_stop_epoch']))
            f.write('\n')
        with open(self.__best_params_file, 'w') as f:
            f.write('\t'.join(['outer_fold', 'trial', 'optimizer', 'learning_rate', 'weight_decay', 'dropout', 'early_stop_epoch', 'median_inner_fold']))
            f.write('\n')
        with open(self.__test_results_file, 'w') as f:
            f.write('\t'.join(['outer_fold', 'auroc', 'aupr']))
            f.write('\n')
        

    def record_input_dir(self, input_dir):
        '''
        Record the directory containing the indices of the training data.
        '''
        dt = datetime.now().strftime('%y-%m-%d %H:%M:%S')
        with open(self.__log_file, 'a') as f:
            f.write('\n{} -- TRAINING DATA INDICES:'.format(dt))
            f.write('\n\t{}'.format(input_dir))
            f.write('\n')


    def set_outer_fold(self, outer_fold):
        '''
        Update the current outer fold.
        '''
        if outer_fold >= 0 and outer_fold < self.__outer_k:
            self.__current_outer_fold = outer_fold
            self.__log_outer_fold()
        else:
            pass


    def __log_outer_fold(self):
        '''
        Log the current outer fold.
        '''
        dt = datetime.now().strftime('%y-%m-%d %H:%M:%S')
        with open(self.__log_file, 'a') as f:
            f.write('\n{} -- SET OUTER FOLD:'.format(dt))
            f.write('\n\t{}'.format(self.__current_outer_fold))
            f.write('\n')


    def set_inner_fold(self, inner_fold):
        '''
        Update the current inner fold.
        '''
        if inner_fold >= 0 and inner_fold < self.__inner_k:
            self.__current_inner_fold = inner_fold
            self.__log_inner_fold()
        else:
            pass


    def __log_inner_fold(self):
        '''
        Log the current inner fold.
        '''
        dt = datetime.now().strftime('%y-%m-%d %H:%M:%S')
        with open(self.__log_file, 'a') as f:
            f.write('\n{} -- SET INNER FOLD:'.format(dt))
            f.write('\n\t{}'.format(self.__current_inner_fold))
            f.write('\n')

    
    def get_outer_fold(self):
        '''
        Return the current outer fold.
        '''
        return self.__current_outer_fold
    

    def get_inner_fold(self):
        '''
        Return the current inner fold.
        '''
        return self.__current_inner_fold
    

    def get_trial_no(self):
        '''
        Return the current trial number.
        '''
        return self.__current_trial_no
    

    def set_search_space(self, trial_no, optimizer, learning_rate, weight_decay, dropout):
        '''
        Change the current set of parameters and call the logging function to record it.
        '''
        self.__current_trial_no = trial_no
        self.__current_opt = optimizer
        self.__current_lr = learning_rate
        self.__current_wd = weight_decay
        self.__current_drop = dropout
        self.__log_search_space()


    def __log_search_space(self):
        '''
        Log the current search space:
            - outer fold
            - trial
            - optimizer
            - learning rate
            - weight decay
            - dropout
        '''
        new_row = [self.__current_outer_fold, self.__current_trial_no, self.__current_opt, self.__current_lr, self.__current_wd, self.__current_drop]
        # add row to table
        with open(self.__search_spaces_file, 'a') as f:
            f.write('\t'.join([ str(i) for i in new_row ]))
            f.write('\n')
        # record in log file
        dt = datetime.now().strftime('%y-%m-%d %H:%M:%S')
        with open(self.__log_file, 'a') as f:
            f.write('\n{} -- SEARCH SPACE:'.format(dt))
            f.write('\n\tfold = {}, trial = {}, opt = {}, lr = {}, wd = {}, drop = {}'.format(*new_row))
            f.write('\n')


    def save_val_results(self):
        '''
        Log new validation results:
            - outer fold
            - inner fold
            - trial
            - training loss
            - validation loss
            - AUROC
            - AUPR
            - early stop epoch
        '''
        new_row = [self.__current_outer_fold, self.__current_inner_fold, self.__current_trial_no, self.__best_val_loss, self.__best_val_auroc, self.__best_val_aupr, self.__best_epoch]
        # add row to table
        with open(self.__val_results_file, 'a') as f:
            f.write('\t'.join([ str(i) for i in new_row ]))
            f.write('\n')
        # record in log file
        dt = datetime.now().strftime('%y-%m-%d %H:%M:%S')
        with open(self.__log_file, 'a') as f:
            f.write('\n{} -- VALIDATION RESULTS:'.format(dt))
            f.write('\n\tfold = {}.{}, trial = {}, val_loss = {}, auroc = {}, aupr = {}, early_stop_epoch = {}'.format(*new_row))
            f.write('\n')


    def reset_loss_curves(self):
        '''
        Reset the variables storing the training and validation loss curves.
        '''
        self.__train_loss_epochs = []
        self.__val_loss_epochs = []


    def append_to_train_loss(self, epoch, train_loss):
        '''
        Append a new point to the training loss curve.
        '''
        self.__train_loss_epochs.append([epoch, train_loss])


    def append_to_val_loss(self, epoch, val_loss):
        '''
        Append a new point to the validation loss curve.
        '''
        self.__val_loss_epochs.append([epoch, val_loss])


    def save_loss_curve_data(self):
        '''
        Write the training and validation loss curves to file.
        '''
        loss_data = {}
        loss_data['train'] = np.array(self.__train_loss_epochs).T
        loss_data['val'] = np.array(self.__val_loss_epochs).T
        np.savez(self.__loss_data_dir + '/{}.{}.{}.npz'.format(self.__current_outer_fold, self.__current_trial_no, self.__current_inner_fold), **loss_data)
        dt = datetime.now().strftime('%y-%m-%d %H:%M:%S')
        with open(self.__log_file, 'a') as f:
            f.write('\n{} -- SAVED LOSS CURVE DATA:'.format(dt))
            f.write('\n\touter_fold = {}, trial = {}, inner_fold = {}'.format(self.__current_outer_fold, self.__current_trial_no, self.__current_inner_fold))
            f.write('\n')


    def save_test_metrics(self, metrics):
        '''
        Write test metrics to file. Should use output of calculate_metrics() from the rimodl_functions module.
        '''
        with open(self.__metrics_data_dir + '/{}_metrics.pkl'.format(self.__current_outer_fold), 'wb') as f:
            pickle.dump(metrics, f)
        self.__log_test_results(metrics['auroc'], metrics['aupr'])


    def get_model_save_path(self):
        '''
        Return the full path with a unique filename corresponding to the current outer fold, parameter trial, and inner fold.
        '''
        return self.__models_dir + '/{}.{}.{}.pt'.format(self.__current_outer_fold, self.__current_trial_no, self.__current_inner_fold)


    def get_best_params(self, outer_fold):
        '''
        Get best parameters selected by random search in inner CV:
            - outer fold
            - trial
            - optimizer
            - learning rate
            - weight decay
            - dropout
            - early stop epoch (from inner fold with the median AUROC)
        '''
        # read data from validation results file
        df = pd.read_csv(self.__val_results_file, sep = '\t')
        best_trial = df[df.outer_fold == outer_fold].groupby(by = 'trial').mean().auroc.idxmax()
        subdf = df[df.outer_fold == outer_fold][df.trial == best_trial]
        # get the inner fold with the parameter set of the best trial and the median auroc
        median_auroc = subdf.auroc.median()
        median_row = subdf[subdf.auroc == median_auroc]
        median_inner_fold = median_row.inner_fold.values[0]
        median_early_stop = median_row.early_stop_epoch.values[0]
        # read data from search spaces file
        df = pd.read_csv(self.__search_spaces_file, sep = '\t')
        trial_data = list(df[df.outer_fold == outer_fold][df.trial == best_trial].values.flatten())
        trial_data.extend([median_early_stop, median_inner_fold])
        # log results
        self.__log_best_params(trial_data)
        return trial_data[0], trial_data[1], trial_data[7] # outer_fold, trial, median inner fold


    def __log_best_params(self, best_params):
        '''
        Log best parameters selected by random search in inner CV:
            - outer fold
            - trial
            - optimizer
            - learning rate
            - weight decay
            - dropout
            - early stop epoch
        '''
        # add row to table
        with open(self.__best_params_file, 'a') as f:
            f.write('\t'.join([ str(i) for i in best_params ]))
            f.write('\n')
        # record in log file
        dt = datetime.now().strftime('%y-%m-%d %H:%M:%S')
        with open(self.__log_file, 'a') as f:
            f.write('\n{} -- BEST PARAMETERS:'.format(dt))
            f.write('\n\tfold = {}, trial = {}, opt = {}, lr = {}, wd = {}, drop = {}, early_stop_epoch = {}'.format(*best_params))
            f.write('\n')


    def __log_test_results(self, auroc, aupr):
        '''
        Log model performance on the held-out test set:
            - AUROC
            - AUPR
        '''
        new_row = [self.__current_outer_fold, auroc, aupr]
        # add row to table
        with open(self.__test_results_file, 'a') as f:
            f.write('\t'.join([ str(i) for i in new_row ]))
            f.write('\n')
        # record in log file
        dt = datetime.now().strftime('%y-%m-%d %H:%M:%S')
        with open(self.__log_file, 'a') as f:
            f.write('\n{} -- TEST RESULTS:'.format(dt))
            f.write('\n\tfold = {}, auroc = {}, aupr = {}'.format(*new_row))
            f.write('\n')
    

    def set_best_epoch(self, epoch, val_loss, val_auroc, val_aupr):
        '''
        Update the best validation loss.
        '''
        self.__best_val_loss = val_loss
        self.__best_val_auroc = val_auroc
        self.__best_val_aupr = val_aupr
        self.__best_epoch = epoch


    def reset_best_epoch(self):
        '''
        Reset the best epoch (when starting a new round of training).
        '''
        self.__best_val_loss = 100.0
        self.__best_val_auroc = 0
        self.__best_val_aupr = 0
        self.__best_epoch = 0


    def get_best_epoch(self):
        '''
        Return the best epoch and its corresponding metrics (for early stopping).
        '''
        return self.__best_epoch, self.__best_val_auroc, self.__best_val_aupr, self.__best_val_loss


    def get_path_to_model(self, outer_fold, trial, inner_fold):
        '''
        Return the path to the model with the selected parameters.
        '''
        return self.__models_dir + '/{}.{}.{}.pt'.format(outer_fold, trial, inner_fold)
    
    
    def signal_completion(self):
        '''
        Record the finish time.
        '''
        current_dt = datetime.now()
        dt = current_dt.strftime('%y-%m-%d %H:%M:%S')
        if self.__resuming == False:
            runtime = current_dt - self.__date_time
            # record in log file
            with open(self.__log_file, 'a') as f:
                f.write('\n{} -- RUN COMPLETE'.format(dt))
                f.write('\n\ttotal runtime {}'.format(runtime))
                f.write('\n')
        else:
            with open(self.__log_file, 'a') as f:
                f.write('\n{} -- RUN COMPLETE'.format(dt))
                f.write('\n\tcannot calculate total runtime due to interruption')
                f.write('\n')


    def __signal_resumption(self):
        '''
        Record the resumption time and current fold information.
        '''
        current_dt = datetime.now()
        dt = current_dt.strftime('%y-%m-%d %H:%M:%S')
        # record in log file
        with open(self.__log_file, 'a') as f:
            f.write('\n--!--!--!-- INTERRUPTION --!--!--!--')
            f.write('\n------------------------------------\n')
            f.write('\n{} -- RESUMING RUN AFTER PREVIOUS:'.format(dt))
            f.write('\n\touter fold = {}, trial = {}, inner fold = {}'.format(self.__current_outer_fold, self.__current_trial_no, self.__current_inner_fold))
            f.write('\n')