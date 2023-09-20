import os, sys, math
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

'''
A collection of utility functions for genomic data preprocessing, nested CV train-test split functions with random sampling, and plotting functions.
'''


def check_path(path, dir = False):
    '''
    Check if a path to a file or directory is valid.
    '''
    if dir == False:
        check = os.path.isfile(path)
    else:
        check = os.path.isdir(path)
    if check == True:
        # print('\n::: Found {}'.format(path))
        pass
    else:
        print('\n::: {} does not exist'.format(path))
        sys.exit(2)


def get_promoter(first_exon):
    '''
    Get the location of promoter given the first exon.
    '''
    chrom, start, end, strand = first_exon
    exon_start = int(start)
    exon_end = int(end)
    if strand == '+':
        promoter_end = exon_start - 1
        promoter_start = promoter_end - 500
        promoter = [chrom, promoter_start, promoter_end, strand]
    elif strand == '-':
        promoter_start = exon_end + 1
        promoter_end = promoter_start + 500
        promoter = [chrom, promoter_start, promoter_end, strand]
    else:
        promoter = None
    return promoter


def check_if_upstream(exon1, exon2):
    '''
    Check if exon 2 is upstream of exon 1.
    '''
    _, start1, end1, strand1 = exon1
    _, start2, end2, strand2 = exon2
    if strand1 != strand2:
        upstream = None
    else:
        if strand1 == '+':
            upstream = start2 < start1
        elif strand1 == '-':
            upstream = end2 > end1
        else:
            upstream = None
    return upstream


def get_reverse_complement(seq):
    '''
    Get the reverse complement of a DNA sequence.
    '''
    complement = { 'A': 'T', 'G': 'C', 'T': 'A', 'C': 'G', 'N': 'N' }
    rev_comp = ''
    l = len(seq)
    for n in range(l):
        rev_comp = rev_comp + complement[seq[l - n - 1]]
    return rev_comp


def make_agct_one_hot(seq):
    '''
    Generate a one-hot numpy array of a DNA sequence.

    Parameters
    ----------
    seq : str
        the DNA sequence

    Returns
    -------
    one_hot : numpy array
        the one-hot array with rows corresponding to A, G, C, T
    '''
    l = len(seq)
    one_hot = np.zeros((4, l), dtype = int)
    base_idx = { 'A': 0, 'G': 1, 'C': 2, 'T': 3 }
    for i in range(l):
        base = seq[i]
        one_hot[base_idx[base], i] = 1
    return one_hot


def make_agct_numeric_vector(seq):
    '''
    Generate a numpy array of a DNA sequence. Key: { 'A': 0, 'G': 1, 'C': 2, 'T': 3 }

    Parameters
    ----------
    seq : str
        the DNA sequence

    Returns
    -------
    vec : numpy array
        numeric sequence representing the nucleotide sequence
    '''
    l = len(seq)
    vec = np.zeros(l, dtype = int)
    base_idx = { 'A': 0, 'G': 1, 'C': 2, 'T': 3 }
    for i in range(l):
        base = seq[i]
        vec[i] = base_idx[base]
    return vec


def ignorant_kf_split(n_splits, ordered_cell_idx, ordered_gene_idx, subset_size = None, seed = 1):
    '''
    Perform an ignorant split on cell and gene indices.
    '''
    kf = KFold(n_splits = n_splits, shuffle = True, random_state = 1)
    train_indices = []
    test_indices = []
    if subset_size is not None:
        test_size = math.floor(subset_size / n_splits)
        train_size = subset_size - test_size
        rnd = np.random.RandomState(seed)
    x, y = np.meshgrid(ordered_cell_idx, ordered_gene_idx)
    indices = np.concatenate([x.reshape((x.size, 1)), y.reshape(y.size, 1)], axis = 1)
    for i, (train_idx, test_idx) in enumerate(kf.split(indices)):
        xvy = indices[train_idx]
        if subset_size is not None:
            xvy = xvy[rnd.choice(xvy.shape[0], train_size, replace = False), :]
        train_indices.append(xvy)
        xvy = indices[test_idx]
        if subset_size is not None:
            xvy = xvy[rnd.choice(xvy.shape[0], test_size, replace = False), :]
        test_indices.append(xvy)
    return train_indices, test_indices


def cellwise_kf_split(n_splits, ordered_cell_idx, ordered_gene_idx, subset_size = None, seed = 1):
    '''
    Perform a cellwise split on cell and gene indices.
    '''
    kf = KFold(n_splits = n_splits, shuffle = True, random_state = 1)
    train_indices = []
    test_indices = []
    if subset_size is not None:
        test_size = math.floor(subset_size / n_splits)
        train_size = subset_size - test_size
        rnd = np.random.RandomState(seed)
    for i, (train_idx, test_idx) in enumerate(kf.split(ordered_cell_idx)):
        # train
        ordered_cell_idx_train = ordered_cell_idx[train_idx]
        x, y = np.meshgrid(ordered_cell_idx_train, ordered_gene_idx)
        xvy = np.concatenate([x.reshape((x.size, 1)), y.reshape(y.size, 1)], axis = 1)
        if subset_size is not None:
            xvy = xvy[rnd.choice(xvy.shape[0], train_size, replace = False), :]
        train_indices.append(xvy)
        # test
        ordered_cell_idx_test = ordered_cell_idx[test_idx]
        x, y = np.meshgrid(ordered_cell_idx_test, ordered_gene_idx)
        if subset_size is not None:
            xvy = xvy[rnd.choice(xvy.shape[0], test_size, replace = False), :]
        test_indices.append(xvy)
    return train_indices, test_indices


def genewise_kf_split(n_splits, ordered_cell_idx, ordered_gene_idx, subset_size = None, seed = 1):
    '''
    Perform a genewise split on cell and gene indices.
    '''
    kf = KFold(n_splits = n_splits, shuffle = True, random_state = 1)
    train_indices = []
    test_indices = []
    if subset_size is not None:
        test_size = math.floor(subset_size / n_splits)
        train_size = subset_size - test_size
        rnd = np.random.RandomState(seed)
    for i, (train_idx, test_idx) in enumerate(kf.split(ordered_gene_idx)):
        # train
        ordered_gene_idx_train = ordered_gene_idx[train_idx]
        x, y = np.meshgrid(ordered_cell_idx, ordered_gene_idx_train)
        xvy = np.concatenate([x.reshape((x.size, 1)), y.reshape(y.size, 1)], axis = 1)
        if subset_size is not None:
            xvy = xvy[rnd.choice(xvy.shape[0], train_size, replace = False), :]
        train_indices.append(xvy)
        # test
        ordered_gene_idx_test = ordered_gene_idx[test_idx]
        x, y = np.meshgrid(ordered_cell_idx, ordered_gene_idx_test)
        xvy = np.concatenate([x.reshape((x.size, 1)), y.reshape(y.size, 1)], axis = 1)
        if subset_size is not None:
            xvy = xvy[rnd.choice(xvy.shape[0], test_size, replace = False), :]
        test_indices.append(xvy)
    return train_indices, test_indices


def strict_kf_split(n_splits, ordered_cell_idx, ordered_gene_idx, subset_size = None, seed = 1):
    '''
    Perform a strict split on cell and gene indices.
    '''
    kf = KFold(n_splits = n_splits, shuffle = True, random_state = 1)
    gene_splits = list(kf.split(ordered_gene_idx))
    cell_splits = list(kf.split(ordered_cell_idx))
    train_indices = []
    test_indices = []
    if subset_size is not None:
        test_size = math.floor(subset_size / n_splits)
        train_size = subset_size - test_size
        rnd = np.random.RandomState(seed)
    for i in range(n_splits):
        # get genes
        train_idx, test_idx = gene_splits[i]
        ordered_gene_idx_train = ordered_gene_idx[train_idx]
        ordered_gene_idx_test = ordered_gene_idx[test_idx]
        # get cells
        train_idx, test_idx = cell_splits[i]
        ordered_cell_idx_train = ordered_cell_idx[train_idx]
        ordered_cell_idx_test = ordered_cell_idx[test_idx]
        # get combined indices for training
        x, y = np.meshgrid(ordered_cell_idx_train, ordered_gene_idx_train)
        xvy = np.concatenate([x.reshape((x.size, 1)), y.reshape(y.size, 1)], axis = 1)
        if subset_size is not None:
            xvy = xvy[rnd.choice(xvy.shape[0], train_size, replace = False), :]
        train_indices.append(xvy)
        # same for test
        x, y = np.meshgrid(ordered_cell_idx_test, ordered_gene_idx_test)
        xvy = np.concatenate([x.reshape((x.size, 1)), y.reshape(y.size, 1)], axis = 1)
        if subset_size is not None:
            xvy = xvy[rnd.choice(xvy.shape[0], test_size, replace = False), :]
        test_indices.append(xvy)
    return train_indices, test_indices


def calculate_metrics(y_hat, y):
    fpr, tpr, _ = roc_curve(y, y_hat, pos_label = 1)
    auroc = auc(fpr, tpr)
    precision, recall, _ = precision_recall_curve(y, y_hat, pos_label = 1)
    aupr = average_precision_score(y, y_hat, pos_label = 1)
    base_aupr = y.sum().item() / y.shape[0]
    return { 'fpr': fpr, 'tpr': tpr, 'auroc': auroc, 'precision': precision, 'recall': recall, 'aupr': aupr, 'base_aupr': base_aupr }


def plot_loss(train_epochs, val_epochs, train_loss, val_loss, ax):
    ax.plot(train_epochs, train_loss, color = 'b', label = 'training')
    ax.plot(val_epochs, val_loss, color = 'r', label = 'validation')
    ax.legend(frameon = False, fontsize = 8)
    ax.set_xlabel('epochs')
    ax.set_ylabel('loss')
    ax.set_title('loss over epochs')


def plot_training_loss_components(train_epochs, train_loss, ax, components):
    colors = ['#03045e', '#00b4d8', '#0077b6']
    linestyles = [':', '-.', '--']
    for i in range(len(components)):
        ax.plot(train_epochs, train_loss[i], color = colors[i], linestyle = linestyles[i], label = components[i])
    ax.legend(frameon = False, fontsize = 8)
    ax.set_xlabel('epochs')
    ax.set_ylabel('loss')
    ax.set_title('training loss components')


def plot_roc_curve_single(fpr, tpr, auroc, ax):
    ax.plot((0, 1), (0, 1), color = 'k', linestyle = '--', linewidth = 1)
    ax.plot(fpr, tpr, color = 'k', label = 'AUROC = {0:.3f}'.format(auroc))
    ax.set_xlim(xmin = 0, xmax = 1)
    ax.set_ylim(ymin = 0, ymax = 1)
    ax.legend(frameon = False, loc = 'lower right', fontsize = 8)
    ax.set_xlabel('false positive rate')
    ax.set_ylabel('true positive rate')
    ax.set_title('ROC curve')


def plot_roc_curve_multiple(fprs, tprs, aurocs, ax, colors):
    ax.plot((0, 1), (0, 1), color = 'k', linestyle = '--', linewidth = 1)
    for i in range(len(tprs)):
        ax.plot(fprs[i], tprs[i], color = colors[i], alpha = 0.5, label = 'AUROC = {0:.3f}'.format(aurocs[i]))
    ax.set_xlim(xmin = 0, xmax = 1)
    ax.set_ylim(ymin = 0, ymax = 1)
    ax.legend(frameon = False, loc = 'lower right', fontsize = 8)
    ax.set_xlabel('false positive rate')
    ax.set_ylabel('true positive rate')
    ax.set_title('ROC curve')


def plot_pr_curve_single(precision, recall, aupr, base_aupr, ax):
    ax.plot((0, 1), (base_aupr, base_aupr), color = 'k', linestyle = '--', linewidth = 1)
    ax.plot(recall, precision, color = 'k', label = 'AUPR = {0:.3f}'.format(aupr))
    ax.set_xlim(xmin = 0, xmax = 1)
    ax.set_ylim(ymin = 0, ymax = 1)
    ax.legend(frameon = False, loc = 'upper right', fontsize = 8)
    ax.set_xlabel('recall')
    ax.set_ylabel('precision')
    ax.set_title('PR curve')


def plot_pr_curve_multiple(precisions, recalls, auprs, base_aupr, ax, colors):
    ax.plot((0, 1), (base_aupr, base_aupr), color = 'k', linestyle = '--', linewidth = 1)
    for i in range(len(precisions)):
        ax.plot(recalls[i], precisions[i], color = colors[i], alpha = 0.5, label = 'AUPR = {0:.3f}'.format(auprs[i]))
    ax.set_xlim(xmin = 0, xmax = 1)
    ax.set_ylim(ymin = 0, ymax = 1)
    ax.legend(frameon = False, loc = 'upper right', fontsize = 8)
    ax.set_xlabel('recall')
    ax.set_ylabel('precision')
    ax.set_title('PR curve')


def plot_roc_curve_multiclass(fprs, tprs, aurocs, labels, colors, ax):
    ax.plot((0, 1), (0, 1), color = 'k', linestyle = '--', linewidth = 1)
    for i in range(len(labels)):
        ax.plot(fprs[i], tprs[i], color = colors[i], label = labels[i] + '\nAUROC = {0:.3f}'.format(aurocs[i]))
    ax.set_xlim(xmin = 0, xmax = 1)
    ax.set_ylim(ymin = 0, ymax = 1)
    ax.legend(frameon = False, fontsize = 8, loc = 'lower right')
    ax.set_xlabel('false positive rate')
    ax.set_ylabel('true positive rate')
    ax.set_title('ROC curve (one-vs-rest)')


def plot_roc_curve_cv(roc_splits_exp, aurocs, ax):
    tprs = []
    base_fpr = np.linspace(0, 1, 101)
    for split in range(len(roc_splits_exp)):
        fpr, tpr = roc_splits_exp[split]
        ax.plot(fpr, tpr, color = 'k', alpha = 0.2)
        tpr = np.interp(base_fpr, fpr, tpr)
        tpr[0] = 0.0
        tprs.append(tpr)
    tprs = np.array(tprs)
    mean_tprs = tprs.mean(axis = 0)
    ax.plot(base_fpr, mean_tprs, color = 'k', alpha = 1, label = 'mean AUROC = {0:.3f}'.format(np.mean(aurocs)))
    ax.plot((0, 1), (0, 1), color = 'k', linestyle = '--', linewidth = 1)
    ax.set_xlim(xmin = 0, xmax = 1)
    ax.set_ylim(ymin = 0, ymax = 1)
    ax.legend(frameon = False, loc = 'lower right', fontsize = 8)
    ax.set_xlabel('false positive rate')
    ax.set_ylabel('true positive rate')
    ax.set_title('ROC curve')


def plot_loss_cv(train_losses, val_losses, ax):
    for split in range(len(train_losses)):
        ax.plot(train_losses[split][0], train_losses[split][1], color = 'b', label = 'training', alpha = 0.4)
        ax.plot(val_losses[split][0], val_losses[split][1], color = 'r', label = 'validation', alpha = 0.4)
        if split == 0:
            ax.legend(frameon = False, fontsize = 8)
    ax.set_xlabel('epochs')
    ax.set_ylabel('loss')
    ax.set_title('loss over epochs')