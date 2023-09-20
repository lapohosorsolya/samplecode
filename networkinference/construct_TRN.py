import sys, os, getopt
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
from TRN import construction as cr
from TRN import comparison as cp
from multiprocessing import Pool
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


"""
Infers transcriptional regulatory networks via Elastic Net regression (target gene expression modeled as a function of TF gene expression).

USAGE NOTES

MODE 0: basic TRN reconstruction
--------------------------------
    + Required: m (mode), i (path to metadata file), g (path to target selection directory), f (path to TF selection directory), o (path to output directory)
    + If mode is 0, all other flags are ignored (j, t, n, p)
    + No optional arguments

MODE 1: intra-TRN shuffling
---------------------------
    + Required: m, i, g, f, o, t (file with list of target genes to shuffle)
    + Optional: j (number of jobs, defaults to 4), n (number of shuffles, defaults to 10,000)
    + If mode is 1, the p flag is ignored

MODE 2: inter-TRN shuffling
---------------------------
    + Required: m, i, g, f, o
    + Optional: j, n, p (whether permutation should be done without taking into account group size)
    + If mode is 2, the t flag is ignored

MODE 3: leave-one-out TRNs
--------------------------
    + Required: m, i, g, f, o
    + No optional arguments
    + If mode is 3, the j, t, n and p flags are ignored

"""

def main(argv):
    try:
        opts, args = getopt.getopt(argv, 'm:i:o:g:f:j:t:n:pl')
    except getopt.GetoptError:
        print('TRY AGAIN...')
        sys.exit(2) 
    for opt, arg in opts:
        if opt == '-m':
            global smode
            smode = int(arg)
        elif opt == '-i':
            global input_file
            input_file = arg
        elif opt == '-o':
            global out_dir
            out_dir = arg
        elif opt == '-g':
            global target_dir
            target_dir = arg
        elif opt == '-f':
            global tf_dir
            tf_dir = arg
        elif opt == '-j':
            global njobs
            njobs = int(arg)
        elif opt == '-t':
            global targetfile
            targetfile = arg
        elif opt == '-n':
            global n
            n = int(arg)
        elif opt == '-p':
            global nchoosek
            nchoosek = False
        elif opt == '-l':
            global logtr
            logtr = False
        

def scrambler(i, data, out_dir):
    """
    Scrambles phenotypic identity of each sample, without taking into account the size of the group.

    Parameters
    ----------
    i : int
        index of the iteration
    data : tuple of numpy.Array (X, y, s)
        the regression matrix, the target gene vector, and the scramble matrix
    out_dir : str
        directory for output

    Returns
    -------
    None
    """
    X, y, s = data
    # get indices of samples
    ind_0 = np.where(s[i] == 0)[0]
    ind_1 = np.where(s[i] == 1)[0]
    # split X and y using indices
    X_0 = X[ind_0, :]
    y_0 = y[ind_0, :]
    X_1 = X[ind_1, :]
    y_1 = y[ind_1, :]
    # construct Elastic Net TRN
    print('\t{}.0...'.format(i))
    trn_0 = cr.construct_TRN(X_0, y_0)
    print('\t{}.1...'.format(i))
    trn_1 = cr.construct_TRN(X_1, y_1)
    # subtract TRNs, save absolute values
    trn_diff = np.abs(trn_1 - trn_0)
    np.save(out_dir + '/trn{}.npy'.format(i), trn_diff)


def splitter(i, data, out_dir):
    """
    Splits samples randomly into two groups using permutations of `n` samples in a group.

    Parameters
    ----------
    i : int
        index of the iteration
    data : tuple of numpy.Array (X, y, n)
        the regression matrix, the target gene vector, and the number of samples in group 0
    out_dir : str
        directory for output

    Returns
    -------
    None
    """
    X, y, size = data

    # split X and y
    X_0, X_1, y_0, y_1 = train_test_split(X, y, test_size = size)

    # construct Elastic Net TRN
    print('\t{}.0...'.format(i))
    trn_0 = cr.construct_TRN(X_0, y_0)
    print('\t{}.1...'.format(i))
    trn_1 = cr.construct_TRN(X_1, y_1)

    # subtract TRNs, save absolute values
    trn_diff = np.abs(trn_1 - trn_0)
    np.save(out_dir + '/trn{}.npy'.format(i), trn_diff)


def target_organizer(ind, gene, data, out_dir):
    """
    Splits the target genes for intra-TRN shuffling.

    Parameters
    ----------
    ind : int
        index of the target gene
    gene : str
        target gene ID
    data : tuple of numpy.Array (X_0, y_0, X_1, y_1)
        the regression matrices and the target gene matrices for each phenotype
    out_dir : str
        directory for output

    Returns
    -------
    None
    """
    X_0, y_0, X_1, y_1 = data

    # Elastic Net with shuffling of target vector
    target_0 = cr.shuffled_target_EN(X_0, y_0[:, ind], n_scrambles = n)
    target_1 = cr.shuffled_target_EN(X_1, y_1[:, ind], n_scrambles = n)

    # save shuffle results
    np.save(out_dir + '/target_{}_{}.npy'.format(gene, groups[0]), target_0)
    np.save(out_dir + '/target_{}_{}.npy'.format(gene, groups[1]), target_1)


if __name__ == "__main__":

    # default values
    smode = 0           # default mode 0 (regular differential network analysis)
    logtr = True        # log-transform RPKM expression data (set to False for single-cell RNA-seq UMI counts)
    nchoosek = True     # default sample split using n choose k
    njobs = 4           # default parallel jobs = 4
    n = 10000           # default number of shuffles

    main(sys.argv[1:])

    # read files
    metadata = pd.read_csv(input_file, sep = '\t')
    exp_df = pd.read_csv(target_dir + '/expression_RPKM.csv', index_col = 0).T
    degs = np.loadtxt(target_dir + '/DEGs.txt', delimiter = '\n', dtype = str)
    tfs = np.loadtxt(tf_dir + '/SelectedTFs.txt', delimiter = '\n', dtype = str)

    # log-transform and scale
    if logtr == True: # set this to False for scRNA-seq data
        exp = np.log2(exp_df + 1).to_numpy()
    else:
        exp = exp_df.to_numpy()
    norm_exp = StandardScaler().fit(exp).transform(exp)

    # find indices of tf columns
    mask_tfs = [ (gene in tfs) for gene in exp_df.columns ]
    ind_tfs = np.where(mask_tfs)[0]

    # remove any tfs from degs (no self-edges)
    targets = [ gene for gene in degs if gene not in tfs ]
    print('Removed {} TFs from target genes (self-edges)'.format(len(degs) - len(targets)))

    # find indices of target columns
    mask_targets = [ (gene in targets) for gene in exp_df.columns ]
    ind_targets = np.where(mask_targets)[0]

    # print lists of TFs and targets to accompany output TRNs
    ordered_tfs = exp_df.columns[ind_tfs].to_list()
    ordered_targets = exp_df.columns[ind_targets].to_list()
    with open(out_dir + '/TRN_TFs.txt', mode='wt', encoding='utf-8') as f:
        f.write('\n'.join(ordered_tfs))
    with open(out_dir + '/TRN_targets.txt', mode='wt', encoding='utf-8') as f:
        f.write('\n'.join(ordered_targets))

    # subset normalized expression matrix
    X = norm_exp[:, ind_tfs]
    y = norm_exp[:, ind_targets]

    # split expression into two groups
    groups = sorted(list(set(metadata.group.to_list())))
    print('Found {} groups:\n{}'.format(len(groups), groups))
    if len(groups) != 2:
        print('Metadata file does not contain exactly 2 phenotypes in the "group" column!')
        if smode != 3:
            sys.exit(2)

    if smode == 0: # regular TRNs

        # get indices of samples
        ind_0 = np.where(metadata.group == groups[0])[0]
        ind_1 = np.where(metadata.group == groups[1])[0]

        # split X and y using indices
        X_0 = X[ind_0, :]
        y_0 = y[ind_0, :]
        X_1 = X[ind_1, :]
        y_1 = y[ind_1, :]

        # construct Elastic Net TRN
        trn_0, alphas_0, l1s_0 = cr.construct_TRN(X_0, y_0, return_chosen_params = True)
        trn_1, alphas_1, l1s_1 = cr.construct_TRN(X_1, y_1, return_chosen_params = True)
        trn_glb, alphas_glb, l1s_glb = cr.construct_TRN(X, y, return_chosen_params = True)

        # subtract TRNs
        trn_diff = trn_1 - trn_0
        trn_diff_abs = np.abs(trn_diff)

        # save TRNs
        np.save(out_dir + '/trn_{}.npy'.format(groups[0]), trn_0)
        np.save(out_dir + '/trn_{}.npy'.format(groups[1]), trn_1)
        np.save(out_dir + '/trn_diff.npy', trn_diff)
        np.save(out_dir + '/trn_diff_abs.npy', trn_diff_abs)
        np.save(out_dir + '/trn_global.npy', trn_glb)

        # save parameters chosen by CV
        np.save(out_dir + '/trn_{}_alphas.npy'.format(groups[0]), alphas_0)
        np.save(out_dir + '/trn_{}_alphas.npy'.format(groups[1]), alphas_1)
        np.save(out_dir + '/trn_{}_l1s.npy'.format(groups[0]), l1s_0)
        np.save(out_dir + '/trn_{}_l1s.npy'.format(groups[1]), l1s_1)

    elif smode == 1: # TRNs with shuffling of target gene vector

        shuffle_targets = np.loadtxt(targetfile, delimiter = '\n', dtype = str)
        target_dict = { t: ordered_targets.index(t) for t in shuffle_targets }

        # get indices of samples
        ind_0 = np.where(metadata.group == groups[0])[0]
        ind_1 = np.where(metadata.group == groups[1])[0]

        # split X and y using indices
        X_0 = X[ind_0, :]
        y_0 = y[ind_0, :]
        X_1 = X[ind_1, :]
        y_1 = y[ind_1, :]
        
        with Pool(njobs) as pool:
            pool.starmap(target_organizer, [ (target_dict[shuffle_targets[i]], shuffle_targets[i], (X_0, y_0, X_1, y_1), out_dir) for i in range(len(shuffle_targets)) ])

    elif smode == 2: # TRNs with shuffling of sample phenotype

        # set split size based on number of samples in group 0
        size = metadata.group.value_counts().loc[groups[0]]
        print('Split size = {}'.format(size))

        if nchoosek == True:
            with Pool(njobs) as pool:
                pool.starmap(splitter, [ (i, (X, y, size), out_dir) for i in range(n) ])
        else:
            scramble_mat = np.random.randint(2, size = (int(n), np.shape(X)[0]))
            with Pool(njobs) as pool:
                pool.starmap(scrambler, [ (i, (X, y, scramble_mat), out_dir) for i in range(n) ])

    elif smode == 3: # leave-one-out TRNs

        treatments = list(set(metadata.treatment.to_list()))

        # can run this mode using a treatment column for two phenotypes or any treatment contribution to the global TRN
        if len(groups) == 2:

            # original TRNs
            ind_0 = np.where(metadata.group == groups[0])[0]
            ind_1 = np.where(metadata.group == groups[1])[0]

            X_0 = X[ind_0, :]
            y_0 = y[ind_0, :]
            X_1 = X[ind_1, :]
            y_1 = y[ind_1, :]

            trn_0 = cr.construct_TRN(X_0, y_0)
            trn_1 = cr.construct_TRN(X_1, y_1)

            for treatment in treatments:

                # leave out the current treatment group
                ind_t = np.where(metadata.treatment == treatment)[0]
                ind_0u = [ i for i in ind_0 if i not in ind_t ]
                ind_1u = [ i for i in ind_1 if i not in ind_t ]

                # split X and y using updated indices
                X_0 = X[ind_0u, :]
                y_0 = y[ind_0u, :]
                X_1 = X[ind_1u, :]
                y_1 = y[ind_1u, :]

                # construct Elastic Net TRNs
                trn_0u = cr.construct_TRN(X_0, y_0)
                trn_1u = cr.construct_TRN(X_1, y_1)

                # subtract leave-one out TRNs (inter-TRN contribution of treatment)
                trn_diff = trn_1u - trn_0u
                trn_diff_abs = np.abs(trn_diff)
                np.save(out_dir + '/{}_contribution_to_diff_trn.npy'.format(treatment), trn_diff)
                np.save(out_dir + '/{}_contribution_to_diff_trn_abs.npy'.format(treatment), trn_diff_abs)

                # find out which group is missing a treatment group and subtract the leave-one-out TRN from the original (intra-TRN contribution of treatment)
                if len(ind_0) > len(ind_0u):
                    missing = groups[0]
                    np.save(out_dir + '/{}_trn_leave_{}_out.npy'.format(missing, treatment), trn_0u)
                    trn_diff = trn_0 - trn_0u
                else:
                    missing = groups[1]
                    np.save(out_dir + '/{}_trn_leave_{}_out.npy'.format(missing, treatment), trn_1u)
                    trn_diff = np.abs(trn_1 - trn_1u)
                
                trn_diff_abs = np.abs(trn_diff)
                np.save(out_dir + '/{}_contribution_to_{}_trn.npy'.format(treatment, missing), trn_diff)
                np.save(out_dir + '/{}_contribution_to_{}_trn_abs.npy'.format(treatment, missing), trn_diff_abs)

        else:

            # original TRN
            trn_0 = cr.construct_TRN(X, y)

            for treatment in treatments:

                # leave out the current treatment group
                ind_t = np.where(metadata.treatment == treatment)[0]
                ind_0u = [ i for i in range(X.shape[0]) if i not in ind_t ]

                # split X and y using updated indices
                X_0 = X[ind_0u, :]
                y_0 = y[ind_0u, :]

                # construct Elastic Net TRNs
                trn_0u = cr.construct_TRN(X_0, y_0)

                # subtract leave-one out TRN from original (contribution of treatment)
                trn_diff = trn_0 - trn_0u
                trn_diff_abs = np.abs(trn_diff)
                np.save(out_dir + '/{}_contribution_to_trn.npy'.format(treatment), trn_diff)
                np.save(out_dir + '/{}_contribution_to_trn_abs.npy'.format(treatment), trn_diff_abs)


    print('\nFinished constructing TRNs.')