import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler

'''
Assess the stability of Elastic Net regression for reconstructing TRNs (for a single target gene).
'''

def stability_selection(ElasticNet, alphas, n_bootstrap_iterations, X, y, seed):
    n_samples, n_variables = X.shape
    n_params = alphas.shape[0]

    rnd = np.random.RandomState(seed)
    selected_variables = np.zeros((n_variables, n_bootstrap_iterations))
    stability_scores = np.zeros((n_variables, n_params))

    for idx, a, in enumerate(alphas):
        # bootstrap sampling
        for iteration in range(n_bootstrap_iterations):
            bootstrap = rnd.choice(np.arange(n_samples), size = n_samples // 2, replace = False)
            X_train = X[bootstrap, :]
            y_train = y[bootstrap]

            # fit model
            params = {'alpha': a}
            ElasticNet.set_params(**params).fit(X_train, y_train)
            selected_variables[:, iteration] = (np.abs(ElasticNet.coef_) > 1e-4)

        # compute stability score
        stability_scores[:, idx] = selected_variables.mean(axis = 1)

    return stability_scores


if __name__ == "__main__":

    targetgene = 'ENSMUSG00000005057'

    # read files
    metadata = pd.read_csv('/mnt/data/olapohos/trn/input/metadata.txt', sep = '\t')
    exp_df = pd.read_csv('/mnt/data/olapohos/trn/output/I_target_selection' + '/expression_RPKM.csv', index_col = 0).T
    degs = np.loadtxt('/mnt/data/olapohos/trn/output/I_target_selection' + '/DEGs.txt', delimiter = '\n', dtype = str)
    tfs = np.loadtxt('/mnt/data/olapohos/trn/output/I_tf_selection' + '/SelectedTFs.txt', delimiter = '\n', dtype = str)

    # log-transform and scale
    exp = np.log2(exp_df + 1).to_numpy()
    norm_exp = StandardScaler().fit(exp).transform(exp)

    # find indices of tf columns
    mask_tfs = [ (gene in tfs) for gene in exp_df.columns ]
    ind_tfs = np.where(mask_tfs)[0]

    # remove any tfs from degs (no self-edges)
    targets = [ gene for gene in degs if gene not in tfs ]
    print('Removed {} TFs from target genes (self-edges)'.format(len(degs) - len(targets)))

    # find index of target column
    mask_targets = [ (gene == targetgene) for gene in exp_df.columns ]
    ind_target = np.where(mask_targets)[0]

    # subset normalized expression matrix
    X = norm_exp[:, ind_tfs]
    y = norm_exp[:, ind_target]

    # split expression into two groups
    groups = sorted(list(set(metadata.group.to_list())))
    print('Found {} groups:\n{}'.format(len(groups), groups))
    ind_0 = np.where(metadata.group == groups[0])[0]
    ind_1 = np.where(metadata.group == groups[1])[0]

    # split X and y using indices
    X_0 = X[ind_0, :]
    y_0 = y[ind_0, :]
    X_1 = X[ind_1, :]
    y_1 = y[ind_1, :]

    # perform stability selection
    # l1s = np.array([0.5 , 0.55, 0.6 , 0.65, 0.7 , 0.75, 0.8 , 0.85, 0.9 , 0.95, 1])
    # alphas = np.arange(0.05, 1, 0.05)
    r1 = np.arange(0.001, 0.01, 0.001)
    r2 = np.arange(0.01, 0.05, 0.01)
    r3 = np.arange(0.05, 1, 0.05)
    alphas = np.concatenate((r1, r2, r3))
    n = 1000
    en_0 = ElasticNet(random_state = 0)
    en_1 = ElasticNet(random_state = 0)

    scores_0 = stability_selection(en_0, alphas, n, X_0, y_0, 1)
    scores_1 = stability_selection(en_1, alphas, n, X_1, y_1, 1)

    np.save('/mnt/data/olapohos/trn/output/II_trn_stability' + '/{}_M1_stability_scores_alpha.npy'.format(targetgene), scores_0)
    np.save('/mnt/data/olapohos/trn/output/II_trn_stability' + '/{}_M2_stability_scores_alpha.npy'.format(targetgene), scores_1)


    print('\nFinished running stability selection.')