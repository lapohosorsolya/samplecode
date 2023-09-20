import os, sys, getopt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from networkinference.TRN import comparison as cp
from networkinference.TRN import plotting as plot
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler

'''
Performs a basic comparison of transcriptional regulatory networks reconstructed under two different conditions.
'''


def main(argv):
    try:
        opts, args = getopt.getopt(argv, 'i:o:w:')
    except getopt.GetoptError:
        print('TRY AGAIN...')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-i':
            global input_file
            input_file = arg
        elif opt == '-w':
            global work_dir
            work_dir = arg
        elif opt == '-o':
            global out_dir
            out_dir = arg


if __name__ == "__main__":

    main(sys.argv[1:])

    metadata = pd.read_csv(input_file, index_col = 0, sep = '\t')
    grouplist = metadata.group.to_list()
    groups = list(set(grouplist))

    df_0_raw = pd.read_csv(out_dir + '/' + groups[0] + '.csv', index_col = 0)
    df_0 = pd.read_csv(out_dir + '/' + groups[0] + '_filtered.csv', index_col = 0)
    df_0_ternary = pd.read_csv(out_dir + '/' + groups[0] + '_filtered_ternary.csv', index_col = 0)
    df_0_binary = pd.read_csv(out_dir + '/' + groups[0] + '_filtered_binary.csv', index_col = 0)

    df_1_raw = pd.read_csv(out_dir + '/' + groups[1] + '.csv', index_col = 0)
    df_1 = pd.read_csv(out_dir + '/' + groups[1] + '_filtered.csv', index_col = 0)
    df_1_ternary = pd.read_csv(out_dir + '/' + groups[1] + '_filtered_ternary.csv', index_col = 0)
    df_1_binary = pd.read_csv(out_dir + '/' + groups[1] + '_filtered_binary.csv', index_col = 0)

    # rank TFs and genes in TRNs
    tfs_ranked_0 = cp.rank_TFs(df_0_binary)
    genes_ranked_0 = cp.rank_genes(df_0_binary)
    tfs_ranked_0.to_csv(out_dir + '/' + groups[0] + '_tf_ranks.csv')
    genes_ranked_0.to_csv(out_dir + '/' + groups[0] + '_gene_ranks.csv')

    tfs_ranked_1 = cp.rank_TFs(df_1_binary)
    genes_ranked_1 = cp.rank_genes(df_1_binary)
    tfs_ranked_1.to_csv(out_dir + '/' + groups[1] + '_tf_ranks.csv')
    genes_ranked_1.to_csv(out_dir + '/' + groups[1] + '_gene_ranks.csv')

    # step plots for ranked genes and TFs
    plot.rank_step_plots(tfs_ranked_0, genes_ranked_0, tfs_ranked_1, genes_ranked_1, out_dir, labels = groups)

    # edge effect plots for top genes and TFs
    top_genes = list(set(genes_ranked_0.iloc[:5,:].index.to_list() + genes_ranked_1.iloc[:5,:].index.to_list()))
    top_tfs = list(set(tfs_ranked_0.iloc[:5,:].index.to_list() + tfs_ranked_1.iloc[:5,:].index.to_list()))
    gene_dir = out_dir + '/genes'
    if not os.path.exists(gene_dir):
        os.mkdir(gene_dir)
    for gene in top_genes:
        df = pd.concat([df_0_raw.T[gene], df_1_raw.T[gene]], axis = 1)
        df.columns = ['effect_0', 'effect_1']
        plot.edge_effect_plot(df, gene, gene_dir, type = 'gene', labels = groups)
        plot.effect_dist_plot(df, gene, gene_dir, type = 'gene', labels = groups)
    tf_dir = out_dir + '/tfs'
    if not os.path.exists(tf_dir):
        os.mkdir(tf_dir)
    for tf in top_tfs:
        df = pd.concat([df_0_raw[tf], df_1_raw[tf]], axis = 1)
        df.columns = ['effect_0', 'effect_1']
        plot.edge_effect_plot(df, tf, tf_dir, type = 'tf', labels = groups)
        plot.effect_dist_plot(df, tf, tf_dir, type = 'tf', labels = groups)

    # spearman correlation between effects in TRNs
    tfs = tfs_ranked_0.index.to_list()
    genes = genes_ranked_0.index.to_list()
    
    rho_vec = []
    p_vec = []
    for tf in tfs:
        rho, p = spearmanr(a = df_0_raw[tf], b = df_1_raw[tf])
        rho_vec.append(rho)
        p_vec.append(p)
    corr_df = pd.concat([pd.Series(rho_vec, index = tfs, name = 'rho'), pd.Series(p_vec, index = tfs, name = 'pval')], axis = 1)
    corr_df.to_csv(out_dir + '/tf_spearman_corr.csv')
    plot.corr_plot(corr_df, out_dir, type = 'tf')

    rho_vec = []
    p_vec = []
    for gene in genes:
        rho, p = spearmanr(a = df_0_raw.loc[gene], b = df_1_raw.loc[gene])
        rho_vec.append(rho)
        p_vec.append(p)
    corr_df = pd.concat([pd.Series(rho_vec, index = genes, name = 'rho'), pd.Series(p_vec, index = genes, name = 'pval')], axis = 1)
    corr_df.to_csv(out_dir + '/gene_spearman_corr.csv')
    plot.corr_plot(corr_df, out_dir, type = 'gene')

    # compare effects, tendencies
    effect_df = cp.compare_effects(df_0_ternary, df_1_ternary)
    plot.effect_plot(effect_df, out_dir, labels = groups)
    plot.tendency_plot(effect_df, out_dir, labels = groups)

    # select 2 TFs from each quadrant of tendency plot
    x = effect_df.tendency_0
    y = effect_df.tendency_1
    effect_df['quad'] = np.where(x == 0, None, np.where(x < 0, np.where(y > 0, 3, 4), np.where(y < 0, 2, 1)))
    effect_df['tendency_diff'] = np.abs(np.abs(x) - np.abs(y))
    effect_df['scaling_factor'] = np.abs(np.abs(x) + np.abs(y))
    effect_df['scaled_diff'] = np.divide(effect_df.tendency_diff, effect_df.scaling_factor)
    q1 = effect_df[effect_df.quad == 1].sort_values(by = 'scaled_diff').index.to_list()[:2]
    q2 = effect_df[effect_df.quad == 2].sort_values(by = 'scaled_diff').index.to_list()[:2]
    q3 = effect_df[effect_df.quad == 3].sort_values(by = 'scaled_diff').index.to_list()[:2]
    q4 = effect_df[effect_df.quad == 4].sort_values(by = 'scaled_diff').index.to_list()[:2]
    chosen_tfs = q1 + q2 + q3 + q4
    print(chosen_tfs)

    effect_df.to_csv(out_dir + '/tf_effect_summary.csv')

    # transform to log2(RPKM + 1), then scale to standard normal distribution
    exp_df = pd.read_csv('/Users/orsi/Desktop/PhD/Projects/CytokineTRN/Output/expression_RPKM.csv', index_col = 0).T
    exp_df = np.log2(exp_df + 1)
    norm = StandardScaler().fit(exp_df).transform(exp_df)
    norm_df = pd.DataFrame(norm, columns = exp_df.columns, index = exp_df.index)

    # make a dict for tfs and their targets
    tf_target_dict_0 = {}
    tf_target_dict_1 = {}
    for tf in chosen_tfs:
        targets_0 = df_0_binary[df_0_binary[tf] == 1].index.to_list()
        targets_1 = df_1_binary[df_1_binary[tf] == 1].index.to_list()
        tf_target_dict_0[tf] = targets_0
        tf_target_dict_1[tf] = targets_1

    plot.selected_tf_plots(chosen_tfs, norm_df, tf_target_dict_0, tf_target_dict_1, out_dir, metadata, labels = groups)

    # differential network analysis (DNA)
    ternary_diff = df_0_ternary.subtract(df_1_ternary)
    binary_diff = df_0_binary.subtract(df_1_binary)


    print('Compared TRNs')