import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import hypergeom
from sklearn.linear_model import ElasticNetCV


'''
Two collections of functions used for TRN reconstruction from genomic data and visualization of inferred TRNs.
'''


class construction:


    def map_name_to_ens(gene_ann, gene_ens):
        """
        Make a dictionary mapping gene symbols to Ensembl IDs.

        Parameters
        ----------
        gene_ann : list of str
            gene names
        gene_ens : list of str
            gene Ensembl IDs

        Returns
        -------
        gene_dict : dict
            dictionary of gene symbol keys and Ensembl ID values
        """
        gene_dict = {}
        for i in range(len(gene_ann)):
            a = gene_ann[i].split('|')[0]
            gene_dict[a] = gene_ens[i]
        return gene_dict


    def construct_TRN(X, y_mat, return_chosen_params = False):
        """
        Construct a bipartite TRN using elastic net regression.

        Parameters
        ----------
        X : numpy.Array
            regression matrix (TF expression)
        y_mat : numpy.Array
            target gene expression

        Returns
        -------
        coef_mat : numpy.Array
            TRN
        """
        targets = np.shape(y_mat)[1]
        coef_mat = np.empty([np.shape(X)[1], targets])
        alphas = np.empty(targets)
        l1s = np.empty(targets)
        for i in range(targets):
            y = y_mat[:, [i]]
            EN_model = ElasticNetCV(l1_ratio = 0.5, cv = 4)
            EN_model.fit(X, y)
            coef_mat[:, i] = EN_model.coef_
            if return_chosen_params == True:
                alphas[i] = EN_model.alpha_
                l1s[i] = EN_model.l1_ratio_
        if return_chosen_params == True:
            return coef_mat, alphas, l1s
        else:
            return coef_mat


    def shuffled_target_EN(X, y, n_scrambles):
        """
        Construct a bipartite TRN using elastic net regression, shuffling target gene vectors `n_scrambles` times.

        Parameters
        ----------
        X : numpy.Array
            regression matrix
        y : numpy.Array
            target gene expression
        n_scrambles : int
            number of times to shuffle the target gene vector

        Returns
        -------
        coef_mat : numpy.Array
            `n_scrambles` stacked edge weight vectors for a target gene
        """
        coef_mat = np.empty([np.shape(X)[1], n_scrambles])
        for i in range(n_scrambles):
            np.random.shuffle(y)
            EN_model = ElasticNetCV(l1_ratio = 0.5, cv = 4)
            EN_model.fit(X, y)
            coef_mat[:, i] = EN_model.coef_
        return coef_mat



class comparison:


    def simplify_coefs(df):
        """
        Make a ternary TRN.

        Parameters
        ----------
        df : pandas.DataFrame
            TRN

        Returns
        -------
        df : pandas.DataFrame
            ternary TRN (all positive edges are 1, all negative edges are -1, non-edges are 0)
        """
        for i in range(df.shape[0]):
            for j in range(df.shape[1]):
                coef = df.iloc[i, j]
                if coef == 0:
                    df.iloc[i, j] = 0
                elif coef < 0:
                    df.iloc[i, j] = -1
                else:
                    df.iloc[i, j] = 1
        return df


    def binarize(df):
        """
        Make a binary TRN.

        Parameters
        ----------
        df : pandas.DataFrame
            TRN

        Returns
        -------
        df : pandas.DataFrame
            binary TRN (all edges are 1, non-edges are 0)
        """
        for i in range(df.shape[0]):
            for j in range(df.shape[1]):
                coef = df.iloc[i, j]
                if coef != 0:
                    df.iloc[i, j] = 1
        return df


    def rank_TFs(df):
        """
        Rank TFs by the number of regulated target genes in a TRN.

        Parameters
        ----------
        df : pandas.DataFrame
            binary TRN

        Returns
        -------
        num_genes : pandas.DataFrame
            table of TFs, containing number of genes regulated and corresponding ranks
        """
        tfs = list(df.columns)
        num_genes = pd.DataFrame(columns = ['num_genes'])
        for tf in tfs:
            num_genes.loc[tf] = np.absolute(df[tf]).sum()
        num_genes.sort_values(by = 'num_genes', ascending = False, inplace = True)
        rank = []
        prev_n = 0
        prev_rank = 0
        for n in list(num_genes.num_genes):
            if n == prev_n:
                rank.append(prev_rank)
            else:
                rank.append(prev_rank + 1)
                prev_rank = prev_rank + 1
                prev_n = n
        num_genes['rank'] = rank
        return num_genes


    def rank_genes(df):
        """
        Rank target genes by the number of regulating TFs in a TRN.

        Parameters
        ----------
        df : pandas.DataFrame
            binary TRN

        Returns
        -------
        num_tfs : pandas.DataFrame
            table of genes, containing number of regulating TFs and corresponding ranks
        """
        df = df.T
        genes = list(df.columns)
        num_tfs = pd.DataFrame(columns = ['num_tfs'])
        for gene in genes:
            num_tfs.loc[gene] = np.absolute(df[gene]).sum()
        num_tfs.sort_values(by = 'num_tfs', ascending = False, inplace = True)
        rank = []
        prev_n = 0
        prev_rank = 0
        for n in list(num_tfs.num_tfs):
            if n == prev_n:
                rank.append(prev_rank)
            else:
                rank.append(prev_rank + 1)
                prev_rank = prev_rank + 1
                prev_n = n
        num_tfs['rank'] = rank
        return num_tfs


    def compare_effects(df0, df1):
        """
        Compare overall effects (positive or negative) for each TF in two TRNs.

        Parameters
        ----------
        df0 : pandas.DataFrame
            ternary TRN
        df1 : pandas.DataFrame
            ternary TRN

        Returns
        -------
        effect_df : pandas.DataFrame
            table of calculated positive and negative effects for each TF, as well as tendencies
        """
        tfs = df0.columns
        effect_df = pd.DataFrame(columns = ['neg_0', 'pos_0', 'neg_1', 'pos_1'])
        for tf in tfs:
            neg_0 = 0
            pos_0 = 0
            neg_1 = 0
            pos_1 = 0
            vals_0 = df0[tf].value_counts()
            vals_1 = df1[tf].value_counts()
            if 1 in vals_0.index:
                pos_0 = vals_0[1]
            if -1 in vals_0.index:
                neg_0 = vals_0[-1]
            if 1 in vals_1.index:
                pos_1 = vals_1[1]
            if -1 in vals_1.index:
                neg_1 = vals_1[-1]
            effect_df.loc[tf] = [neg_0, pos_0, neg_1, pos_1]
        effect_df['tendency_0'] = effect_df.pos_0 - effect_df.neg_0
        effect_df['tendency_1'] = effect_df.pos_1 - effect_df.neg_1
        return effect_df


    def hg_test(df1, df2, shared_targets, tfs):
        """
        Perform hypergeometric tests on gene sets regulated by each TF in two conditions. For each TF, determines the probability that the gene set regulated by a TF in TRN 2 is drawn from the same population as that in TRN 1.

        Parameters
        ----------
        df1 : pandas.DataFrame
            Population (background) TRN edges. Dataframe with columns for target genes and rows for TFs, with 1 or 0 indicating a regulatory effect.
        df2 : pandas.DataFrame
            Sample (drawn) TRN edges. Dataframe with columns for target genes and rows for TFs, with 1 or 0 indicating a regulatory effect.
        shared_targets : list of str
            List of all target genes present in both TRNs.
        tfs : list of str
            List of all TFs present in both TRNs.

        Returns
        -------
        pvals : list of float
            List of p-values corresponding to the given list of TFs. Each p-value represents the probability that the gene set regulated by a TF in TRN 2 is drawn from the same population as that in TRN 1.
        """
        # M = population size (ex.: total number of possible target genes)
        M = len(shared_targets)
        pvals = []
        for tf in tfs:
            s0 = df1.loc[tf]
            s0_genes = s0[s0 == 1].index.to_list()
            s1 = df2.loc[tf]
            s1_genes = s1[s1 == 1].index.to_list()
            # n = number of successes in the population (ex.: number of target genes in M1 TRN)
            n = len(s0_genes)
            # N = sample size (ex.: number of target genes in healthy TRN)
            N = len(s1_genes)
            # x = number of drawn successes (ex.: intersection of target gene sets in M1 TRN and healthy TRN)
            x = len([ i for i in s0_genes if i in s1_genes ])
            # get p-value
            p = hypergeom.sf(x - 1, M, n, N)
            pvals.append(p)
        return pvals



class plotting:


    def plot_coef_heatmap(mat, filename, work_dir):
        """
        Plot a heatmap of edge weights in a TRN.

        Parameters
        ----------
        mat : numpy.Array
            the TRN edge matrix
        filename : str
            the name of the TRN
        work_dir : str
            where to save the plot

        Returns
        -------
        None
        """
        c = sns.diverging_palette(250, 10, as_cmap = True)
        ax = sns.clustermap(mat, figsize = (10, 10), yticklabels = False, xticklabels = False, col_cluster = True, cbar_kws = {'label': 'EN coef'}, cbar_pos = (1, 0.4, 0.05, 0.2), cmap = c, center = 0)
        plt.yticks(rotation = 0, size = 12) 
        plt.xticks(rotation = 45, ha = 'right', size = 12)
        ax.savefig(work_dir + '/' + filename + '.pdf', bbox_inches = "tight", dpi = 300)


    def equalize_axis_scale(ax):
        """
        Adjust x and y axis limits of a plot such that they are equal (square).

        Parameters
        ----------
        ax : matplotlib.pyplot.axis
            the plot being modified

        Returns
        -------
        None
        """
        x_0, x_1 = ax.get_xlim()
        y_0, y_1 = ax.get_ylim()
        lower_bound = min([x_0, y_0])
        upper_bound = max([x_1, y_1])
        ax.set_xlim(xmin = lower_bound, xmax = upper_bound)
        ax.set_ylim(ymin = lower_bound, ymax = upper_bound)
        ax.locator_params(nbins = 6)


    def rank_step_plots(tfs_ranked_0, genes_ranked_0, tfs_ranked_1, genes_ranked_1, work_dir, labels = ['0', '1']):
        """
        Plot number of TFs regulating a gene and number of genes regulated by a TF, by rank.

        Parameters
        ----------
        tfs_ranked_0 : pandas.DataFrame
            dataframe of ranked TFs from TRN 0
        genes_ranked_0 : pandas.DataFrame
            dataframe of ranked genes from TRN 0
        tfs_ranked_1 : pandas.DataFrame
            dataframe of ranked TFs from TRN 1
        genes_ranked_1 : pandas.DataFrame
            dataframe of ranked genes from TRN 1
        work_dir : str
            directory for output
        labels : list
            labels for the TRNs being compared (length must be 2)

        Returns
        -------
        None
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (10, 4))
        ax1.plot(range(len(genes_ranked_0)), genes_ranked_0.num_tfs, label = labels[0], color = 'blue', drawstyle = 'steps')
        ax1.plot(range(len(genes_ranked_1)), genes_ranked_1.num_tfs, label = labels[1], color = 'red', drawstyle = 'steps')
        ax1.set_xlabel('genes by rank')
        ax1.set_ylabel('number of regulating TFs')
        ax1.set_ylim(ymin = 0)
        ax2.plot(range(len(tfs_ranked_0)), tfs_ranked_0.num_genes, label = labels[0], color = 'blue', drawstyle = 'steps')
        ax2.plot(range(len(tfs_ranked_1)), tfs_ranked_1.num_genes, label = labels[1], color = 'red', drawstyle = 'steps')
        ax2.set_xlabel('TFs by rank')
        ax2.set_ylabel('number of genes regulated')
        ax2.set_ylim(ymin = 0)
        fig.tight_layout()
        handles, labels = ax1.get_legend_handles_labels()
        fig.legend(handles, labels, loc = 'center left', bbox_to_anchor = [1, 0.5], frameon = False)
        fig.savefig(work_dir + '/comparison_tf_gene_ranks.pdf', dpi = 300, bbox_inches = 'tight')


    def edge_effect_plot(df, gene, work_dir, type, labels = ['0', '1']):
        """
        Plot edge weights in one TRN vs another, for a TF or target gene.

        Parameters
        ----------
        df : pandas.DataFrame
            dataframe of edge weights for a chosen TF, from two TRNs
        gene : str
            chosen gene
        work_dir : str
            directory for output
        labels : list
            labels for the TRNs being compared (length must be 2)

        Returns
        -------
        None
        """
        fig, ax = plt.subplots(figsize = (3, 3))
        x = df.effect_0
        y = df.effect_1
        ax_c = np.where(x == 0, 'silver', np.where(x < 0, np.where(y > 0, 'r', 'silver'), np.where(y < 0, 'b', 'silver')))
        ax.scatter(x, y, color = ax_c, s = 6)
        x_0, x_1 = ax.get_xlim()
        y_0, y_1 = ax.get_ylim()
        lower_bound = min([x_0, y_0])
        upper_bound = max([x_1, y_1])
        max_bound = max(abs(lower_bound), abs(upper_bound))
        ax.set_xlim(xmin = -max_bound, xmax = max_bound)
        ax.set_ylim(ymin = -max_bound, ymax = max_bound)
        ax.set_xticks([-1, -0.5, 0, 0.5, 1])
        ax.set_yticks([-1, -0.5, 0, 0.5, 1])
        ax.axhline(y = 0, color = 'black', linewidth = 1)
        ax.axvline(x = 0, color = 'black', linewidth = 1)
        ax.set_title(gene)
        ax.set_xlabel('edge weight in {} TRN'.format(labels[0]))
        ax.set_ylabel('edge weight in {} TRN'.format(labels[1]))
        fig.savefig(work_dir + '/{}_{}_effect.pdf'.format(type, gene), dpi = 300, bbox_inches = 'tight')


    def effect_dist_plot(df, gene, work_dir, type, labels = ['0', '1']):
        """
        Plot edge weights distribution in one TRN vs. another, for a TF or target gene.

        Parameters
        ----------
        df : pandas.DataFrame
            dataframe of edge weights for a chosen TF, from two TRNs
        type : str
            'gene' or 'tf'
        gene : str
            chosen gene
        work_dir : str
            directory for output
        labels : list
            labels for the TRNs being compared (length must be 2)

        Returns
        -------
        None
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize = (4, 2), sharex = True)
        b = 20
        x = df.effect_0
        y = df.effect_1
        sns.distplot(x, color = 'b', bins = b, ax = ax1)
        sns.distplot(y, color = 'r', bins = b, ax = ax2)
        x_0, x_1 = ax1.get_ylim()
        y_0, y_1 = ax2.get_ylim()
        upper_bound = max([x_1, y_1])
        ax1.set_ylim(ymax = upper_bound)
        ax2.set_ylim(ymax = upper_bound)
        ax2.invert_yaxis()
        ax1.axvline(x = 0, color = 'k', linewidth = 1)
        ax2.axvline(x = 0, color = 'k', linewidth = 1)
        ax1.axvline(x = np.median(x), color = 'b', linewidth = 2, linestyle = ':')
        ax2.axvline(x = np.median(y), color = 'r', linewidth = 2, linestyle = ':')
        fig.suptitle('{} edge weight distribution'.format(gene))
        ax2.set_xlabel('edge weight')
        ax1.set_ylabel('{} TRN'.format(labels[0]))
        ax2.set_ylabel('{} TRN'.format(labels[1]))
        fig.subplots_adjust(hspace = 0)
        fig.text(-0.05, 0.5, 'density', va = 'center', rotation = 'vertical')
        fig.savefig(work_dir + '/{}_{}_effect_dist.pdf'.format(type, gene), dpi = 300, bbox_inches = 'tight')


    def corr_plot(df, work_dir, type):
        """
        Plot correlation between edge weights in 2 TRNs, for TFs or target genes.

        Parameters
        ----------
        df : pandas.DataFrame
            table of Spearman correlation rho and p-values for all TFs or target genes
        type : str
            'gene' or 'tf'
        work_dir : str
            directory for output

        Returns
        -------
        None
        """
        fig, ax = plt.subplots(figsize = (3, 3))
        x = df.rho
        y = -np.log10(df.pval)
        # ax_c = np.where(y > 5, 'orange', 'black')
        ax.scatter(x, y, color = 'k', s = 5) # color = ax_c
        ax.axvline(x = 0, color = 'black', linewidth = 1)
        ax.set_title('{} effect correlation'.format(type))
        ax.set_xlabel('Spearman\'s R')
        ax.set_ylabel('-log10(p-value)')
        fig.savefig(work_dir + '/{}_spearman_corr.pdf'.format(type), dpi = 300, bbox_inches = 'tight')


    def effect_plot(effect_df, work_dir, labels = ['0', '1']):
        """
        Plot the number of positive and negative effects of TFs in two TRNs.

        Parameters
        ----------
        effect_df : pandas.DataFrame
            output of the `compare_effects()` function
        work_dir : str
            directory for output
        labels : list
            list of group names in the dataset; defaults to ['0', '1']

        Returns
        -------
        None
        """
        effect_df.sort_values(by = 'pos_0', ascending = False, inplace = True)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize = (10, 4), sharex = True)
        tfs = effect_df.index
        x = np.arange(len(tfs))
        width = 0.4
        groups = ['negative', 'positive']
        pos = x - (len(groups)/2 * width/2)
        ax1.bar(pos, effect_df['pos_0'], width, label = '{} TRN'.format(labels[0]), color = 'b', alpha = 0.8)
        ax2.bar(pos, effect_df['neg_0'], width, label = '{} TRN'.format(labels[0]), color = 'b', alpha = 0.8)
        pos += width
        ax1.bar(pos, effect_df['pos_1'], width, label = '{} TRN'.format(labels[1]), color = 'r', alpha = 0.8)
        ax2.bar(pos, effect_df['neg_1'], width, label = '{} TRN'.format(labels[1]), color = 'r', alpha = 0.8)
        ax2.set_xticks(x)
        ax2.set_xticklabels(tfs, rotation = 90)
        x_0, x_1 = ax1.get_ylim()
        y_0, y_1 = ax2.get_ylim()
        upper_bound = max([x_1, y_1])
        ax1.set_ylim(ymax = upper_bound)
        ax2.set_ylim(ymax = upper_bound)
        ax2.invert_yaxis()
        fig.suptitle('TF effects on targets')
        ax1.set_ylabel('positive')
        ax2.set_ylabel('negative')
        ax1.legend(frameon = False, bbox_to_anchor = [1, 0.8])
        fig.subplots_adjust(hspace = 0)
        fig.text(0.05, 0.5, 'edge count', va = 'center', rotation = 'vertical')
        fig.savefig(work_dir + '/tf_effects_summary.pdf', dpi = 300, bbox_inches = 'tight')


    def tendency_plot(effect_df, work_dir, labels = ['0', '1']):
        """
        Plot the overall tendency of each TF in one TRN vs. another.

        Parameters
        ----------
        effect_df : pandas.DataFrame
            output of the `compare_effects()` function
        work_dir : str
            directory for output
        labels : list
            list of group names in the dataset; defaults to ['0', '1']

        Returns
        -------
        None
        """
        fig, ax = plt.subplots(figsize = (3, 3))
        x = effect_df.tendency_0
        y = effect_df.tendency_1
        ax_c = np.where(x == 0, 'k', np.where(x < 0, np.where(y > 0, 'r', 'silver'), np.where(y < 0, 'b', 'purple')))
        ax.scatter(x, y, color = ax_c, s = 10)
        ax.axhline(y = 0, color = 'black', linewidth = 1)
        ax.axvline(x = 0, color = 'black', linewidth = 1)
        ax.set_title('TF tendency')
        ax.set_xlabel('tendency in {} TRN'.format(labels[0]))
        ax.set_ylabel('tendency in {} TRN'.format(labels[1]))
        fig.savefig(work_dir + '/tf_tendency.pdf', dpi = 300, bbox_inches = 'tight')


    def selected_tf_plots(tfs, norm_df, tf_target_dict_0, tf_target_dict_1, work_dir, metadata, labels = ['0', '1']):
        """
        Plot target gene expression vs. TF expression for a set of TFs.

        Parameters
        ----------
        tfs : list of str
            list of selected TFs
        norm_df : pandas.DataFrame
            normalized gene expression
        tf_target_dict_0 : dict
            dictionary of TFs and their targets in TRN 1
        tf_target_dict_1 : dict
            dictionary of TFs and their targets in TRN 2
        work_dir : str
            directory for output
        metadata : pandas.DataFrame
            table of metadata containing phenotype identities for each sample
        labels : list
            list of group names in the dataset; defaults to ['0', '1']

        Returns
        -------
        None
        """
        fig, (axes1, axes2) = plt.subplots(2, len(tfs), figsize = (16, 3.5), sharey = 'row', sharex = True, gridspec_kw={'height_ratios': [2, 1]})
        norm_df_0 = norm_df.loc[metadata[metadata.group == labels[0]].sampleID.to_list()]
        norm_df_1 = norm_df.loc[metadata[metadata.group == labels[1]].sampleID.to_list()]
        for i, ax in enumerate(axes1):
            # group 0
            targets_0 = tf_target_dict_0[tfs[i]]
            x = []
            y = []
            for target in targets_0:
                ax.scatter(norm_df_0[tfs[i]], norm_df_0[target], s = 5, color = 'b', alpha = 0.3, edgecolors = 'none')
                x.extend(norm_df_0[tfs[i]].to_list())
                y.extend(norm_df_0[target].to_list())
            b, a = np.polyfit(x, y, deg = 1)
            xseq = np.linspace(min(x) - 1, max(x) + 1, num = 100)
            ax.plot(xseq, a + b * xseq, color = 'b', lw = 1)
            # group 1
            targets_1 = tf_target_dict_1[tfs[i]]
            x = []
            y = []
            for target in targets_1:
                ax.scatter(norm_df_1[tfs[i]], norm_df_1[target], s = 5, color = 'r', alpha = 0.3, edgecolors = 'none')
                x.extend(norm_df_1[tfs[i]].to_list())
                y.extend(norm_df_1[target].to_list())
            b, a = np.polyfit(x, y, deg = 1)
            xseq = np.linspace(min(x) - 1, max(x) + 1, num = 100)
            ax.plot(xseq, a + b * xseq, color = 'r', lw = 1)
            ax.set_title(tfs[i] + '\n{}: {},  {}: {}'.format(labels[0], len(targets_0), labels[1], len(targets_1)), fontsize = 10)
        for i, ax in enumerate(axes2):
            sns.kdeplot(norm_df_0[tfs[i]], color = 'b', ax = ax)
            sns.kdeplot(norm_df_1[tfs[i]], color = 'r', ax = ax)
            ax.set_xlabel(None)
            ax.set_ylabel(None)
        fig.text(0.48, -0.02, 'TF expression', va = 'center')
        fig.text(-0.01, 0.65, 'target expression', va = 'center', rotation = 'vertical')
        fig.text(-0.01, 0.225, 'density', va = 'center', rotation = 'vertical')
        fig.tight_layout()
        fig.savefig(work_dir + '/chosen_tfs_vs_targets.pdf', dpi = 300, bbox_inches = 'tight')


    def scramble_pvals_coefs(diffs, pvals, work_dir, name = '', fdr_corrected = False):
        """
        Plot p-values and edge weights (for DECEPTRN mode 2).

        Parameters
        ----------
        diffs : numpy.Array
            differential TRN
        pvals : numpy.Array
            matrix of p-values or FDRs correspoinding to edges in the differential TRN
        work_dir : str
            directory for output
        name : str
            additional label to add to the filename (optional)
        fdr_corrected : bool
            whether the provided p-values are FDR-corrected; defaults to False

        Returns
        -------
        None
        """
        fig, ax = plt.subplots(figsize = (6, 4))
        x = diffs.flatten()
        y = pvals.flatten()
        c = np.where(y < 0.05, 'r', 'k')
        ax.scatter(x, y, color = c, s = 2)
        ax.set_xlabel('absolute difference in coefficient')
        if fdr_corrected == False:
            ax.set_ylabel('p-value')
            lab = 'pvals'
        else:
            ax.set_ylabel('FDR')
            lab = 'fdrs'
        if name != '':
            name = '_' + name
        fig.savefig(work_dir + '/coefs_{}{}.pdf'.format(lab, name), dpi = 300, bbox_inches = 'tight')


    def scramble_pvals_ranked(pvals, work_dir, name = '', fdr_corrected = False):
        """
        Plot p-values by rank (for DECEPTRN mode 2).

        Parameters
        ----------
        pvals : numpy.Array
            matrix of p-values or FDRs correspoinding to edges in the differential TRN
        work_dir : str
            directory for output
        name : str
            additional label to add to the filename (optional)
        fdr_corrected : bool
            whether the provided p-values are FDR-corrected; defaults to False

        Returns
        -------
        None
        """
        fig, ax = plt.subplots(figsize = (6, 4))
        x = range(len(pvals.flatten()))
        y = sorted(pvals.flatten())
        ax.plot(x, y, color = 'k')
        ax.set_xlabel('ranked edges')
        if fdr_corrected == False:
            ax.set_ylabel('p-value')
            lab = 'pvals'
        else:
            ax.set_ylabel('FDR')
            lab = 'fdrs'
        if name != '':
            name = '_' + name
        fig.savefig(work_dir + '/ranked_{}{}.pdf'.format(lab, name), dpi = 300, bbox_inches = 'tight')