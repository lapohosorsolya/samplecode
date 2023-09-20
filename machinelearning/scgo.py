import os
import numpy as np
import pandas as pd
import anndata
import scanpy as sc
import seaborn as sns
import matplotlib.pyplot as plt
import time
from scipy.stats import mannwhitneyu
from goatools.anno.gaf_reader import GafReader
from statsmodels.stats.multitest import multipletests
from sklearn.metrics import confusion_matrix, accuracy_score

'''
Functions used by makefeatures.py, IRFC2.py, and SVM.py.
'''


class ldmethods:


    def make_features(index, n, adata, feature_dict, feature_list):
        '''
        Generate an anndata object using gene sets as features. Features can represent mean or standard deviation of gene expression within a gene set (detected from feature name suffix).

        Parameters
        ----------
        index : int
            start index
        n : int
            number of features to use
        adata : anndata.AnnData
            anndata object containing gene expression data
        feature_dict : dict
            dictionary mapping genes to features terms
        feature_list : list
            ranked list of features

        Returns
        -------
        adata_features : anndata.AnnData
            anndata object with feature var_names (GO terms or KEGG pathways)
        '''
        # get the next n features from the ranked feature list
        print('Grabbing features from index {} to {}'.format(index, index + n))
        new_features = feature_list[index:index + n]
        counts = np.empty([len(new_features), adata.shape[0]])
        # extract the mean or standard deviation of gene expression for each feature
        for i, feat in enumerate(new_features):
            name = feat.split('_')[0]
            measure = feat.split('_')[-1]
            genevars = list(feature_dict[name])
            if measure == 'mean':
                counts[i] = np.mean(adata[:, genevars].to_df(), axis = 1)
            else:
                counts[i] = np.std(adata[:, genevars].to_df(), axis = 1)
        # construct a new AnnData object using the new features
        mtx = counts.T
        obs = adata.obs
        var = pd.DataFrame(index = new_features)
        adata_features = anndata.AnnData(X = mtx, obs = obs, var = var)
        return adata_features


    def rank_KEGG_features(kegg_pathways, adata, kegg_map):
        '''
        Rank KEGG features by p-values from live-dead comparisons.

        Parameters
        ----------
        kegg_pathways : list of str
            list of kegg pathway features to rank
        adata : anndata.AnnData
            anndata object containing gene expression
        kegg_map : dict
            dictionary mapping genes to KEGG pathways
        
        Returns
        -------
        ranked_list : list of str
            list of ranked features
        '''
        # make features
        kegg_adata = ldmethods.make_features(0, len(kegg_pathways), adata, kegg_map, kegg_pathways)
        # compare live and dead
        df = ldmethods.compare_live_dead(kegg_adata)
        # return ranked list of features
        ranked_list = list(df.index)
        return ranked_list

    
    def rank_features(features, adata, feature_dict, save_df = False, outpath = None):
        '''
        Rank features (gene sets) by p-values from live-dead comparison.

        Parameters
        ----------
        features : list of str
            list of gene sets to rank
        adata : anndata.AnnData
            anndata object containing gene expression
        feature_dict : dict
            dictionary mapping genes to gene sets
        save_df : bool
            whether the entire table of initial p-values should be saved (default False)
        outpath : str
            the directory where the df should be written, if save_df is True
        
        Returns
        -------
        ranked_list : list of str
            list of ranked features
        '''
        # make features
        new_adata = ldmethods.make_features(0, len(features), adata, feature_dict, features)
        # compare live and dead
        df = ldmethods.compare_live_dead(new_adata)
        # save the dataframe
        if save_df == True and outpath != None:
            df.to_csv(outpath + 'initial_feature_pvals.csv')
        # return ranked list of features
        ranked_list = list(df.index)
        return ranked_list


    def build_GO_map(annfile, genes, subset = None, upper = False):
        '''
        Build a dictionary of GO terms and genes from a GAF file.

        Parameters
        ----------
        annfile : str
            path of the GAF file containing GO term and gene annotations
        genes : list
            list of the genes in the data
        subset : None (default) or list
            list of GO terms to use as subset
        upper : bool
            if True, convert all gene symbols to uppercase

        Returns
        -------
        subdict or go_dict : dict
            GO term keys linked to gene lists, subset if provided
        '''
        tic = time.process_time()
        gafreader = GafReader(annfile)
        associations = gafreader.get_associations()
        go_dict = {}
        for a in associations:
            goterm = a.GO_ID
            genesymbol = a.DB_Symbol
            if upper == True:
                genesymbol = genesymbol.upper()
            if genesymbol in genes:
                if goterm not in go_dict.keys():
                    go_dict[goterm] = [genesymbol]
                elif genesymbol not in go_dict[goterm]:
                    go_dict[goterm].append(genesymbol)
        toc = time.process_time()
        if subset != None:
            subdict = { i : go_dict[i] for i in go_dict if i in subset }
            print('Built a dictionary of {} GO terms in {} seconds'.format(len(subdict), toc - tic))
            return subdict
        else:
            print('Built a dictionary of {} GO terms in {} seconds'.format(len(go_dict), toc - tic))
            return go_dict


    def build_KEGG_map(kegg_dir, translator):
        '''
        Build a dictionary of KEGG pathways and their genes.

        Parameters
        ----------
        kegg_dir : str
            path of the directory containing kegg pathways and their NCBI gene IDs
        translator : dict
            dictionary of NCBI gene ID keys and gene symbol values (generated by translate_gene method)

        Returns
        -------
        kegg_dict : dict
            KEGG pathway keys linked to gene lists
        '''
        tic = time.process_time()
        kegg_dict = {}
        k = list(translator.keys())
        for pathway in os.listdir(kegg_dir):
            with open(kegg_dir + pathway) as f:
                ncbi_list = [ int(line.rstrip()) for line in f ]
            pathname = pathway.split('.')[0]
            genelist = [ translator[ncbi] for ncbi in ncbi_list if ncbi in k ]
            kegg_dict[pathname] = list(set(genelist))
        toc = time.process_time()
        print('Built a dictionary of {} KEGG pathways in {} seconds'.format(len(kegg_dict), toc - tic))
        return kegg_dict


    def extract_annot(associations, go_terms):
        '''
        Find MGI gene IDs associated with a list of GO terms.

        Parameters
        ----------
        associations : dict
            dictionary created by GafReader, using a GAF file downloaded from the Gene Ontology Consortium
        go_terms : list
            list of GO terms

        Returns
        -------
        select_assoc : dict
            GO term keys linked to lists of MGI gene IDs
        '''
        # create parallel lists from association dict
        terms = []
        mgi_genes = []
        for key in associations:
            t_list = associations[key]
            for t in t_list:
                terms.append(t)
                mgi_genes.append(key)
        # create dictionary with GO term keys and MGI gene values
        select_assoc = {}
        for term in go_terms:
            newlist = []
            for i, t in enumerate(terms):
                if t == term:
                    newlist.append(mgi_genes[i])
            if len(newlist) > 0:
                select_assoc[term] = newlist
        return select_assoc


    def translate_gene(library, genes):
        '''
        Make a dictionary of NCBI gene IDs mapped to gene symbols.

        Parameters
        ----------
        library : pandas.DataFrame
            dataframe containing a gene database downloaded from NCBI
        genes : list
            gene symbols found in the data

        Returns
        -------
        translator : dict
            NCBI gene ID keys and list values containing only gene symbols in the data
        '''
        # create dictionary with NCBI gene ID keys and gene symbol values
        ncbi = list(library['GeneID'])
        names = list(library['Symbol'])
        translator = { ncbi[i]: names[i] for i in range(len(ncbi)) if names[i] in genes }
        return translator


    def compare_live_dead(adata_go, sort = True):
        '''
        Compare live and dead cells for each feature. Calculate log2 fold change and Wilcoxon signed-rank (Mann-Whitney U) test p-values.

        Parameters
        ----------
        adata_go : anndata.AnnData
            anndata object with features as var_names and obs.dead boolean labels
        sort : bool
            whether the returned dataframe should be sorted by p-value

        Returns
        -------
        df : pandas.DataFrame
            dataframe with mean expression values for dead and live cells, log2 fold change, raw p-values, and FDR-adjusted p-values for each feature
        '''
        df = pd.DataFrame(columns = ['dead_expression', 'live_expression', 'log2fc', 'pval'])
        terms = adata_go.var_names
        # Wilcoxon signed-rank (also MWU) test for each gene set
        for term in terms:
            dead = adata_go[adata_go.obs.dead == True, adata_go.var_names == term].to_df()
            dead_mean = np.mean(dead.values)
            live = adata_go[adata_go.obs.dead == False, adata_go.var_names == term].to_df()
            live_mean = np.mean(live.values)
            fc = dead_mean - live_mean
            if set(dead.values.flatten()) != set(live.values.flatten()):
                _, pval = mannwhitneyu(dead.values, live.values)
            else:
                # print('Skipping MWU test for {} -- all numbers are identical and log2fc = {}'.format(term, fc))
                pval = 1
            df.loc[term] = [dead_mean, live_mean, fc, pval]
        # multiple hypothesis correction
        _, padj, _, _ = multipletests(df['pval'], alpha = 0.05, method = 'fdr_bh', is_sorted = False, returnsorted = False)
        df['fdr_adj_pval'] = padj
        # return sorted dataframe
        if sort:
            df.sort_values(by = 'fdr_adj_pval', ascending = True, inplace = True)
        return df


    def drop_GO(df, p_cutoff = 0.05, fc_cutoff = None):
        '''
        Determine which features to drop based on specified thresholds.

        Parameters
        ----------
        df : pandas.DataFrame
            dataframe returned by compare_live_dead function
        p_cutoff : float
            FDR-adjusted p-value cutoff
        fc_cutoff : float
            fold change cutoff (should be > 1)

        Returns
        -------
        to_drop : list
            GO terms to drop
        to_keep : list
            GO terms to keep
        '''
        pvals = df.fdr_adj_pval
        terms = df.index
        to_drop = []
        to_keep = []
        if fc_cutoff != None:
            fc = df.log2fc
            high = np.log2(fc_cutoff)
            low = - high
            for i, p in enumerate(pvals):
                if (p > p_cutoff) or (low < fc[i] < high):
                    to_drop.append(terms[i])
                else:
                    to_keep.append(terms[i])
        else:
            for i, p in enumerate(pvals):
                if (p > p_cutoff):
                    to_drop.append(terms[i])
                else:
                    to_keep.append(terms[i])
        return to_drop, to_keep


    def similarity(go_terms, ratio = True):
        '''
        Calculate similarity between GO terms, either as a ratio or raw counts. Returns the ordered list of compared GO terms and the corresponding similarity matrix.

        Parameters
        ----------
        go_terms : dict
            dictionary of GO terms in the format { 'GO_1': ['gene_1', 'gene_2', ..., 'gene_n'], ... }
        ratio : bool
            if True, calculate the ratio of the intersection to the union of the compared gene lists (default); if False, return only the number of shared genes

        Returns
        -------
        go_keys : list
            ordered list of GO terms compared in the similarity matrix
        go_matrix : numpy.ndarray
            similarity matrix containing ratios or counts
        '''
        go_matrix = np.empty((len(go_terms), len(go_terms)))
        go_keys = list(go_terms.keys())
        for m, i in enumerate(go_keys):
            list_i = go_terms[i]
            for n, j in enumerate(go_keys):
                list_j = go_terms[j]
                intersect = [ value for value in list_i if value in list_j ]
                count = len(intersect)
                if ratio == False:
                    go_matrix[m, n] = count
                else:
                    union = set(list_i + list_j)
                    go_matrix[m, n] = count / len(union)
        return go_keys, go_matrix


    def plot_sim_matrix(sim_df, filename):
        '''
        Save a heatmap of the GO term similarity matrix.

        Parameters
        ----------
        sim_df : pandas.DataFrame
            similarity dataframe constructed from matrix returned by similarity function
        filename : str
            path and name of output file (without extension)

        Returns
        -------
        None
        '''
        if len(sim_df) < 15:
            ann =  True
        else:
            ann = False
        fig, ax = plt.subplots(figsize = (8, 8))
        sns.heatmap(ax = ax, data = sim_df, annot = ann, square = True, yticklabels = False, xticklabels = False, cbar_kws = {'label': 'similarity'})
        plt.yticks(rotation = 0)
        plt.xticks(rotation = 90)
        fig.savefig(filename + '.png', bbox_inches = 'tight', dpi = 300)
        return


    def plot_final_features(adata, filename, columns = ['dead']):
        '''
        Save a UMAP plot of the final features.

        Parameters
        ----------
        adata : anndata.AnnData
            AnnData object containing the selected features
        filename : str
            path and name of output file

        Returns
        -------
        None
        '''
        # scale and calculate neighborhood
        sc.pp.scale(adata)
        sc.pp.neighbors(adata, n_neighbors = 10, n_pcs = 40, use_rep = 'X')
        sc.tl.umap(adata)
        # convert boolean obs column to categorical type for plotting
        for col in columns:
            adata.obs[col] = adata.obs[col].astype(str).astype('category')
        n_plots = len(columns)
        # plot each column in a subplot
        fig, axes = plt.subplots(1, n_plots, figsize = (3 * n_plots, 3), gridspec_kw = {'wspace':0.2})
        for i in range(n_plots):
            sc.pl.umap(adata, color = columns[i], palette = 'coolwarm', size = 40, ax = axes[i], show = False, legend_loc = None, frameon = False)
        fig.savefig(filename, bbox_inches = 'tight', dpi = 300)
        return


    def plot_iter_progress(x, y, ylab, filename):
        '''
        Save a line plot of iteration progress, such as dead/live ratio or dropped features.
        
        Parameters
        ----------
        x : list-like
            x ordinates
        y : list-like
            y ordinates
        ylab : str
            description of y ordinates
        filename : str
            path and name of output file (without extension)

        Returns
        -------
        None
        '''
        n_iter = len(y)
        if n_iter < 20:
            length = 4
        else:
            length = 0.1 * (n_iter - 20) + 4
        fig, ax = plt.subplots(figsize = (length, 4))
        ax.plot(x, y, color = 'red')
        ax.set_ylabel(ylab)
        ax.set_xlabel('iteration')
        plt.tight_layout()
        fig.savefig(filename + '.png', bbox_inches = 'tight', dpi = 300)
        return


    def score_classifier(true_labels, predicted_labels):
        '''
        Score a binary classifier in multiple ways.

        Parameters
        ----------
        true_labels : list
            list of true binary labels for each observation
        predicted_labels : list
            list of predicted binary labels for each observation

        Returns
        -------
        sensitivity : float
            tp / (tp + fn)
        specificity : float
            tn / (tn + fp)
        precision : float
            tp / (tp + fp)
        accuracy : float
            accuracy score
        F1_score : float
            harmonic mean of precision and sensitivity (recall)
        '''
        tn, fp, fn, tp = confusion_matrix(true_labels, predicted_labels).ravel()
        print(tn, fp, fn, tp)
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        precision = tp / (tp + fp)
        accuracy = accuracy_score(true_labels, predicted_labels)
        F1_score = 2 * (precision * sensitivity) / (precision + sensitivity)
        return sensitivity, specificity, precision, accuracy, F1_score


    def plot_classifier_scores(df, write_dir):
        '''
        Make a bar plot comparing scores from  different classifiers.

        Parameters
        ----------
        df : pandas.DataFrame
            dataframe with columns containing scoring metrics and rows containing classifier performance

        Returns
        -------
        None
        '''
        c = ['lightgray', 'royalblue', 'deepskyblue', 'mediumseagreen', 'darkslategrey']
        labels = df.columns
        x = np.arange(len(labels))
        width = 1 / (len(x) + 1)
        groups = df.index.tolist()
        pos = x - (len(groups)/2 * width/2)
        fig, ax = plt.subplots(figsize = (5, 4))
        for i, group in enumerate(groups):
            ax.bar(pos, df.loc[group], width, label = group, color = c[i])
            pos += width
        ax.set_ylabel('score')
        ax.set_ylim(0, 1)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend(frameon = False)
        plt.tight_layout()
        fig.savefig(write_dir + 'plot_scores.png', bbox_inches = 'tight', dpi = 300)



class preprocessing:


    def label_batches(adata):
        '''
        Parse obs.index to determine unique numbers for each batch (mouse). Add the batch numbers to a new column, 'batch' in adata.obs.

        Parameters
        ----------
        adata : anndata.AnnData
            anndata object with gene expression

        Returns
        -------
        adata : anndata.AnnData
            modified anndata object with batch labels
        '''
        names = list(adata.obs.index)
        mouse = []
        for name in names:
            n = name.split(':')[1]
            mouse.append(n)
        mice = list(set(mouse))
        batches = list(range(1, len(mice)+1))
        batch_dict = { mice[i]: batches[i] for i in range(len(mice)) }
        batch = [batch_dict[i] for i in mouse]
        adata.obs['mouse'] = mouse
        adata.obs['batch'] = batch
        return adata