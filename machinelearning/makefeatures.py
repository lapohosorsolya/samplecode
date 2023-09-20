#!/usr/bin/env python3
import os, sys, argparse
import numpy as np
import pandas as pd
import scanpy as sc
import time
from datetime import datetime
from scgo import ldmethods


'''
Makes input features for the Iterative Random Forest Classifier.
Maps KEGG pathways and/or GO terms to genes in an scRNA-seq data file.
'''


normal = '\033[0m'
blue = '\033[94m' 
red = '\033[93m'


if __name__ == '__main__':

    first_tic = time.process_time()
    now = datetime.now()

    ###### READ INPUTS ######

    parser = argparse.ArgumentParser(description = 'Use this program to build a set of features (GO terms and/or KEGG pathways) for use in the iterative random forest classifier (IRFC2.py)')
    parser.add_argument('workdir', help = 'full path to directory containing all input files', type = str) # '/mnt/data/olapohos/sclivedead/ld_data/input/'
    parser.add_argument('--go', help = 'name of file containing ranked GO terms', default = 'dead_ranked_goterms.csv', type = str)
    parser.add_argument('--gaf', help = 'name of file containing GO annotations (GAF)', default = 'mgi.gaf', type = str)
    parser.add_argument('--library', help = 'name of file containing NCBI gene database', default = 'Mus_musculus.gene_info', type = str)
    parser.add_argument('--kegg', help = 'name of directory containing kegg pathway files with gene names', default = 'kegg_pathways/', type = str)
    parser.add_argument('datafile', help = 'full path to count matrix file (.h5ad format)', type = str) # work_dir + 'Fonseca-batches.h5ad'
    parser.add_argument('writedir', help = 'full path to directory where output files will be written', type = str) # '/mnt/data/olapohos/sclivedead/ld_data/trial_Fonseca/'
    parser.add_argument('--genesets', help = 'types of gene sets to use as features (valid inputs are <both>, <KEGG>, or <GO>)', default = 'both', type = str)
    parser.add_argument('--featurereps', help = 'representation of gene expression to use in features (valid inputs are <both>, <mean>, or <SD>)', default = 'both', type = str)
    parser.add_argument('--initdead', help = 'percentage of reads in mitochondrial genes to use as cutoff for assigning intial viability labels', default = 5, type = float)
    args = parser.parse_args()
    
    work_dir = args.workdir
    go_terms_file = work_dir + args.go
    ann_file = work_dir + args.gaf
    library_file = work_dir + args.library
    kegg_dir = work_dir + args.kegg

    data_file = args.datafile
    write_dir = args.writedir
    genesets = args.genesets
    featurereps = args.featurereps
    init_dead = args.initdead
    
    if not os.path.isdir(write_dir):
        os.mkdir(write_dir)

    # determine gene sets to use
    if genesets == 'GO':
        feature_types = ['GO']
    elif genesets == 'KEGG':
        feature_types = ['KEGG']
    else:
        feature_types = ['GO', 'KEGG']

    # determine feature representation to use
    if featurereps == 'mean':
        feature_data = ['mean']
    elif featurereps == 'SD':
        feature_data = ['SD']
    else:
        feature_data = ['mean', 'SD']

    # read data
    go_terms = pd.read_csv(go_terms_file, header = None)
    origdata = sc.read_h5ad(data_file)
    library = pd.read_csv(library_file, sep = '\t')

    # save information to file
    params = [ ['work_dir', work_dir], ['go_terms_file', go_terms_file], ['ann_file', ann_file], ['library_file', library_file], ['kegg_dir', kegg_dir], ['data_file', data_file], ['write_dir', write_dir], ['feature_types', ', '.join(feature_types)], ['feature_data', ', '.join(feature_data)] ]
    params_df = pd.DataFrame(params, columns = ['var', 'val'])
    params_df.to_csv(write_dir + 'params.tsv', index = False, sep = '\t')
    

    ###### ORGANIZE DATA ######

    print('Filtering doublets and assigning initial labels...')

    # basic filtering; use mean counts in all cells as reference for removing doublets
    sc.pp.calculate_qc_metrics(origdata, percent_top = None, log1p = False, inplace = True)
    mean_counts = np.mean(origdata.obs.total_counts.values)
    sc.pp.filter_cells(origdata, max_counts = 2 * mean_counts)

    # assign viability labels
    # origdata.var['mt'] = origdata.var_names.str.startswith('mt-') # doesn't work for all ref genes
    mito_genes = ['ATP6', 'ATP8', 'COX1', 'COX2', 'COX3', 'CYTB', 'ND1', 'ND2', 'ND3', 'ND4', 'ND4L', 'ND5', 'ND6']
    mito = []
    for name in origdata.var_names:
        if name in mito_genes:
            mito.append(True)
        else:
            mito.append(False)
    origdata.var['mt'] = mito
    sc.pp.calculate_qc_metrics(origdata, qc_vars = ['mt'], percent_top = None, log1p = False, inplace = True)
    origdata.obs['dead'] = (origdata.obs['n_genes_by_counts'] < 200) | (origdata.obs['pct_counts_mt'] > init_dead)

    # normalize to total mapped reads
    sc.pp.normalize_total(origdata, target_sum = 1e4)
    sc.pp.log1p(origdata, base = 2)

    # save anndata object
    origdata.write_h5ad(write_dir + 'intermediate_data.h5ad', compression = 'gzip', compression_opts = 9)


    ###### BUILD FEATURES ######

    print('Building features from library...')
    genes = origdata.var_names.tolist()
    feature_list = []

    # append GO terms to feature list if needed
    print('Assembling GO features...')
    if 'GO' in feature_types:
        # map genes to GO terms
        go_terms.columns = ['Term', 'Pvalue']
        go_terms = list(go_terms['Term'])
        go_map = ldmethods.build_GO_map(ann_file, genes, go_terms)
        # remove GO terms with no genes in the data
        new_go = []
        for term in go_terms:
            if term in go_map:
                new_go.append(term)
        # make mean and/or SD feature names
        if 'mean' in feature_data and 'SD' in feature_data:
            for feature in new_go:
                feature_list.append(feature + '_mean')
                feature_list.append(feature + '_SD')
        elif 'SD' in feature_data:
            for feature in new_go:
                feature_list.append(feature + '_SD')
        else:
            for feature in new_go:
                feature_list.append(feature + '_mean')

    # append KEGG pathways to feature list, if needed
    print('Assembling KEGG features...')
    if 'KEGG' in feature_types:
        # map genes to KEGG pathways
        translator = ldmethods.translate_gene(library, genes)
        kegg_map = ldmethods.build_KEGG_map(kegg_dir, translator)
        # build list of KEGG features
        kegg_pathways = []
        if 'mean' in feature_data:
            a = [ k + '_mean' for k in kegg_map.keys() ]
            kegg_pathways.extend(a)
        if 'SD' in feature_data:
            a = [ k + '_SD' for k in kegg_map.keys() ]
            kegg_pathways.extend(a)
        # add KEGG pathways to feature list
        feature_list.extend(kegg_pathways)

    # set the feature dict to include GO and/or KEGG
    if 'GO' in feature_types and 'KEGG' in feature_types:
        feature_dict = { **go_map, **kegg_map }
    elif 'KEGG' in feature_types:
        feature_dict = kegg_map
    else:
        feature_dict = go_map

    # rank all features by p-value using the initial viability labels
    ranked_features = ldmethods.rank_features(feature_list, origdata, feature_dict, save_df = True, outpath = write_dir)

    # save intermediate of feature dict, mapping takes a long time to run
    print('Saving feature dictionary...')
    intermed = [ [i, ','.join(feature_dict[i])] for i in feature_dict ]
    df = pd.DataFrame(intermed, columns = ['feature', 'genes'])
    df.to_csv(write_dir + 'feature_dict.tsv', index = False, sep = '\t')

    # save intermediate of ordered feature list
    with open(write_dir + 'ranked_feature_list.txt', 'w') as f:
        for item in ranked_features:
            f.write("%s\n" % item)

    last_toc = time.process_time()
    runtime = last_toc - first_tic
    print('Finished making features in {} sec'.format(runtime))