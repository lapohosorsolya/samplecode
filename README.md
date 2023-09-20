# samplecode

This repository holds representative samples of code I've written for various projects. [Descriptions](#brief-description-of-directories) are available below. I have also included links to a workshop and lab guide I've written in the [Extras](#extras) section.

## Contents

+ [Repository Tree](#repository-tree)
+ [Descriptions of Directories](#brief-description-of-directories)
  + [Bioinformatics](#bioinformatics)
  + [Deep Learning](#deeplearning)
  + [Machine Learning](#machinelearning)
  + [Network Inference](#networkinference)
  + [Pipelines](#pipelines)
+ [Extras](#extras)
  + [Workshop](#workshop)
  + [Lab Guide](#lab-guide)


## Repository Tree
    .
    ├── bioinformatics                # miscellaneous exerpts/scripts
    │   ├── KEGGparser.py
    │   ├── buildKEGGrefs.sh
    │   ├── convertMotifs.sh
    │   ├── filterATAC.sh
    │   ├── getRefSeqByLoc.py
    │   ├── make_trackDb.py
    │   ├── ncbi_to_ens.py
    │   └── simple_DGE.R
    ├── deeplearning                  # project 1 (full working example)
    │   ├── model_training.py
    │   ├── models.py
    │   ├── multiome.py
    │   ├── plot_metrics.py
    │   ├── train_model_nested_cv.py
    │   ├── training_logger.py
    │   └── utilities.py
    ├── machinelearning               # project 2 (working sample)
    │   ├── IRFC2.py
    │   ├── makefeatures.py
    │   ├── scgo.py
    │   └── svm.py
    ├── networkinference              # project 3 (working sample)
    │   ├── TRN.py
    │   ├── basic_diff_net_analysis.py
    │   ├── construct_TRN.py
    │   ├── filter_TRN_edges.py
    │   └── stability_selection.py
    └── pipelines                     # bash wrapper scripts
        ├── processATAC
        └── repeat_run_model

## Descriptions of Directories

### bioinformatics

A collection of utility scripts written in Bash, Python and R for preprocessing, analyzing, or visualizing genomic data (RNA-seq, ATAC-seq).

### deeplearning

Code from a project on predicting single-cell gene expression from ATAC-seq signals and DNA sequences. Complete with:

- A 1D convolution-based deep learning model (PyTorch)
- Nested cross-validation with random search for hyperparameter tuning, and with random sampling of the data (with various options to split the data by cell/gene to avoid leakage)
- An extensive custom training logger class that works with early stopping and saves validation and test results, as well as the best models
- A multiomic data class to link the PyTorch DataLoader with the multiomic H5AD data file containing paired single-cell RNA-seq and ATAC-seq data
- Utility functions
- A script to automatically generate plots of test metrics (using the logger's output)

The entire nested CV process can be run from the command line (with additional user-defined arguments), and training can be resumed in case of interruption. Any PyTorch model (with the same inputs and outputs) can be plugged into the models module, with no other code needing modification. The name of the new model can simply be given as a command line argument.

### machinelearning

Code from a project on predicting broken and intact cells in single-cell RNA-seq data. Gene expression is summarized into functional features (ontologies/pathways) and these are used as input to an SVM or iterative random forest classifier (IRFC).

### networkinference

Code from a project on reconstructing transcriptional regulatory networks (TRNs), bipartite graphs with transcription factor (TF) and target gene nodes. Gene expression data is taken from two biological conditions/phenotypes. A TRN is reconstructed for each condition via Elastic Net regression, in which each target gene's expression is modelled as a function of the gene expression of all TFs. Edge weights in the inferred TRN are derived from the coefficients learned by Elastic Net. 

Inferred TRNs can be compared between two conditions via basic differential network analysis. Sample code is included for additional stability analysis of Elastic Net regression, and for automated filtering of TRN edges.

### pipelines

This directory contains two Bash wrappers to run pipelines from the command line. One takes a directory of raw ATAC-seq fastq files and processes them into peak files. The other pipeline runs a machine learning model for each pair of training and testing data files in a given directory.

## Extras

### Workshop

I designed and taught a workshop on ATAC-seq data processing and analysis. The full demonstration is available [here](https://fonseca.lab.mcgill.ca/resources/20220810_ATAC_Analysis_Demo/guide.html).

### Lab Guide

I also wrote a guide for new students joining my PhD supervisor's lab, explaining command line basics, how to use our server, beginner coding tips, and some bioinformatic tools. Please see the guide [here](https://fonseca-lab-mcgill.notion.site/Command-Line-Basics-a6f9cae0dbb9435fbd1271b50d9a9055).