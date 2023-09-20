# samplecode
A collection of some sample code I've written for various computational biology projects.

## Tree
  .
  ├── bioinformatics
  │   ├── KEGGparser.py
  │   ├── buildKEGGrefs.sh
  │   ├── convertMotifs.sh
  │   ├── filterATAC.sh
  │   ├── getRefSeqByLoc.py
  │   ├── make_trackDb.py
  │   ├── ncbi_to_ens.py
  │   └── simple_DGE.R
  ├── deeplearning
  │   ├── model_training.py
  │   ├── models.py
  │   ├── multiome.py
  │   ├── plot_metrics.py
  │   ├── train_model_nested_cv.py
  │   ├── training_logger.py
  │   └── utilities.py
  ├── machinelearning
  │   ├── IRFC2.py
  │   ├── makefeatures.py
  │   ├── scgo.py
  │   └── svm.py
  ├── networkinference
  │   ├── TRN.py
  │   ├── basic_diff_net_analysis.py
  │   ├── construct_TRN.py
  │   ├── filter_TRN_edges.py
  │   └── stability_selection.py
  └── pipelines
      ├── processATAC
      └── repeat_run_model

## Brief Description of Contents

### bioinformatics
A collection of utility scripts for preprocessing or visualizing genomic data.

### deeplearning
Code from a project on predicting single-cell gene expression from ATAC-seq signals and DNA sequences. Complete with:

- a 1D convolution-based deep learning model (PyTorch)
- nested cross-validation with random search for hyperparameter tuning, and random sampling of the data (with various options to split the data by cell/gene to avoid leakage)
- an extensive custom training logger class that works with early stopping and saves validation and test results, as well as the best models
- a multiomic data class to link the PyTorch DataLoader with the multiomic H5AD data file containing paired single-cell RNA-seq and ATAC-seq data
- utility functions
- a script to generate plots of test metrics

The entire nested CV process can be run from the command line (with additional user-defined arguments), and training can be resumed in case of interruption. Any PyTorch model (with the same inputs and outputs) can be plugged into the models module, with no other code needing modification. The name of the new model can simply be given as a command line argument.

### machinelearning
Code from a project on predicting broken and intact cells in single-cell RNA-seq data.
