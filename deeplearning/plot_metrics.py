import sys, getopt
import pickle
from utilities import *
import matplotlib.pyplot as plt


'''
USAGE NOTES

Prerequisite
------------
train_model_nested_cv.py

Required Inputs
---------------
    -d (directory)
        the full path to the directory containing metrics
    -o (output directory)
        the directory where the output should be written

Example
-------
/mnt/data/orsi/anaconda3/bin/python plot_metrics.py -d /mnt/data/orsi/rimodl_output/cellwise_1_predict_8 -o /mnt/data/orsi/rimodl_output/plots


'''


def main(argv):
    try:
        opts, _ = getopt.getopt(argv, 'd:o:')
    except getopt.GetoptError:
        print('\n::: Error: cannot parse command line inputs')
        sys.exit(2) 
    for opt, arg in opts:
        if opt == '-d':
            global input_dir
            input_dir = arg
        elif opt == '-o':
            global output_dir
            output_dir = arg


if __name__ == "__main__":

    main(sys.argv[1:])

    # check if directories exist
    check_path(input_dir, dir = True)
    check_path(output_dir, dir = True)

    name = input_dir.split('/')[-1]

    # read data
    precisions = []
    recalls = []
    tprs = []
    fprs = []
    auprs = []
    aurocs = []
    base_aupr = []
    for i in range(5):
        with open(input_dir + '/{}_metrics.pkl'.format(i), 'rb') as f:
            m = pickle.load(f)
            precisions.append(m['precision'])
            recalls.append(m['recall'])
            auprs.append(m['aupr'])
            aurocs.append(m['auroc'])
            tprs.append(m['tpr'])
            fprs.append(m['fpr'])
            base_aupr = m['base_aupr']

    # make plots
    colors = ['xkcd:goldenrod', 'xkcd:coral', 'xkcd:magenta', 'xkcd:azure', 'xkcd:teal']
    fig, axes = plt.subplots(1, 2, figsize = (6, 3))
    plot_roc_curve_multiple(fprs, tprs, aurocs, axes[0], colors)
    plot_pr_curve_multiple(precisions, recalls, auprs, base_aupr, axes[1], colors)
    fig.tight_layout()
    fig.savefig(output_dir + '/metrics_plots_{}.pdf'.format(name), bbox_inches = 'tight')