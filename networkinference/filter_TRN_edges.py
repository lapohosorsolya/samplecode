import sys, getopt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import kneed
from TRN import comparison as cp

'''
Automatically selects the best threshold to filter network edges, using the kneedle library.
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
    groups = list(set(metadata.group.to_list()))

    with open(work_dir + '/TRN_TFs.txt') as file:
        lines = file.readlines()
        tfs = [line.rstrip() for line in lines]

    with open(work_dir + '/TRN_targets.txt') as file:
        lines = file.readlines()
        genes = [line.rstrip() for line in lines]

    for trn in groups:

        trn_df = pd.DataFrame(np.load(work_dir + '/trn_' + trn + '.npy'), index = tfs, columns = genes)
        trn_df.to_csv(out_dir + '/' + trn + '.csv')
        
        edgenames = []
        edgescores = []

        for i, tf in enumerate(tfs):
            for j, gene in enumerate(genes):
                edgenames.append('-'.join([tf, gene]))
                edgescores.append(trn_df.iloc[i, j])

        edge_df = pd.DataFrame(columns = ['edge', 'score'])
        edge_df.edge = edgenames
        edge_df.score = np.abs(edgescores)

        edge_df.sort_values(by = 'score', ascending = False, inplace = True)
        edge_df = edge_df[edge_df.score > 0]
        edge_df.reset_index(inplace = True)


        # max method
        kl = kneed.KneeLocator(edge_df.index, edge_df.score, curve = 'convex', direction = 'decreasing')
        max_y = max(kl.y_difference)
        max_i = np.where(kl.y_difference == max_y)[0][0]
        max_x = kl.x_difference[max_i]
        # print(max_x, max_y)

        # scale back to original
        x_cutoff = (len(edge_df)-1) * max_x
        y_cutoff = edge_df.score[edge_df.index == round(x_cutoff)].values[0]
        # print(x_cutoff, y_cutoff)

        fig, ax = plt.subplots(figsize = (6, 4))
        ax.plot(kl.x, kl.y, color = 'k')
        ax.vlines(x_cutoff, 0, edge_df.score.max(), linestyles = "--", colors = 'r')
        ax.hlines(y_cutoff, 0, len(edge_df), linestyles = "--", colors = 'r')
        ax.set_ylabel('absolute Elastic Net coef')
        ax.set_xlabel('TRN edge by rank')
        ax.set_title(trn + ' Kneedle curve')
        fig.savefig(out_dir + '/' + trn + '_kneedle_curve_max.png', dpi = 300)

        # filter the edges
        threshold = y_cutoff
        df = trn_df
        df_orig = trn_df
        for i in range(df.shape[0]):
            for j in range(df.shape[1]):
                coef = df.iloc[i, j]
                if abs(coef) <= threshold:
                    df.iloc[i, j] = 0
                    df_orig.iloc[i, j] = 0
                elif coef > 0:
                    df.iloc[i, j] = 1
                else:
                    df.iloc[i, j] = -1

        # save filtered TRNs
        df_orig.to_csv(out_dir + '/' + trn + '_filtered.csv')
        df.to_csv(out_dir + '/' + trn + '_filtered_ternary.csv')
        df = df.abs()
        df.to_csv(out_dir + '/' + trn + '_filtered_binary.csv')

    print('Filtered TRNs')