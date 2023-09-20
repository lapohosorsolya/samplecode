import torch
import anndata
import numpy as np
from tqdm import tqdm


class MultiOmicDataset():
    '''
    A class to organize the data and fetch examples by index for model training.
    Contains paired RNA-seq and ATAC-seq data for each single cell, along with sequences.
    Source data files must be indexed in the same order.
    '''
    def __init__(self, src_dir):
        print('::: Reading data...')
        self.rna = np.asarray(anndata.read_h5ad(src_dir + '/rna.h5ad').X.todense(), dtype = int)
        self.atac = np.asarray(anndata.read_h5ad(src_dir + '/atac.h5ad').X.todense(), dtype = int)
        self.seq = np.load(src_dir + '/seqs.npy')


    def encode_seq(self, seq):
        return torch.nn.functional.one_hot(torch.tensor(seq), 4).T
    

    def fetch_samples(self, indices):
        print('::: Fetching samples...')
        n = indices.shape[0]
        s = torch.zeros((n, 4, 500), dtype = int)
        a = torch.zeros((n), dtype = int)
        r = torch.zeros((n), dtype = int)
        for i in tqdm(range(n), total = n):
            seq = self.seq[indices[i][1]]
            s[i] = self.encode_seq(seq)
            a[i] = self.atac[indices[i][0], indices[i][1]]
            r[i] = self.rna[indices[i][0], indices[i][1]]
        return s, a, r
    

