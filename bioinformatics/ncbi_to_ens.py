import os, sys, getopt
import pandas as pd

'''
Map the output of buildKEGGrefs.sh to ENSEMBL IDs.

USAGE:

python3 ncbi_to_ens.py
-i /bioinformatics/KEGG/pathways
-l /bioinformatics/KEGG/Homo_sapiens.gene_info
-o /bioinformatics/KEGG/pathway_genes_ensembl
'''

def main(argv):
    try:
        opts, args = getopt.getopt(argv, 'i:o:l:')
    except getopt.GetoptError:
        print('Wrong parameters!')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-i':
            global input_dir
            input_dir = arg
        elif opt == '-l':
            global library_file
            library_file = arg
        elif opt == '-o':
            global out_dir
            out_dir = arg


if __name__ == "__main__":

    main(sys.argv[1:])

    library = pd.read_csv(library_file, sep = '\t', index_col = 0)
    ncbi = library.GeneID.to_list()
    ens = [ i.split('Ensembl:')[-1].split('|')[0] for i in library.dbXrefs.to_list() ]
    translator = { str(ncbi[i]): ens[i] for i in range(len(ncbi)) if 'ENSG' in ens[i] }

    for f in os.listdir(input_dir):
        if '_converted.txt' in f:
            name = f.split('_converted.txt')[0]
            print(name)
            with open(input_dir + '/' + f) as file:
                lines = file.readlines()
                cleanlines = [ line.rstrip('\n') for line in lines ]
                genes = [ i.split('ncbi-geneid:')[-1] for i in cleanlines ]
            ens_genes = [ translator[i] for i in genes if i in translator.keys() ]
            with open(out_dir + '/' + name + '.txt', 'a') as file:
                file.write('\n'.join(ens_genes))