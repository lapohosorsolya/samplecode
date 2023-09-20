import os, sys, getopt
from  Bio.KEGG.KGML import KGML_parser

'''
Parses pathway data downloaded through the KEGG API.
'''


def main(argv):
    try:
        opts, args = getopt.getopt(argv, 'i:o:')
    except getopt.GetoptError:
        print('TRY AGAIN...')
        sys.exit(2) 
    for opt, arg in opts:
        if opt == '-i':
            global input_file
            input_file = arg
        elif opt == '-o':
            global out_file
            out_file = arg
   

if __name__ == "__main__":
    main(sys.argv[1:])
    pathway = KGML_parser.read(open(input_file, 'r'))
    genelist = []
    for p in pathway.genes:
        mmulist = p._names
        for i in mmulist:
            genelist.append(i)
    with open(out_file, "w") as o:
        o.write('\n'.join(genelist))