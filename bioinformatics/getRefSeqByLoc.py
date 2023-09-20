#!/usr/bin/env python3

'''
Extracts sequences from a reference genome, using location information.

COMMAND LINE USAGE:

python getRefSeqByLoc.py
-i <input directory with regions oragnized by chr>
-s <reference genome fasta sequence organized by chr>
-o <output file (.fasta)>

'''

import os, sys, getopt
import pandas as pd
import numpy as np
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord


# colours :)
normal = '\033[0m' # '\x1b[0;37;49m'
blue = '\033[94m' # '\x1b[0;34;49m' 
yellow = '\033[96m' # '\x1b[0;93;49m'
red = '\033[93m' # '\x1b[0;31;49m'


def main(argv):
   
    try:
        opts, args = getopt.getopt(argv, 'i:s:o:')
    except getopt.GetoptError:
        print(red + 'REQUIRED ARGUMENTS:\n' +
              '-i <input directory with regions oragnized by chr>\n' +
              '-s <reference genome fasta sequence organized by chr>\n' +
              '-o <output file (.fasta)>\n' + normal)
        sys.exit(2)
        
    for opt, arg in opts:

        if opt == '-i':
            global input_dir
            input_dir = arg
            print(blue + 'INPUT DIRECTORY:\t' + input_dir + normal)
        elif opt == '-s':
            global mapping_dir
            mapping_dir = arg
            print(blue + 'REFERENCE GENOME:\t' + mapping_dir + normal)
        elif opt == '-o':
            global out_file
            out_file = arg
            print(blue + 'OUTPUT FILE:\t\t' + out_file + normal)
   
        
if __name__ == '__main__':
    
    main(sys.argv[1:]) # start at index 1 because sys.argv[0] is the script name
    
    out = open(out_file, 'w')

    # loop through input directory (contains peak locations organized by chromosome)
    for loc_file in os.listdir(input_dir):

        chr_num = loc_file.split('.')[0]

        loc_data = pd.read_csv(input_dir + loc_file, sep = '\t', header = None, names = ['chr', 'start', 'end', 'tags'])
        locations = loc_data[['start', 'end']].to_numpy()

        # get the corresponding reference chromosome file
        ref_file = mapping_dir + chr_num + '.txt'

        print(yellow + '\nExtracting sequences from chromosome {} . . .'.format(chr_num) + normal)

        ref_list = list(SeqIO.parse(ref_file, 'fasta'))
        ref = ref_list[0].seq

        # loop through locations and write string to output file
        for i in locations:

            start, end = i
            name = 'chr' + str(chr_num) + ':' + str(start) + '-' + str(end)
            extracted = ref[start:end+1]
            extr_seq = SeqRecord(extracted, name = 'sequence', id = name, description = '')
            SeqIO.write(extr_seq, out, "fasta")
              
        print(yellow + '\tDONE' + normal)

    out.close()
    print(blue + 'Finished extracting all sequences!\n' + normal)
