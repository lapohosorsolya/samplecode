import os, sys, getopt
import numpy as np

'''
Prepares a tarckDb file for peak files to be visualized on the UCSC genome browser (via webserver).
'''

def main(argv):
    try:
        opts, args = getopt.getopt(argv, 'i:o:')
    except getopt.GetoptError:
        print('TRY AGAIN...')
        sys.exit(2) 
    for opt, arg in opts:
        if opt == '-i':
            global input_dir
            input_dir = arg
        elif opt == '-o':
            global out_file
            out_file = arg
   

if __name__ == "__main__":

    main(sys.argv[1:])

    trackdb_str = []

    filelist = sorted(os.listdir(input_dir))
    track_names = []

    last_group = ''
    last_color = ''

    for file in filelist:
        track_type = file.split('.')[-1]
        tokens = file.split('_')
        track_name = tokens[5] + '_' + tokens[6]
        if track_type == 'bigBed':
            track_name = track_name + '_peaks'
        elif track_type == 'bigWig':
            track_name = track_name + '_tags'
        if track_name in track_names:
            track_name = track_name + '_' + str(np.random.choice(range(1000)))
        track_label = tokens[5] + ' ' + tokens[6]
        group = tokens[5]
        if last_group != group:
            rgb = np.random.choice(range(256), size = 3)
            color_str = str(rgb[0]) + ',' + str(rgb[1]) + ',' + str(rgb[2])
            last_group = group
            last_color = color_str
        trackdb_str.append('track {}\ntype {}\nbigDataUrl {}\nshortLabel {}\nlongLabel {}\ncolor {}\n\n'.format(track_name, track_type, file, track_label, track_label, color_str))
        track_names.append(track_name)

    with open(out_file, "w") as o:
        o.write('\n'.join(trackdb_str))