#!/bin/bash

# Filters ATAC-seq peaks using Irreproducible Discovery Rate Analysis.
# find_reproducible.py is from the TF Binding Analysis paper (Fonseca & Tao 2019)


# requires inputs: (1) input directory (tag directories of replicates), (2) output directory
if (($# != 2)); then
    echo "Usage: $0 <input directory> <output directory>"
    exit 1
fi


# print the input and output directories
echo "Read files from $1"
echo "Output to $2"


# create subdirectories
peaks_nofilter=${2}/peaks_nofilter
idr_results=${2}/idr_results
mkdir $peaks_nofilter $idr_results
echo "Created subdirectories"


# generate sample list
files=()
samples=()
suffix="_rep"
for d in ${1}/*/; do
    name=$(basename $d)
    files+=($name)
    short=${name%"$suffix"*}
    sample=$(echo $short | rev | cut -d"_" -f1  | rev)
    samples+=($sample)
done

printf -v filelist $'%s\n' "${files[@]}"
echo "$filelist"


# get unique groups
groups=($(printf "%s\n" "${samples[@]}" | sort -u))
numgroups=${#groups[@]}
echo "There are $numgroups groups to analyze:"
echo "${groups[@]}"


# run unfiltered findPeaks and IDR by group
for i in "${groups[@]}"; do
    # find the number of replicates in each group
    instances=($(printf "%s\n" "${samples[@]}" | grep -c "$i"))
    echo $i
    if [[ $instances > 1 ]]; then
        if [[ $instances == 2 ]]; then
            echo "Exactly 2 replicates."
        else
            echo "More than 2 replicates!!! Taking first two of $instances files..."
        fi
        reps=($(echo "$filelist" | grep "$i"))
        echo "Will run IDR with \n ${reps[0]} \n and \n ${reps[1]} \n"
        
        # generate unfiltered peak files for each replicate
        findPeaks ${1}/${reps[0]}/ -style factor -F 0 -L 0 -C 0 > ${peaks_nofilter}/${reps[0]}.txt
        findPeaks ${1}/${reps[1]}/ -style factor -F 0 -L 0 -C 0 > ${peaks_nofilter}/${reps[1]}.txt
        
        # run IDR script
        python3 /home/olapohos/pybin/find_reproducible.py ${peaks_nofilter}/${reps[0]}.txt ${peaks_nofilter}/${reps[1]}.txt $idr_results
        
    else
        echo "Not enough replicates for IDR."
    fi
done


echo "DONE :)"