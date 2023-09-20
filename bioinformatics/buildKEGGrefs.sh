#!/bin/bash

# Downloads all KEGG pathway files for a given species, then parses the files to extract gene sets.

if (($# != 3)); then
    echo "Usage: $0 <KEGG pathway> <NCBI gene info database> <output directory>"
    # example: bash buildKEGGrefs.sh hsa ncbi-geneid /bioinformatics/KEGG
    # running with nohup is highly recommended!
    # KEGG REST API is very slow
    # nohup /bin/bash -c '{ cd /home/olapohos/sbin; bash buildKEGGrefs.sh hsa ncbi-geneid /bioinformatics/KEGG; }' &
    exit 1
fi


target=${3}/pathways
if [[ -d $target ]]; then
    echo $target exists
else
    mkdir $target
fi
echo "Writing output to $target"

echo "Looking up $1 pathways from KEGG"
PATHWAYS=$(wget -O- http://rest.kegg.jp/list/pathway/${1} | cut -f 1 | sed -r 's/path://g')
echo "$PATHWAYS" > ${3}/pathlist.txt

pathlist=()
while IFS= read -r line; do
    pathlist+=($line)
    wget -O ${target}/${line}.kgml http://rest.kegg.jp/get/path:${line}/kgml
    done < ${3}/pathlist.txt

# python venv
export PYTHONPATH=/usr/bin/python3/
python3 -m venv ./myvenv
source ./myvenv/bin/activate
pip3 install --upgrade pip
pip3 install biopython

for i in "${pathlist[@]}"; do
    echo "Getting genes from " $i
    python3 KEGGparser.py -i ${target}/${i}.kgml -o ${target}/${i}.txt
    done

for i in "${pathlist[@]}"; do
    f="${target}/${i}.txt"
    newfile="${target}/${i}_converted.txt"
    touch ${newfile}
    while IFS= read -r line; do
        wget -O- http://rest.kegg.jp/conv/ncbi-geneid/${line} >> $newfile
        echo $line
        done < $f
    done



##############


# while IFS= read -r line; do
#     echo $line
#     wget -O ./converter/${line} https://rest.kegg.jp/list/${line}
#     done < pathlist.txt

# for f in *; do 
#     cat $f >> ../pathnames.txt
#     done
