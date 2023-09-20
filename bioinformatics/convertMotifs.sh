#!/bin/bash
if (($# != 2)); then
    echo "Usage: $0 <search directory name> <output file name>"
    exit 1
fi
echo $1
target=${1}/aux
echo $target
if [[ -d $target ]]; then
    echo $target exists
else
    mkdir $target
fi

cat - > $target/$2 <<EOF
MEME version 4

ALPHABET= ACGT

strands: + -

EOF

shopt -s extglob

for f in ${1}/motif+([0-9]).motif; do
    echo $f
    x=$(sed -n "1s_^.*P:1e-\(.*\)\$_\1_p" $f)
    if (( $x > 11 )); then
	sed "1s_[^:]*:\([^\t]*\)\t.*P:\(.*\)_MOTIF \1\nletter-probability matrix: P= \2_" $f >> $target/$2
	echo >> $target/$2
    else
	echo $f does not qualify
    fi
done
