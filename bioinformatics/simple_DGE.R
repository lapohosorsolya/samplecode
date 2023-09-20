#!/usr/bin/env Rscript
args = commandArgs(trailingOnly = TRUE)

# arg[1] must be the input directory
# arg[2] must be the output directory

# Performs differential gene expression analysis between two experimental conditions, using the likelihood ratio test.

# imports
library(edgeR)

# read inputs
input_dir = args[1]
output_dir = args[2]

# directory setup
setwd(input_dir)

# read inputs
targets = readTargets("Metadata.txt")
lengths = read.table("GeneLength.txt", header = TRUE, sep = "\t")

setwd("./Counts")
allcounts = readDGE(targets$filename, group = targets$group, header = F)

# filtering
keep = rowSums(cpm(allcounts) > 1) >= 2
counts = allcounts[keep, ]
counts = calcNormFactors(counts, method = "TMM")
nrow = nrow(counts)

# change directory
setwd(output_dir)

# assign samples to groups based on scramble matrix
design = model.matrix(~ 0 + group, data = counts$samples)
disp_counts = estimateDisp(counts, design)
lrt_fit = glmFit(disp_counts, design)
lrt_M1_M2 = glmLRT(lrt_fit, contrast = c(-1, 1))

# generate a table
toptable = as.data.frame(topTags(lrt_M1_M2, n = nrow, adjust.method = "fdr"))
colnames(toptable) = c("logFC", "logCPM", "LR", "PValue", "FDR")

# label up- and down-regulated genes based on FDR and logFC
toptable$DE <- "NO"
toptable$DE[toptable$logFC > 1 & toptable$FDR < 0.05] <- "UP"
toptable$DE[toptable$logFC < -1 & toptable$FDR < 0.05] <- "DOWN"

# save DEGs
degs = rownames(toptable[toptable$DE != "NO", ])
f = file(paste("DEGs.txt", sep = ""))
writeLines(degs, f)
close(f)

# save table
write.csv(toptable, "DGEA_results.csv", row.names = TRUE)

# save an expression matrix for regression (RPKM units)
genelist = row.names(counts$counts)
gene_info = subset(lengths, Ensembl %in% genelist)
rpkm = rpkm.DGEList(counts, gene.length = gene_info$Length)
write.csv(rpkm, "expression_RPKM.csv", row.names = TRUE)