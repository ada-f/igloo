#!/bin/bash
#SBATCH --job-name=process_loops
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 16
#SBATCH --partition cpu
#SBATCH --error logs/process_loops.%j.err
#SBATCH --output logs/process_loops.%j.out
#SBATCH --time 01:00:00

export PDB_DIR="sabdab" 

python 0_process_annotate_sabdab_data.py \
    --parquet_in sabdab_2025-05-06-paired.parquet \
    --ncpu 16 --annotation_level "residue"

python 1_load_extract_loops.py \
    --metadata_parquet sabdab_2025-05-06-paired_metadata.parquet \
    --annotations_parquet sabdab_2025-05-06-paired_annotations.parquet \
    --ncpu 16

python 2_extract_chains.py \
    --metadata_parquet sabdab_2025-05-06-paired_metadata.parquet \
    --annotations_parquet sabdab_2025-05-06-paired_annotations.parquet \
    --ncpu 16

python 3_assign_canonical_Kelow_cluster.py \
    --parquet_in sabdab_2025-05-06-paired_loops.parquet

python 3_assign_canonical_Kelow_cluster.py \
    --parquet_in sabdab_2025-05-06-paired_loops.parquet \
    --threshold 0.1

python 3_assign_canonical_Kelow_cluster.py \
    --parquet_in sabdab_2025-05-06-paired_loops.parquet \
    --threshold 0.61

python 4b_prepare_mmseq_cdr_only.py \
    --parquet_in sabdab_2025-05-06-paired_loops.parquet \
    --parquet_out sabdab_2025-05-06-paired_loops_with_sequence_id.parquet \
    --fasta_path sabdab_2025-05-06-paired_loops.fasta \
    --sequence_id_key 'sabdab_id'

fasta_file="sabdab_2025-05-06-paired_loops.fasta"
fasta_output_path="clustering/80"
TMP_DIR="/tmp"
MMSEQS_BIN="mmseqs/bin/mmseqs"
"$MMSEQS_BIN" easy-cluster "$fasta_file" "$fasta_output_path" "$TMP_DIR" --min-seq-id 0.8 -c 0.8 --cov-mode 1

python 5_split_data.py \
    --parquet_in sabdab_2025-05-06-paired_loops_with_sequence_id.parquet \
    --cluster_path "${fasta_output_path}_cluster.tsv" \
    --output_dir data/ \
    --seed 42
