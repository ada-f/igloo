#!/bin/bash
#SBATCH --job-name=process_loops
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 16
#SBATCH --partition cpu
#SBATCH --error /data2/fanga5/logs/process_loops.%j.err
#SBATCH --output /data2/fanga5/logs/process_loops.%j.out
#SBATCH --time 01:00:00

export PDB_DIR="/data/fanga5/sabdab" 

/homefs/home/fanga5/micromamba/envs/pyenv/bin/python 0_process_annotate_sabdab_data.py \
    --parquet_in /data/fanga5/preprocessed_data/sabdab_2025-05-06-paired.parquet \
    --ncpu 16 --annotation_level "residue"

/homefs/home/fanga5/micromamba/envs/pyenv/bin/python 1_load_extract_loops.py \
    --metadata_parquet /data/fanga5/preprocessed_data/sabdab_2025-05-06-paired_metadata.parquet \
    --annotations_parquet /data/fanga5/preprocessed_data/sabdab_2025-05-06-paired_annotations.parquet \
    --ncpu 16

/homefs/home/fanga5/micromamba/envs/pyenv/bin/python 2_extract_chains.py \
    --metadata_parquet /data/fanga5/preprocessed_data/sabdab_2025-05-06-paired_metadata.parquet \
    --annotations_parquet /data/fanga5/preprocessed_data/sabdab_2025-05-06-paired_annotations.parquet \
    --ncpu 16

/homefs/home/fanga5/micromamba/envs/pyenv/bin/python 3_assign_canonical_Kelow_cluster.py \
    --parquet_in /data/fanga5/preprocessed_data/sabdab_2025-05-06-paired_loops.parquet

/homefs/home/fanga5/micromamba/envs/pyenv/bin/python 3_assign_canonical_Kelow_cluster.py \
    --parquet_in /data/fanga5/preprocessed_data/sabdab_2025-05-06-paired_loops.parquet \
    --threshold 0.1

/homefs/home/fanga5/micromamba/envs/pyenv/bin/python 3_assign_canonical_Kelow_cluster.py \
    --parquet_in /data/fanga5/preprocessed_data/sabdab_2025-05-06-paired_loops.parquet \
    --threshold 0.61

/homefs/home/fanga5/micromamba/envs/pyenv/bin/python 4b_prepare_mmseq_cdr_only.py \
    --parquet_in /data/fanga5/preprocessed_data/sabdab_2025-05-06-paired_loops.parquet \
    --parquet_out /data/fanga5/preprocessed_data/sabdab_2025-05-06-paired_loops_with_sequence_id.parquet \
    --fasta_path /data/fanga5/preprocessed_data/sabdab_2025-05-06-paired_loops.fasta \
    --sequence_id_key 'sabdab_id'

fasta_file="/data/fanga5/preprocessed_data/sabdab_2025-05-06-paired_loops.fasta"
fasta_output_path="/data/fanga5/preprocessed_data/clustering/80"
TMP_DIR="/tmp"
MMSEQS_BIN="/homefs/home/fanga5/mmseqs/bin/mmseqs"
"$MMSEQS_BIN" easy-cluster "$fasta_file" "$fasta_output_path" "$TMP_DIR" --min-seq-id 0.8 -c 0.8 --cov-mode 1

/homefs/home/fanga5/micromamba/envs/pyenv/bin/python 5_split_data.py \
    --parquet_in /data/fanga5/preprocessed_data/sabdab_2025-05-06-paired_loops_with_sequence_id.parquet \
    --cluster_path "${fasta_output_path}_cluster.tsv" \
    --output_dir /data/fanga5/data/ \
    --seed 42
