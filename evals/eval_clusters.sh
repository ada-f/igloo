#!/bin/bash
#SBATCH --job-name=calc_tm_score
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 128
#SBATCH --partition cpu
#SBATCH --error logs/calc_tm_score.%j.err
#SBATCH --output logs/calc_tm_score.%j.out
#SBATCH --time 06:00:00

# took 01:07:27 to run this
# python eval_clusters.py \
#     --num_workers 128 \
#     --cluster_file loop_Kelow_cluster_assignment.parquet \
#     --data_dir sabdab/ \
#     --output_dir Kelow_cluster_evals/tm_scores \
#     --cluster_key assigned_cluster

python eval_clusters.py \
    --num_workers 128 \
    --cluster_file Igloo/version_26/clusters_epoch_380.parquet \
    --data_dir sabdab/ \
    --output_dir Igloo/version_26/tm_scores \

