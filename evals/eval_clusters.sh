#!/bin/bash
#SBATCH --job-name=calc_tm_score
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 128
#SBATCH --partition cpu
#SBATCH --error /data2/fanga5/logs/calc_tm_score.%j.err
#SBATCH --output /data2/fanga5/logs/calc_tm_score.%j.out
#SBATCH --time 06:00:00

# took 01:07:27 to run this
# /homefs/home/fanga5/micromamba/envs/pyenv/bin/python eval_clusters.py \
#     --num_workers 128 \
#     --cluster_file /data/fanga5/loop_Kelow_cluster_assignment.parquet \
#     --data_dir /data/fanga5/sabdab/ \
#     --output_dir /data/fanga5/Kelow_cluster_evals/tm_scores \
#     --cluster_key assigned_cluster

/homefs/home/fanga5/micromamba/envs/pyenv/bin/python eval_clusters.py \
    --num_workers 128 \
    --cluster_file /data2/fanga5/CDRCluster/version_26/clusters_epoch_380.parquet \
    --data_dir /data/fanga5/sabdab/ \
    --output_dir /data2/fanga5/CDRCluster/version_26/tm_scores \

