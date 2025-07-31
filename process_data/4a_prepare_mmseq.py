from biotoolkit import load_structure, sequence, multiprocess
import pandas as pd
import argparse
import os

def process_one_cluster(canonical_cluster: str):
    cluster_sequences = []
    cluster_df = df[df.assigned_cluster == canonical_cluster].drop_duplicates(subset=['ab_fname'])
    chain_id = canonical_cluster[0]
    if chain_id not in ["H", "L"]:
        raise ValueError(f"Invalid chain ID: {chain_id}. Expected 'H' or 'L'.")
    for entry in cluster_df.itertuples():
        structure = load_structure(entry)
        structure = structure[structure.chain_id == chain_id]
        seq = sequence(structure)
        cluster_sequences.append((f'{entry.sabdab_id}_{chain_id}', seq))
    
    with open(os.path.join(args.fasta_dir, get_output_fname(canonical_cluster)), 'w') as f:
        for id, seq in cluster_sequences:
            f.write(f">{id}\n{seq}\n")
    print(f"Saved sequences for {canonical_cluster} to {args.fasta_dir}/{get_output_fname(canonical_cluster)}")

def get_output_fname(canonical_cluster: str) -> str:
    return canonical_cluster.replace("-*", "-x") + ".fasta"

if __name__ == "__main__":
    argparse = argparse.ArgumentParser()
    argparse.add_argument("--parquet_in", type=str, required=True)
    argparse.add_argument("--fasta_dir", type=str, required=True)
    argparse.add_argument("--ncpu", type=int, default=1)
    args = argparse.parse_args()

    os.makedirs(args.fasta_dir, exist_ok=True)

    df = pd.read_parquet(args.parquet_in)

    canonical_clusters = df.assigned_cluster.unique().tolist()
    multiprocess(
        process_one_cluster,
        canonical_clusters,
        ncpu=args.ncpu,
    )
