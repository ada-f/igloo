import argparse
import pandas as pd

argparse = argparse.ArgumentParser()
argparse.add_argument("--parquet_in", type=str, nargs='+', required=True)
argparse.add_argument("--sequence_id_key", type=str, nargs='+', help="Column name for sequence ID in the DataFrame that the loop originates from.")
argparse.add_argument("--parquet_out", type=str, required=True)
argparse.add_argument("--fasta_path", type=str, required=True)
args = argparse.parse_args()

keep_cols = ['loop_type', 'sequence_id', 'loop_sequence', 'c_alpha_atoms', 'stem_c_alpha_atoms', 'phi', 'psi', 'omega',
            'assigned_cluster', 'min_dih_dist', 'assigned_cluster_D=0.1', 'min_dih_dist_D=0.1']
if len(args.parquet_in) > 1:
    assert len(args.sequence_id_key) == len(args.parquet_in), "Number of sequence ID keys must match number of input parquet files"
    dfs = []
    for i, parquet_file in enumerate(args.parquet_in):
        df = pd.read_parquet(parquet_file)
        df.rename(columns={args.sequence_id_key[i]: 'sequence_id'}, inplace=True)
        df.astype({'sequence_id': str}, errors='ignore')
        dfs.append(df[keep_cols])
    df = pd.concat(dfs, ignore_index=True)
else:
    assert len(args.sequence_id_key) == 1, "If only one parquet file is provided, only one sequence ID key should be specified"
    args.sequence_id_key = args.sequence_id_key[0]
    df = pd.read_parquet(args.parquet_in)
    df.rename(columns={args.sequence_id_key: 'sequence_id'}, inplace=True)

df.sort_values(by=['sequence_id', 'loop_type'], inplace=True)
df['chain'] = df['loop_type'].str[0]
df_agg_sabdab = df.groupby(['sequence_id', 'chain']).agg({'loop_sequence': 'sum', 'loop_type': list}).reset_index().rename(
    columns={'loop_sequence': 'cdr_sequence', 'loop_type': 'cdr_type'})
unique_sequences = df_agg_sabdab['cdr_sequence'].unique().tolist()
unique_sequences_to_id = {seq: i for i, seq in enumerate(unique_sequences)}
df_agg_sabdab['cdr_sequence_id'] = df_agg_sabdab['cdr_sequence'].map(unique_sequences_to_id)
sabdab_id_to_seq_id = df_agg_sabdab.set_index(['sequence_id', 'chain'])['cdr_sequence_id'].to_dict()
df['cdr_sequence_id'] = df.set_index(['sequence_id', 'chain']).index.map(sabdab_id_to_seq_id)

with open(args.fasta_path, 'w') as f:
    for seq, id in unique_sequences_to_id.items():
        f.write(f">{id}\n{seq}\n")

print(f"Saved unique CDR sequences to {args.fasta_path}")
df['sequence_id'] = df['sequence_id'].astype(str)
df.to_parquet(args.parquet_out, index=False)
