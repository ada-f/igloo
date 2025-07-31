# first process dihedrals into a parquet file with process_data/process_dihedrals.py
# then run this script to add dihedral angles to the OAS sequence data

import pandas as pd
import numpy as np
import argparse
from glob import glob
from tqdm import tqdm
import shutil

parser = argparse.ArgumentParser(description="Add dihedral angles to OAS sequence data")
parser.add_argument("--oas_data_path", type=str, required=True, help="Path to the OAS sequence data parquet file")
parser.add_argument("--dihedral_data_path", type=str, required=True, help="Path to the dihedral angles data parquet file, wildcard supported")
parser.add_argument("--output_path", type=str, required=True, help="Path to save the output parquet file with dihedral angles")
args = parser.parse_args()

df_sequences = pd.read_parquet(args.oas_data_path)

output_files = []
dihedral_files = sorted(glob(args.dihedral_data_path))
for dataset_i, dihedral_file in tqdm(enumerate(dihedral_files), total=len(dihedral_files), desc="Processing dihedral files"):
    df_loops = pd.read_parquet(dihedral_file)
    df_loops['chain'] = df_loops['loop_type'].str[0]
    df_loops['phi'] = df_loops['phi'].apply(lambda x: np.concatenate([np.zeros(1), x, np.zeros(36 - len(x) - 1)]))
    df_loops['psi'] = df_loops['psi'].apply(lambda x: np.concatenate([np.zeros(1), x, np.zeros(36 - len(x) - 1)]))
    df_loops['omega'] = df_loops['omega'].apply(lambda x: np.concatenate([np.zeros(1), x, np.zeros(36 - len(x) - 1)])) # start is cls, end is eos, padding zeros
    df_loops['angles'] = df_loops.apply(lambda row: np.stack([row['phi'], row['psi'], row['omega']], axis=1), axis=1) # shape (36, 3)

    agg_loops = (
        df_loops
            .groupby(["oas_id", "chain"], as_index=False)
            .agg(angles=("angles", lambda s: np.stack(s.to_numpy())))
    )

    agg_loops['angles'] = agg_loops['angles'].apply(lambda x: x.tolist())  # Convert numpy arrays to lists for parquet compatibility

    df = df_sequences.merge(agg_loops[agg_loops['chain'] == 'H'][['oas_id', 'angles']], left_on='seqid', right_on='oas_id', how='inner').drop(columns='oas_id').rename(columns={'angles': 'H_angles'})
    df = df.merge(agg_loops[agg_loops['chain'] == 'L'][['oas_id', 'angles']], left_on='seqid', right_on='oas_id', how='inner').drop(columns='oas_id').rename(columns={'angles': 'L_angles'})

    output_path = args.output_path.replace(".parquet", f"_{dataset_i}_of_{len(dihedral_files)}.parquet")
    df.to_parquet(output_path, index=False)
    print(f"Saved dihedral angles (N={len(df)}) to {output_path}")
    output_files.append(output_path)


final_df = []
for output_file in output_files:
    df = pd.read_parquet(output_file)
    final_df.append(df)
final_df = pd.concat(final_df, ignore_index=True)
final_df.to_parquet(args.output_path, index=False)
print(f"Saved combined dihedral angles to {args.output_path}, total entries: {len(final_df)}")

for output_file in output_files:
    shutil.rmtree(output_file, ignore_errors=True)  # Clean up individual files if needed


