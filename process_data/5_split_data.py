import argparse
import pandas as pd
import random
import os
import json

def get_data_entry(entry):
    return {
        "loop_id": f"{entry.sequence_id}_{entry.loop_type}",
        "loop_sequence": entry.loop_sequence,
        "phi": entry.phi.tolist(),
        "psi": entry.psi.tolist(),
        "omega": entry.omega.tolist(),
        "loop_c_alpha_atoms": [x.tolist() for x in entry.c_alpha_atoms.tolist()],
        "stem_c_alpha_atoms": [x.tolist() for x in entry.stem_c_alpha_atoms.tolist()],
    }

argparse = argparse.ArgumentParser()
argparse.add_argument("--parquet_in", type=str, required=True)
argparse.add_argument("--cluster_path", type=str, required=True)
argparse.add_argument("--output_dir", type=str, required=True)
argparse.add_argument("--loop_len", type=int, default=-1, help="Length of the loop to filter on")
argparse.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
argparse.add_argument("--sequence_id_key", type=str, default='sequence_id', help="Column name for sequence ID in the DataFrame that the loop originates from.")
args = argparse.parse_args()

random.seed(args.seed)

clusters_df = pd.read_csv(args.cluster_path, sep="\t", header=None, names=['rep_id', 'seq_id'])
df = pd.read_parquet(args.parquet_in)
df.rename(columns={args.sequence_id_key: 'sequence_id'}, inplace=True)

seq_to_rep = clusters_df.set_index('seq_id').to_dict()['rep_id']
df['rep_id'] = df['cdr_sequence_id'].map(seq_to_rep)

df['loop_len'] = df['loop_sequence'].str.len()
if args.loop_len > 0:
    df = df[df['loop_len']==args.loop_len]

all_reps = df['rep_id'].unique().tolist()
num_test_reps = len(all_reps) // 10
num_val_reps = num_test_reps
num_train_reps = len(all_reps) - num_test_reps - num_val_reps
random.shuffle(all_reps)
train_reps = all_reps[:num_train_reps]
val_reps = all_reps[num_train_reps:num_train_reps + num_val_reps]
test_reps = all_reps[num_train_reps + num_val_reps:]

train_df = df[df['rep_id'].isin(train_reps)]
val_df = df[df['rep_id'].isin(val_reps)]
test_df = df[df['rep_id'].isin(test_reps)]

train_seqs = set(train_df['loop_sequence'].unique().tolist())
val_seqs = set(val_df['loop_sequence'].unique().tolist())
test_seqs = set(test_df['loop_sequence'].unique().tolist())
print(f"Train sequences: {len(train_seqs)}")
print(f"Validation sequences: {len(val_seqs)}")
print(f"Test sequences: {len(test_seqs)}")
print(f"Train/val overlap: {len(train_seqs & val_seqs)}")
print(f"Train/test overlap: {len(train_seqs & test_seqs)}")
print(f"Val/test overlap: {len(val_seqs & test_seqs)}")

# Remove overlapping sequences from val and test sets
val_df = val_df[~val_df['loop_sequence'].isin(train_seqs)]
test_df = test_df[~test_df['loop_sequence'].isin(train_seqs) & ~test_df['loop_sequence'].isin(val_seqs)]
print(f"After removing overlaps, train sequences: {len(train_df['loop_sequence'].unique())}")
print(f"After removing overlaps, validation sequences: {len(val_df['loop_sequence'].unique())}")
print(f"After removing overlaps, test sequences: {len(test_df['loop_sequence'].unique())}")
print(f"Train/val overlap after filtering: {len(set(train_df['loop_sequence'].unique()) & set(val_df['loop_sequence'].unique()))}")
print(f"Train/test overlap after filtering: {len(set(train_df['loop_sequence'].unique()) & set(test_df['loop_sequence'].unique()))}")
print(f"Val/test overlap after filtering: {len(set(val_df['loop_sequence'].unique()) & set(test_df['loop_sequence'].unique()))}")

train_loop_types = train_df['loop_type'].value_counts()
val_loop_types = val_df['loop_type'].value_counts()
test_loop_types = test_df['loop_type'].value_counts()
loop_type_counts = pd.concat([train_loop_types, val_loop_types, test_loop_types], axis=1).fillna(0).astype(int)
loop_type_counts.columns = ['train', 'val', 'test']
print("Loop types:")
print(loop_type_counts)


train_canonical_clusters = train_df['assigned_cluster'].value_counts()
val_canonical_clusters = val_df['assigned_cluster'].value_counts()
test_canonical_clusters = test_df['assigned_cluster'].value_counts()
canonical_cluster_counts = pd.concat([train_canonical_clusters, val_canonical_clusters, test_canonical_clusters], axis=1).fillna(0).astype(int)
canonical_cluster_counts.columns = ['train', 'val', 'test']
print("Canonical clusters:")
print(canonical_cluster_counts)

train_data = [get_data_entry(entry) for entry in train_df.itertuples()]
val_data = [get_data_entry(entry) for entry in val_df.itertuples()]
test_data = [get_data_entry(entry) for entry in test_df.itertuples()]

if args.loop_len == -1:
    loop_len = "all"
else:
    loop_len = args.loop_len
train_path = os.path.join(args.output_dir, f"train_loop_len_{loop_len}_seed_{args.seed}.jsonl")
val_path = os.path.join(args.output_dir, f"val_loop_len_{loop_len}_seed_{args.seed}.jsonl")
test_path = os.path.join(args.output_dir, f"test_loop_len_{loop_len}_seed_{args.seed}.jsonl")

with open(train_path, 'w') as f:
    for entry in train_data:
        f.write(json.dumps(entry) + "\n")
with open(val_path, 'w') as f:
    for entry in val_data:
        f.write(json.dumps(entry) + "\n")
with open(test_path, 'w') as f:
    for entry in test_data:
        f.write(json.dumps(entry) + "\n")

print(f"Train data [N={len(train_data)}] saved to {train_path}")
print(f"Val data [N={len(val_data)}] saved to {val_path}")
print(f"Test data [N={len(test_data)}] saved to {test_path}")