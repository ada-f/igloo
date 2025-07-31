import ablang2
import argparse
import pandas as pd
import torch
import numpy as np

parser = argparse.ArgumentParser(description='Run ablang2 model')
parser.add_argument('--sequence_path', type=str, required=True, help='Path to the dataset file with sequences')
parser.add_argument('--output_path', type=str, required=True, help='Path to the npy output file')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

ablang = ablang2.pretrained(model_to_use='ablang2-paired', random_init=False, ncpu=1, device=device)
dataset = pd.read_csv(args.sequence_path)
all_seqs = []
for _, row in dataset.iterrows():
    all_seqs.append([row['fv_heavy'], row['fv_light']])

embeddings = []
batch_size = 64
with torch.no_grad():
    for i in range(0, len(all_seqs), batch_size):
        batch = all_seqs[i:i + batch_size]
        seqcoding = ablang(batch, mode='seqcoding')
        embeddings.append(seqcoding)
embeddings = np.concatenate(embeddings, axis=0)

np.save(args.output_path, embeddings)
print(f"Embeddings saved to {args.output_path}, shape: {embeddings.shape}")