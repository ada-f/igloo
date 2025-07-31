import ablang2
import argparse
import pandas as pd
import torch
import numpy as np
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Run ablang2 model')
parser.add_argument('--sequence_path', type=str, required=True, help='Path to the dataset file with sequences')
parser.add_argument('--output_path', type=str, required=True, help='Path to the npy output file')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

ablang = ablang2.pretrained(model_to_use='ablang2-paired', random_init=False, ncpu=1, device=device)
dataset = pd.read_csv(args.sequence_path)
all_seqs = []
for _, row in dataset.iterrows():
    all_seqs.append([row['sequence'], ""]) # sequence is heavy chain, light chain is empty

embeddings = []
batch_size = 64
with torch.no_grad():
    for i in tqdm(range(0, len(all_seqs), batch_size), desc="Processing batches", total=len(all_seqs) // batch_size):
        batch = all_seqs[i:i + batch_size]
        rescoding = ablang(batch, mode='rescoding')
        for j in range(i, min(i + batch_size, len(all_seqs))):
            start_idx = dataset['start'].iloc[j]
            end_idx = dataset['end'].iloc[j]
            seqcoding = rescoding[j - i][1 + start_idx : 1 + end_idx].mean(0)
            embeddings.append(seqcoding)
embeddings = np.array(embeddings)

np.save(args.output_path, embeddings)
print(f"Embeddings saved to {args.output_path}, shape: {embeddings.shape}")