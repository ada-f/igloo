import os

os.environ["TORCH_HOME"] = "/data2/fanga5/baseline_models/esm2"

import torch
import esm
import pandas as pd
import argparse
import numpy as np

model, alphabet = esm.pretrained.esm2_t36_3B_UR50D()
batch_converter = alphabet.get_batch_converter()
model.eval() 

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)

parser = argparse.ArgumentParser(description='Run ablang2 model')
parser.add_argument('--sequence_path', type=str, required=True, help='Path to the dataset file with sequences')
parser.add_argument('--output_path', type=str, required=True, help='Path to the npy output file')
args = parser.parse_args()

dataset = pd.read_csv(args.sequence_path)
all_seqs = []
for i, row in dataset.iterrows():
    all_seqs.append((i, row['sequence']))

batch_labels, batch_strs, batch_tokens = batch_converter(all_seqs)
batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
starts = dataset['start'].values
ends = dataset['end'].values

batch_size = 32
sequence_representations = []
with torch.no_grad():
    for i in range(0, len(batch_tokens), batch_size):
        batch = batch_tokens[i:i + batch_size]
        batch = batch.to(device)
        results = model(batch, repr_layers=[33], return_contacts=True)
        token_representations = results["representations"][33]

        for j, (tokens_len, start_idx, end_idx) in enumerate(zip(batch_lens[i:i + batch_size], starts[i:i + batch_size], ends[i:i + batch_size])):
            sequence_representations.append(
                token_representations[j, 1 + start_idx : 1 + end_idx].mean(0).cpu().numpy()
            )

sequence_representations = np.array(sequence_representations)
np.save(args.output_path, sequence_representations)
print(f"Embeddings saved to {args.output_path}, shape: {sequence_representations.shape}")