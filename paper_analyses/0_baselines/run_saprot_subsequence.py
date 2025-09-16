import os
os.environ['HF_HOME']='saprot'
os.environ['HF_DATASETS_CACHE']='saprot/datasets'
os.environ['HF_MODELS_CACHE']='saprot/models'
os.environ['HF_DATASETS_DOWNLOADED_DATASETS_PATH']='saprot/datasets'
os.environ['HUGGINGFACE_HUB_CACHE']='saprot/hub'

from transformers import EsmTokenizer, EsmForMaskedLM
import argparse
import pandas as pd
import torch
import numpy as np
from tqdm import tqdm

def fasta_to_dict(path):
    """Return {record_id: sequence} from a FASTA file."""
    records = {}
    with open(path) as fh:
        seq_id = None
        seq_chunks = []
        for line in fh:
            line = line.strip()
            if not line:
                continue                     # skip blank lines
            if line.startswith(">"):
                # save previous record
                if seq_id is not None:
                    records[seq_id] = "".join(seq_chunks)
                # start new record
                seq_id = line[1:].split()[0] # header up to first space
                seq_chunks = []
            else:
                seq_chunks.append(line.upper())
        # don't forget the last record
        if seq_id is not None:
            records[seq_id] = "".join(seq_chunks)
    return records

model_path = "westlake-repl/SaProt_1.3B_AFDB_OMG_NCBI"
tokenizer = EsmTokenizer.from_pretrained(model_path)
model = EsmForMaskedLM.from_pretrained(model_path)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()

parser = argparse.ArgumentParser(description="Run SAPROT model for masked language modeling.")
parser.add_argument("--sequence_path", type=str, required=True, help="Path to the amino acid sequence file. The sequences and 3di sequences are a column in a .csv file.")
parser.add_argument("--output_path", type=str, required=True, help="Path to the output sequence embeddings. The sequences are saved as a .npy file.")
args = parser.parse_args()

saprot_sequences = []
sequence_length = []
starts = []
ends = []
sequences_aa = pd.read_csv(args.sequence_path)
for _, row in sequences_aa[['loop_id', 'sequence', '3di_sequence', 'start', 'end']].iterrows():
    seq_id = row['loop_id']
    sequence_aa = row['sequence']
    sequence_3di = row['3di_sequence']
    saprot_seq = "".join([f"{seq_aa.upper()}{seq_3di.lower()}" for seq_aa, seq_3di in zip(sequence_aa, sequence_3di)])
    saprot_sequences.append(saprot_seq)
    sequence_length.append(len(sequence_aa))
    starts.append(row['start'])
    ends.append(row['end'])

seq_embeddings = []
for batch_start in tqdm(range(0, len(saprot_sequences), 32), desc="SaProt Embedding", total=len(saprot_sequences)//32):
    batch_sequences = saprot_sequences[batch_start:batch_start + 32]
    inputs = tokenizer(batch_sequences, return_tensors="pt", padding=True, truncation=True, max_length=200)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits.cpu().detach().numpy()
    for seq_idx in range(batch_start, min(batch_start + 32, len(saprot_sequences))):
        seq_logits = logits[seq_idx - batch_start]
        seq_length = sequence_length[seq_idx]
        start = starts[seq_idx]
        end = ends[seq_idx]
        seq_logits = seq_logits[1+start:end+1, :]
        seq_mean = np.mean(seq_logits, axis=0)
        seq_embeddings.append(seq_mean)

seq_embeddings = np.array(seq_embeddings)
np.save(args.output_path, seq_embeddings)
print(f"Saved sequence embeddings to {args.output_path}, shape: {seq_embeddings.shape}")