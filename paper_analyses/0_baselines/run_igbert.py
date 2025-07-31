import pandas as pd
import os
os.environ['HF_HOME'] = "/data/fanga5/huggingface"
from transformers import BertModel, BertTokenizer
import torch
import numpy as np
import argparse
from tqdm import tqdm

tokeniser = BertTokenizer.from_pretrained("Exscientia/IgBert", do_lower_case=False)
model = BertModel.from_pretrained("Exscientia/IgBert", add_pooling_layer=False)
model.eval()

def parse_args():
    args = argparse.ArgumentParser(description="Run IgBert model on antibody sequences.")
    args.add_argument('--heavy_chain_key', type=str, default='heavy', help='Column name for heavy chain sequences.')
    args.add_argument('--light_chain_key', type=str, default='light', help='Column name for light chain sequences.')
    args.add_argument('--no_light_chain', action='store_true', help='If set, only process heavy chain sequences.')
    args.add_argument('--dataset_path', type=str, required=True, help='Path to the dataset CSV file.')
    args.add_argument('--output_path', type=str, required=True, help='Path to save the output embeddings as a .npy or .jsonl file.')
    args.add_argument('--subsequence', action='store_true', help='If set, keep embeddings of subsequences instead of full sequences.')
    return args.parse_args()

def get_sequence_embeddings(df, heavy_chain_key='heavy', light_chain_key='light', subsequence=False, no_light_chain=False):
    if no_light_chain:
        paired_sequences = df[heavy_chain_key].apply(lambda x: ' '.join(x))
        max_length = df[heavy_chain_key].str.len().max() + 2
    else:
        paired_sequences = df[light_chain_key].apply(lambda x: ' '.join(x)) + ' [SEP] ' + df[heavy_chain_key].apply(lambda x: ' '.join(x))
        max_length = df[light_chain_key].str.len().max() + df[heavy_chain_key].str.len().max() + 3  # +3 for [CLS], [SEP], [SEP]

    if subsequence:
        starts = df['start'].values
        ends = df['end'].values
    
    print("Sampled sequences:")
    print("\n".join(paired_sequences[:3]))

    print(f"Running IgBert on {len(paired_sequences)} sequences with max length {max_length}.")
    sequence_embeddings = []
    batch_size = 32
    for i in tqdm(range(0, len(paired_sequences), batch_size), desc="Running IgBert", total=len(paired_sequences)//batch_size):
        tokens = tokeniser.batch_encode_plus(
            paired_sequences[i:i+batch_size], 
            add_special_tokens=True, 
            max_length=max_length,
            return_tensors="pt",
            return_special_tokens_mask=True,
            padding='max_length',
        )

        output = model(
            input_ids=tokens['input_ids'], 
            attention_mask=tokens['attention_mask']
        )

        residue_embeddings = output.last_hidden_state
        residue_embeddings[tokens["special_tokens_mask"] == 1] = 0

        if subsequence:
            for j in range(i, min(i + batch_size, len(paired_sequences))):
                start = starts[j]
                end = ends[j]
                sequence_embedding = residue_embeddings[j - i, start+1:end + 1].detach().cpu().numpy()
                sequence_embedding = sequence_embedding.mean(axis=0)  # Average over the subsequence
                sequence_embedding = sequence_embedding.reshape(1, -1)  # Reshape to (1, embedding_dim)
                sequence_embeddings.append(sequence_embedding)
        else:
            sequence_embeddings_sum = residue_embeddings.sum(1)
            sequence_lengths = torch.sum(tokens["special_tokens_mask"] == 0, dim=1)
            sequence_embeddings_batch = sequence_embeddings_sum / sequence_lengths.unsqueeze(1)
            sequence_embeddings_batch = sequence_embeddings_batch.detach().cpu().numpy()
            sequence_embeddings.append(sequence_embeddings_batch)

    sequence_embeddings = np.concatenate(sequence_embeddings, axis=0)
    return sequence_embeddings


if __name__ == "__main__":
    args = parse_args()
    if args.output_path.endswith('.npy'):
        output_format = 'npy'
    elif args.output_path.endswith('.jsonl'):
        output_format = 'jsonl'
    else:
        raise ValueError("Output path must end with .npy or .jsonl")

    if args.no_light_chain:
        print("Processing only heavy chain sequences.")
    
    dataset = pd.read_csv(args.dataset_path) if args.dataset_path.endswith('.csv') else pd.read_parquet(args.dataset_path)
    sequence_embeddings = get_sequence_embeddings(dataset, heavy_chain_key=args.heavy_chain_key, light_chain_key=args.light_chain_key, subsequence=args.subsequence, no_light_chain=args.no_light_chain)
    if args.output_path.endswith('.npy'):
        np.save(args.output_path, sequence_embeddings)
    elif args.output_path.endswith('.jsonl'):
        dataset['sequence_embedding'] = sequence_embeddings.tolist()
        dataset[['sequence_id', 'sequence_embedding']].to_json(args.output_path, orient='records', lines=True)
    print(f"Embeddings saved to {args.output_path}")