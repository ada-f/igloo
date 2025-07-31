from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig
import pandas as pd
import numpy as np
import argparse
from tqdm import tqdm

client = ESMC.from_pretrained("esmc_300m").to("cuda") # or "cpu"

def inference(sequence, start_index=1, end_index=-1):
    protein = ESMProtein(sequence=sequence)
    protein_tensor = client.encode(protein)
    logits_output = client.logits(
        protein_tensor, LogitsConfig(sequence=True, return_embeddings=True)
    )
    return logits_output.embeddings[0, start_index:end_index].mean(axis=0).detach().cpu().numpy()

def main(df_path, sequence_column, output_path, start_column=None, end_column=None):
    if df_path.endswith('.parquet'):
        df = pd.read_parquet(df_path)
    elif df_path.endswith('.csv'):
        df = pd.read_csv(df_path)
    else:
        raise ValueError("Input file must be a CSV or parquet file.")
    sequences = df[sequence_column].tolist()
    if start_column is not None and end_column is not None:
        start_indices = (df[start_column] + 1).tolist()  # +1 for bos token
        end_indices = (df[end_column] + 1).tolist() # +1 for eos token
    else:
        start_indices = [1] * len(sequences)  # Default start index for bos token
        end_indices = [-1] * len(sequences)  # Default end index for eos token
    
    results = []
    for sequence, start, end in tqdm(zip(sequences, start_indices, end_indices), desc="ESM-C Inference", total=len(sequences)):
        results.append(inference(sequence, start, end))
    results = np.stack(results, axis=0)

    np.save(output_path, results)
    print(f"Saved embeddings (shape {results.shape}) to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ESMC inference on sequences.")
    parser.add_argument("--df_path", type=str, required=True, help="Path to the input DataFrame CSV or parquet file.")
    parser.add_argument("--sequence_column", type=str, required=True, help="Column name containing sequences.")
    parser.add_argument("--sequence_idx_start_column", type=str, default=None, help="Column name containing sequence start indices.")
    parser.add_argument("--sequence_idx_end_column", type=str, default=None, help="Column name containing sequence end indices.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the output embeddings as a .npy file.")
    args = parser.parse_args()
    main(args.df_path, args.sequence_column, args.output_path, start_column=args.sequence_idx_start_column, end_column=args.sequence_idx_end_column)