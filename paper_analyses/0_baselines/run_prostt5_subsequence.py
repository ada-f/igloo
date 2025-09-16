
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 14:27:44 2023

@author: mheinzinger
"""

import os
os.environ['HF_HOME']='prostt5'
os.environ['HF_DATASETS_CACHE']='prostt5/datasets'
os.environ['HF_MODELS_CACHE']='prostt5/models'
os.environ['HF_DATASETS_DOWNLOADED_DATASETS_PATH']='prostt5/datasets'
os.environ['HUGGINGFACE_HUB_CACHE']='prostt5/hub'

import argparse
import time
from pathlib import Path
import torch
import h5py
from transformers import T5EncoderModel, T5Tokenizer
import pandas as pd
from tqdm import tqdm

if torch.cuda.is_available():
    device = torch.device('cuda:0')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')
print("Using device: {}".format(device))


def get_T5_model(model_dir):
    print("Loading T5 from: {}".format(model_dir))
    model = T5EncoderModel.from_pretrained(model_dir).to(device)
    model = model.eval()
    vocab = T5Tokenizer.from_pretrained(model_dir, do_lower_case=False )
    return model, vocab


def get_embeddings( seq_path, emb_path, model_dir, 
                       per_protein, half_precision, is_3Di,
                       max_residues=4000, max_seq_len=1000, max_batch=100 ):
    
    emb_dict = dict()

    # Read in fasta
    seq_df = pd.read_csv(seq_path)
    print("Read {} sequences from {}".format(len(seq_df), seq_path))
    prefix = "<fold2AA>" if is_3Di else "<AA2fold>"
    
    model, vocab = get_T5_model(model_dir)
    if half_precision:
        model = model.half()
        print("Using model in half-precision!")

    print('########################################')
    print(f"Input is 3Di: {is_3Di}")

    sequence_key = '3di_sequence' if is_3Di else 'sequence'

    print(f"Example sequence: {seq_df[sequence_key].iloc[0]}")
    print('########################################')
    print('Total number of sequences: {}'.format(len(seq_df)))

    start = time.time()
    batch = list()
    for seq_idx, row in tqdm(seq_df.iterrows(), total=seq_df.shape[0]):
        pdb_id = row['loop_id']
        seq = row[sequence_key].lower() if is_3Di else row[sequence_key].upper()
        # replace non-standard AAs
        seq = seq.replace('U','X').replace('Z','X').replace('O','X')
        seq_len = len(seq)
        seq = prefix + ' ' + ' '.join(list(seq))
        batch.append((pdb_id,seq,seq_len,row['start'],row['end']))

        # count residues in current batch and add the last sequence length to
        # avoid that batches with (n_res_batch > max_residues) get processed 
        n_res_batch = sum([ s_len for  _, _, s_len, _, _ in batch ]) + seq_len 
        if len(batch) >= max_batch or n_res_batch>=max_residues or seq_idx==len(seq_df)-1 or seq_len>max_seq_len:
            pdb_ids, seqs, seq_lens, starts, ends = zip(*batch)
            batch = list()

            token_encoding = vocab.batch_encode_plus(seqs, 
                                                     add_special_tokens=True, 
                                                     padding="longest", 
                                                     return_tensors='pt' 
                                                     ).to(device)
            try:
                with torch.no_grad():
                    embedding_repr = model(token_encoding.input_ids, 
                                           attention_mask=token_encoding.attention_mask
                                           )
            except RuntimeError:
                print("RuntimeError during embedding for {} (L={})".format(
                    pdb_id, seq_len)
                    )
                continue
            
            # batch-size x seq_len x embedding_dim
            # extra token is added at the end of the seq
            for batch_idx, identifier in enumerate(pdb_ids):
                s_len = seq_lens[batch_idx]
                start_idx = starts[batch_idx]
                end_idx = ends[batch_idx]
                # account for prefix in offset
                emb = embedding_repr.last_hidden_state[batch_idx,1+start_idx:1+end_idx]
                
                if per_protein:
                    emb = emb.mean(dim=0)
                emb_dict[ identifier ] = emb.detach().cpu().numpy().squeeze()
                if len(emb_dict) == 1:
                    print("Example: embedded protein {} with length {} to emb. of shape: {}".format(
                                identifier, s_len, emb.shape))

    end = time.time()
    
    with h5py.File(str(emb_path), "w") as hf:
        for sequence_id, embedding in emb_dict.items():
            # noinspection PyUnboundLocalVariable
            hf.create_dataset(str(sequence_id), data=embedding)

    print('\n############# STATS #############')
    print('Total number of embeddings: {}'.format(len(emb_dict)))
    return True


def create_arg_parser():
    """"Creates and returns the ArgumentParser object."""

    # Instantiate the parser
    parser = argparse.ArgumentParser(description=( 
            'embed.py creates ProstT5-Encoder embeddings for a given text '+
            ' file containing sequence(s) in a CSV file.' +
            'Example: python embed.py --input /path/to/some_sequences.csv --output /path/to/some_embeddings.h5 --half 1 --is_3Di 0 --per_protein 1' ) )
    
    # Required positional argument
    parser.add_argument( '-i', '--input', required=True, type=str,
                    help='A path to a CSV file containing protein sequence(s).')

    # Optional positional argument
    parser.add_argument( '-o', '--output', required=True, type=str, 
                    help='A path for saving the created embeddings as NumPy npz file.')

    
    # Required positional argument
    parser.add_argument('--model', required=False, type=str,
                    default="Rostlab/ProstT5",
                    help='Either a path to a directory holding the checkpoint for a pre-trained model or a huggingface repository link.' )

    # Optional argument
    parser.add_argument('--per_protein', type=int, 
                    default=0,
                    help="Whether to return per-residue embeddings (0: default) or the mean-pooled per-protein representation (1).")
        
    parser.add_argument('--half', type=int, 
                    default=0,
                    help="Whether to use half_precision or not. Default: 0 (full-precision)")
    
    parser.add_argument('--is_3Di', type=int, 
                    default=0,
                    help="Whether to create embeddings for 3Di or AA file. Default: 0 (generate AA-embeddings)")
    
    return parser

def main():
    parser     = create_arg_parser()
    args       = parser.parse_args()
    
    seq_path   = Path( args.input ) # path to input FASTAS
    emb_path   = Path( args.output) # path where embeddings should be stored
    model_dir  = args.model # path/repo_link to checkpoint
    
    per_protein    = False if int(args.per_protein) == 0 else True
    half_precision = False if int(args.half)        == 0 else True
    is_3Di         = False if int(args.is_3Di)      == 0 else True


    get_embeddings( 
        seq_path, 
        emb_path, 
        model_dir, 
        per_protein=per_protein,
        half_precision=half_precision, 
        is_3Di=is_3Di 
        )


if __name__ == '__main__':
    main()