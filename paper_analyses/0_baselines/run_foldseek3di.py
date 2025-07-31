import pandas as pd
import mini3di
from mini3di.utils import last
from Bio.PDB import Chain
import Bio.PDB as PDB
import Bio
import torch
import numpy as np
import functools
from typing import Literal
from tqdm import tqdm
import argparse

def parse_args():
    args = argparse.ArgumentParser(description="Run IgBert model on antibody sequences.")
    args.add_argument('--dataset_path', type=str, required=True, help='Path to the dataset CSV file.')
    args.add_argument('--output_path', type=str, required=True, help='Path to save the output embeddings as a .npy')
    args.add_argument('--chain_id', type=str, required=True, help='Chain ID to process (e.g., "H", "L").')
    return args.parse_args()


class WrappedFoldSeekTokenizer():
    # Source https://github.com/KatarinaYuan/StructTokenBench/blob/cd4cfe5026dc0e6919a8116367ec2d129afabd29/src/tokenizer.py

    FOLDSEEK_STRUC_VOCAB = "ACDEFGHIKLMNPQRSTVWYX"
    # reference to https://github.com/althonos/mini3di/blob/faeff98f8c411224fadb73e3878ef6c8ceefa887/mini3di/encoder.py#L27

    def __init__(self, device:  torch.device | str = "cpu"):
        self.device = device

        self.token2id = {s:i for i, s in enumerate(self.FOLDSEEK_STRUC_VOCAB)}
        self.pad_token_id = len(self.FOLDSEEK_STRUC_VOCAB)

        # third party implementation: https://github.com/althonos/mini3di/
        self.tokenizer_encoder = mini3di.Encoder() # based on numpy, not torch tensors
    
    def get_num_tokens(self):
        return len(self.FOLDSEEK_STRUC_VOCAB) + 1 # additional PAD token

    def get_codebook_embedding(self,):        
        return torch.tensor(self.tokenizer_encoder._CENTROIDS) # (20, 2)

    def _encode_structure_mini3di(self, pdb_path, chain_id, use_continuous=False, use_sequence=False):
        if pdb_path.endswith(".pdb"):
            parser = PDB.PDBParser(QUIET=True)
        else:
            parser = PDB.MMCIFParser(QUIET=True)
        structure = parser.get_structure("test", pdb_path)
        chain = structure[0][chain_id]
        if not use_continuous:
            states = self.tokenizer_encoder.encode_chain(chain)
            seq_mini3di = self.tokenizer_encoder.build_sequence(states)
        else:
            seq_mini3di = self.hijack_continuous_reprs(chain)
        residues = [residue for residue in chain.get_residues() if "CA" in residue] # aligned with FoldSeek's internal implementation
        residue_index = np.array([_.get_id()[1] for _ in residues])

        seqs = [Bio.PDB.Polypeptide.three_to_index(_.resname) 
                if Bio.PDB.Polypeptide.is_aa(_.resname, standard=True) else 20 for _ in residues] # 20 for unknown AA

        return seq_mini3di, residue_index, seqs
    
    def hijack_continuous_reprs(self, 
        chain: Chain,
        ca_residue: bool = True,
        disordered_atom: Literal["best", "last"] = "best",
    ):
        """
        Adapted from https://github.com/althonos/mini3di/blob/5bc2fb0257e8d743326f74615ee2c1820c66e7c1/mini3di/encoder.py#L60
        """
        # extract residues
        if ca_residue:
            residues = [residue for residue in chain.get_residues() if "CA" in residue]
        else:
            residues = list(chain.get_residues())
        # extract atom coordinates
        r = len(residues)
        ca = np.array(np.nan, dtype=np.float32).repeat(3 * r).reshape(r, 3)
        cb = ca.copy()
        n = ca.copy()
        c = ca.copy()
        for i, residue in enumerate(residues):
            for atom in residue.get_atoms():
                if atom.is_disordered() and disordered_atom == "last":
                    atom = last(atom)
                if atom.get_name() == "CA":
                    ca[i, :] = atom.coord
                elif atom.get_name() == "N":
                    n[i, :] = atom.coord
                elif atom.get_name() == "C":
                    c[i, :] = atom.coord
                elif atom.get_name() == "CB" or atom.get_name() == "CB A":
                    cb[i, :] = atom.coord
        # encode coordinates
        descriptors = self.tokenizer_encoder.feature_encoder.encode_atoms(ca, cb, n, c)
        
        # the last layer is for discretization
        reprs = functools.reduce(lambda x, f: f(x), self.tokenizer_encoder.vae_encoder.layers[:-1], descriptors.data)

        return reprs

    def encode_structure(self, pdb_path, chain_id, use_continuous=False):
        seq_mini3di, residue_index, seqs = self._encode_structure_mini3di(pdb_path, chain_id, use_continuous)
        if not use_continuous:
            structural_tokens = torch.LongTensor([self.token2id[x] for x in seq_mini3di])
        else:
            structural_tokens = torch.tensor(seq_mini3di, device=self.device)

        # sometimes, seq_mini3di does match the length of pdb_chain from ESM3
        # because mini3di filter residues without CA

        return structural_tokens, residue_index, seqs

if __name__ == "__main__":
    args = parse_args()
    DATA_DIR = "/data/fanga5/sabdab/"

    encoder = WrappedFoldSeekTokenizer(device="cuda" if torch.cuda.is_available() else "cpu")

    raw_df = pd.read_parquet("/data/fanga5/preprocessed_data/sabdab_2025-05-06-paired.parquet")
    raw_df['sabdab_id'] = range(len(raw_df))
    ab_fname_to_id = {fname: sabdab_id for fname, sabdab_id in zip(raw_df['ab_fname'], raw_df['sabdab_id'])}
    id_to_ab_fname = {sabdab_id: fname for fname, sabdab_id in zip(raw_df['ab_fname'], raw_df['sabdab_id'])}

    loop_df = pd.read_csv(args.dataset_path)
    max_length = (loop_df['end']-loop_df['start']).max()

    embeddings = []
    for _, row in tqdm(loop_df.iterrows(), desc="Encoding loops", total=len(loop_df)):
        loop_id = row['loop_id']
        ab_fname = id_to_ab_fname[loop_id]
        start = row['start']
        end = row['end']
        # structural_tokens, residue_index, seqs = encoder.encode_structure(DATA_DIR + ab_fname, args.chain_id, use_continuous=False)
        continuous_structural_tokens, residue_index, seqs = encoder.encode_structure(DATA_DIR + ab_fname, args.chain_id, use_continuous=True)
        embedding_3di = continuous_structural_tokens[start:end].flatten().detach().cpu().numpy()
        embedding_3di = np.pad(embedding_3di, (0, max_length*2 - len(embedding_3di)), mode='constant', constant_values=0)
        embeddings.append(embedding_3di)
    
    embeddings = np.array(embeddings)
    np.save(args.output_path, embeddings)

