import os
import time
import joblib
import pickle
import urllib.request
import functools
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.distributed as dist

import Bio
from Bio import PDB
from biotite.structure.io.pdbx import CIFFile, convert
import numpy as np
from Bio.PDB import Chain
from typing import Literal

# ----- MIF Loading ------- #
import sys
sys.path.append("/data2/fanga5/StructTokenBench/src/baselines/protein-sequence-models")
from sequence_models.pretrained import load_model_and_alphabet as mif_load_model_and_alphabet
from sequence_models.pdb_utils import parse_PDB as mif_parse_PDB
from sequence_models.pdb_utils import process_coords as mif_process_coords

# ----- ProTokens Loading ------- #
import sys
sys.path.append("/data2/fanga5/StructTokenBench/src/baselines/ProToken")
from data_process.preprocess import save_pdb_from_aux, protoken_encoder_preprocess, protoken_decoder_preprocess, init_protoken_model
from data_process.preprocess import protoken_encoder_input_features, protoken_decoder_input_features
from data_process import residue_constants

# ----- AIDO Loading ------- #
import sys
sys.path.append("/data2/fanga5/StructTokenBench/src/baselines/ModelGenerator")
try:
    from modelgenerator.structure_tokenizer.models import EquiformerEncoderLightning
    from modelgenerator.structure_tokenizer.datasets.protein_dataset import ProteinDataset
    from modelgenerator.structure_tokenizer.datasets.protein import Protein
except ModuleNotFoundError:
    print("[Warining]: AIDO.st not found")

# ----- Cheap Loading ------- #
import sys
sys.path.append("/data2/fanga5/StructTokenBench/src/baselines")
try:
    # load from Cheap
    from cheap_proteins.src.cheap.pretrained import (
        CHEAP_shorten_1_dim_64
    )
except ModuleNotFoundError:
    print("[Warining]: cheap not found")
    pass


# ----- ProteinMPNN Loading ------- #
try:
    from baselines.protein_mpnn_utils import ProteinMPNN, tied_featurize, parse_PDB_biounits, parse_PDB
    from baselines.protein_mpnn_cif_parser import parse_cif_pdb_biounits
except ModuleNotFoundError:
    print("[Warining]: ProteinMPNN not found")



SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))

# Reside Name Mapping:
# {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "ASX": "B", "CYS": "C", "GLN": "Q", 
# "GLU": "E", "GLX": "Z", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", 
# "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", 
# "VAL": "V", "KCX": "K"}

# All tokenizers should be registered here
# discretized tokenizers can produce both discretized and continuous strutcural reprs.
# while continuous tokenizers can ONLY produce continuous structural reprs.
ALL_TOKENIZER_TYPE = {
    "discretized": [
        "WrappedProTokensTokenizer",
        "WrappedAIDOTokenizer",
    ],
    "continuous": [
        "WrappedMIFTokenizer",
        "WrappedProteinMPNNTokenizer",
        "WrappedCheapS1D64Tokenizer"
    ]
}

import argparse

def parse_args():
    args = argparse.ArgumentParser(description="Run IgBert model on antibody sequences.")
    args.add_argument('--model_type', type=str, choices=['ProteinMPNN', 'MIF'])
    args.add_argument('--dataset_path', type=str, required=True, help='Path to the dataset CSV file.')
    args.add_argument('--output_path', type=str, required=True, help='Path to save the output embeddings as a .npy')
    args.add_argument('--chain_id', type=str, required=True, help='Chain ID to process (e.g., "H", "L").')
    return args.parse_args()


class WrappedProTokensTokenizer():

    """
    Adapted from https://colab.research.google.com/drive/15bBbfa7WigruoME089cSfE242K1MvRGz
    """

    def __init__(self, device=None):
        self.device = device

        dir_name =  os.path.join(os.path.dirname(__file__), "baselines/ProToken")
        self.tokenizer_encoder_1 = init_protoken_model(512, dir_name)
        self.tokenizer_encoder_2 = init_protoken_model(1024, dir_name)

        codebook_file_name = os.path.join(dir_name, "ProToken_Code_Book.pkl")
        self.codebook_embedding = pickle.load(open(codebook_file_name, "rb"))
        self.pad_token_id = len(self.codebook_embedding)
    
    def get_codebook_embedding(self,):
        return torch.tensor(self.codebook_embedding) # (512, 32)
    
    def get_num_tokens(self, ):
        return len(self.codebook_embedding) + 1
    
    def encode_structure(self, pdb_path, chain_id, use_continuous=False, use_sequence=False):
        assert use_sequence
        encoder_inputs, encoder_aux, seq_len = protoken_encoder_preprocess(pdb_path, task_mode="single", chain_id=chain_id)

        if seq_len <= 512:
            encoder_results = self.tokenizer_encoder_1.encoder(*encoder_inputs)
        elif seq_len <= 1024:
            encoder_results = self.tokenizer_encoder_2.encoder(*encoder_inputs)
        else:
            raise NotImplementedError

        structural_tokens, residue_index, seqs = [], [], []
        for p in range(encoder_aux['seq_mask'].shape[0]):
            if (encoder_aux['seq_mask'][p] 
            and encoder_aux["aatype"][p] < len(residue_constants.restype_order)): # < 20, may be unknown residues like HOH
                structural_tokens.append(encoder_results["protoken_index"][p])
                residue_index.append(encoder_aux["residue_index"][p])
                seqs.append(encoder_aux["aatype"][p])
        
        # tf.Tensor -> np.array -> torch.Tensor
        structural_tokens = torch.tensor(np.asarray(structural_tokens), device=self.device)
        residue_index = np.asarray(residue_index)

        assert (structural_tokens < 512).all()
        return structural_tokens, residue_index, seqs

class WrappedAIDOTokenizer():
    def __init__(self, device=None):
        self.device = device
        self.tokenizer_encoder = EquiformerEncoderLightning("genbio-ai/AIDO.StructureEncoder").to(self.device)
        self.tokenizer_encoder.training = False
        self.codebook_embedding = self.tokenizer_encoder.encoder.codebook.data.cpu()
        self.pad_token_id = len(self.codebook_embedding)
    
    def get_codebook_embedding(self,):
        return self.codebook_embedding # (512, 384)
    
    def get_num_tokens(self, ):
        return len(self.codebook_embedding) + 1
    
    def encode_structure(self, pdb_path, chain_id, use_continuous=False, use_sequence=False):
        # parse the pdb_file into a Protein object
        if pdb_path.endswith(".pdb"):
            protein = Protein.from_pdb_file_path(pdb_path, chain_id)
        elif pdb_path.endswith(".cif"):
            # 2nd arg (entity_id) will be ignored to ensure the same CIF 
            # parsing logic within this codebase
            protein = Protein.from_cif_file_path(pdb_path, 1, chain_id)
        # Do not implement the cropping logic for simplicity 
        # max_nb_res=1024 by default
        # input_crop = ProteinDataset.protein_to_input_crop(protein)
        protein_input = protein.to_torch_input() # a dict of tensors
        protein_batch = ProteinDataset.collate_fn([protein_input])

        output = self.tokenizer_encoder.forward(
            protein_batch["atom_positions"].to(self.device),
            protein_batch["attention_mask"].to(self.device),
            protein_batch["residue_index"].to(self.device),
        )
        structural_tokens = output["idx"][0]
        residue_index = protein_batch["residue_index"][0]
        seqs = protein_input["aatype"]
        return structural_tokens, residue_index, seqs

class WrappedCheapBaseTokenizer():

    def __init__(self, device: torch.device | str = "cpu"):
        self.device = device
        self.pad_token_id = 0 # for pipeline compatibility
    
    def get_num_tokens(self):
        return None

    def get_codebook_embedding(self,):
        return None
    
    def encode_structure(self, pdb_path: str, chain_id: str, use_sequence=False):
        assert use_sequence

        pdb_dict_list = parse_PDB(
            pdb_path, input_chain_list=[chain_id], ca_only=False,
            parse_fn=parse_cif_pdb_biounits
        )
        sequences = [
            pdb_dict_list[0]["seq"]
        ]
        rep, mask = self.pipeline(sequences) # [bsz=1, L, 64], [bsz=1, L]
        seqs = [Bio.PDB.Polypeptide.one_to_index(x) if x != "X" else 20 for x in pdb_dict_list[0]["seq"]] # total 20 standard AA in Bio

        return rep.squeeze(0), pdb_dict_list[0]["ridx"], seqs # [L, dim], [L]

class WrappedCheapS1D64Tokenizer(WrappedCheapBaseTokenizer):
    def __init__(self, device):
        self.pipeline = CHEAP_shorten_1_dim_64(return_pipeline=True)
        super().__init__(device)


class WrappedMIFTokenizer():

    def __init__(self, device: torch.device | str = "cpu"):
        self.device = device
        self.model, self.mif_collater = mif_load_model_and_alphabet('mif')
        self.pad_token_id = 0 # for pipeline compatibility
    
    def get_num_tokens(self):
        return None

    def get_codebook_embedding(self,):
        return None

    def encode_structure(self, pdb_path: str, chain_id: str, use_sequence=False):
        assert use_sequence

        pdb_dict_list = parse_PDB(
            pdb_path, input_chain_list=[chain_id], ca_only=False,
            parse_fn=parse_cif_pdb_biounits
        )
        coords = {
            'N': np.array(pdb_dict_list[0][f"coords_chain_{chain_id}"][f"N_chain_{chain_id}"]),
            'CA': np.array(pdb_dict_list[0][f"coords_chain_{chain_id}"][f"CA_chain_{chain_id}"]),
            'C': np.array(pdb_dict_list[0][f"coords_chain_{chain_id}"][f"C_chain_{chain_id}"])
        }
        dist, omega, theta, phi = mif_process_coords(coords)
        batch = [[pdb_dict_list[0][f"seq_chain_{chain_id}"], torch.tensor(dist, dtype=torch.float),
              torch.tensor(omega, dtype=torch.float),
              torch.tensor(theta, dtype=torch.float), torch.tensor(phi, dtype=torch.float)]]
        src, nodes, edges, connections, edge_mask = self.mif_collater(batch)
        rep = self.model(src, nodes, edges, connections, edge_mask)

        assert len(pdb_dict_list[0]["seq"]) == len(rep[0])
        seqs = [Bio.PDB.Polypeptide.one_to_index(x) if x != "X" else 20 for x in pdb_dict_list[0]["seq"]] # total 20 standard AA in Bio

        return rep.squeeze(0), pdb_dict_list[0]["ridx"], seqs # [L, dim], [L]

class WrappedProteinMPNNTokenizer():
    HIDDEN_DIM = 128
    NUM_LAYERS = 3
    MODEL_CKPT_BASEURL = "https://github.com/dauparas/ProteinMPNN/raw/refs/heads/main/"
    CKPT_CACHE_DIR = os.path.join(SCRIPT_PATH, 'ProteinMPNN')
    CA_ONLY = False

    def __init__(
            self, 
            device: torch.device | str = "cpu", 
            checkpoint_path: str = "vanilla_model_weights/v_48_020.pt"):
        self.device = device
        # init model and load model weights
        local_checkpoint_path = self._download_model_checkpoint(checkpoint_path)
        self._load_protein_mpnn_model(local_checkpoint_path)

        self.pad_token_id = 0 # for pipeline compatibility

    def _download_model_checkpoint(self, checkpoint_path):
        """Download ProteinMPNN checkpoint from GitHub if not locally cached."""
        ckpt_url = self.MODEL_CKPT_BASEURL + checkpoint_path
        cached_checkpoint_path = os.path.join(self.CKPT_CACHE_DIR, checkpoint_path)
        os.makedirs(os.path.dirname(cached_checkpoint_path), exist_ok=True)
        if not os.path.isfile(cached_checkpoint_path):
            urllib.request.urlretrieve(ckpt_url, cached_checkpoint_path)
        return cached_checkpoint_path

    def _load_protein_mpnn_model(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        model = ProteinMPNN(
            ca_only=self.CA_ONLY, 
            num_letters=21, 
            node_features=self.HIDDEN_DIM, 
            edge_features=self.HIDDEN_DIM, 
            hidden_dim=self.HIDDEN_DIM, 
            num_encoder_layers=self.NUM_LAYERS, 
            num_decoder_layers=self.NUM_LAYERS, 
            augment_eps=0.0, 
            k_neighbors=checkpoint['num_edges'])
        model.load_state_dict(checkpoint['model_state_dict'])
        self.model = model.to(self.device)

    def get_num_tokens(self):
        return None

    def get_codebook_embedding(self,):
        return None

    def encode_structure(self, pdb_path: str, chain_id: str, use_sequence=False):
        assert use_sequence
        
        parse_fn = parse_cif_pdb_biounits
        pdb_dict_list = parse_PDB(
            pdb_path, input_chain_list=[chain_id], ca_only=self.CA_ONLY,
            parse_fn=parse_fn
            )

        # this function comes from ProteinMPNN:
        X, _, mask, _, _, chain_encoding_all, _, _, _, _, _, _, residue_idx, _, _, _, _, _, _, _ = tied_featurize(
            pdb_dict_list, 
            self.device, None, None, None, None, None, None, 
            ca_only=self.CA_ONLY)

        h_V, h_E = self.model.encode(X, mask, residue_idx, chain_encoding_all)
        assert len(pdb_dict_list[0]["ridx"]) == len(h_V[0])
        # h_V: [1, L, hidden_dim]
        # h_E: [1, L, n_edges, hidden_dim] 

        assert len(pdb_dict_list[0]["seq"]) == len(h_V[0])
        seqs = [Bio.PDB.Polypeptide.one_to_index(x) if x != "X" else 20 for x in pdb_dict_list[0]["seq"]] # total 20 standard AA in Bio
        return h_V.squeeze(0), pdb_dict_list[0]["ridx"], seqs # [L, 128], [L]



if __name__ == "__main__":
    args = parse_args()
    DATA_DIR = "/data/fanga5/sabdab/"

    if args.model_type == "ProteinMPNN":
        encoder = WrappedProteinMPNNTokenizer(device="cuda" if torch.cuda.is_available() else "cpu")
    elif args.model_type == "MIF":
        encoder = WrappedMIFTokenizer(device="cuda" if torch.cuda.is_available() else "cpu")
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")

    raw_df = pd.read_parquet("/data/fanga5/preprocessed_data/sabdab_2025-05-06-paired.parquet")
    raw_df['sabdab_id'] = range(len(raw_df))
    ab_fname_to_id = {fname: sabdab_id for fname, sabdab_id in zip(raw_df['ab_fname'], raw_df['sabdab_id'])}
    id_to_ab_fname = {sabdab_id: fname for fname, sabdab_id in zip(raw_df['ab_fname'], raw_df['sabdab_id'])}

    loop_df = pd.read_csv(args.dataset_path)
    max_length = (loop_df['end']-loop_df['start']).max()

    embeddings = []
    proteinmpnn_embeddings = []
    for _, row in tqdm(loop_df.iterrows(), desc="Encoding loops", total=len(loop_df)):
        loop_id = row['loop_id']
        ab_fname = id_to_ab_fname[loop_id]
        start = row['start']
        end = row['end']
        continuous_structural_tokens, residue_index, seqs = encoder.encode_structure(DATA_DIR + ab_fname, args.chain_id, use_sequence=True)
        embedding = continuous_structural_tokens[start:end].detach().cpu().numpy().mean(axis=0)
        embeddings.append(embedding)
    embeddings = np.array(embeddings)
    np.save(args.output_path, embeddings)
