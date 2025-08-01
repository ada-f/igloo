import os
os.environ['TRITON_CACHE_DIR'] = "/data2/fanga5/.cache/triton"
import sys
sys.path.append("/data2/fanga5/StructTokenBench/src")
from vqvae_model import VQVAEModel
from protein_chain import WrappedProteinChain
import torch
import numpy as np
from esm.tokenization.sequence_tokenizer import EsmSequenceTokenizer
from esm.utils.constants import esm3 as C
import omegaconf
import pandas as pd
import Bio.PDB.Polypeptide
from tqdm import tqdm
import argparse
import biotite.structure as bs
import biotite.structure.io as bsio

def parse_args():
    args = argparse.ArgumentParser(description="Run IgBert model on antibody sequences.")
    args.add_argument('--dataset_path', type=str, required=True, help='Path to the dataset CSV file.')
    args.add_argument('--output_path', type=str, required=True, help='Path to save the output embeddings as a .npy')
    args.add_argument('--chain_id', type=str, required=True, help='Chain ID to process (e.g., "H", "L").')
    return args.parse_args()

class WrappedOurPretrainedTokenizer():

    def __init__(self, device: torch.device | str = "cpu", model_cfg=None, pretrained_ckpt_path=None, ckpt_name=None):
        self.device = device
        # load
        self.model = VQVAEModel(model_cfg=model_cfg)
        model_states = torch.load(pretrained_ckpt_path, map_location=self.device)["module"]
        new_model_states = {}
        for k,v in model_states.items():
            assert k.startswith("model.")
            new_model_states[k[6:]] = v
        self.model.load_state_dict(new_model_states)
        for p in self.model.parameters():
            p.requires_grad = False
        self.model = self.model.to(self.device)
        
        self.seq_tokenizer = EsmSequenceTokenizer()

        self.ckpt_name = ckpt_name

        # reference: https://github.com/evolutionaryscale/esm/blob/39a3a6cb1e722347947dc375e3f8e2ba80ed8b59/esm/utils/constants/esm3.py#L18C12-L18C35
        self.pad_token_id = self.model.quantizer.codebook.weight.shape[0] + 3

    def get_num_tokens(self):
        return self.model.quantizer.codebook.weight.shape[0] + 5
    
    def get_codebook_embedding(self,):
        if self.ckpt_name == "AminoAseed":
            return self.model.quantizer.linear_proj(self.model.quantizer.codebook.weight)
        else:
            return self.model.quantizer.codebook.weight
    
    def encode_structure(self, pdb_chain, use_continuous=False, use_sequence=False):
        assert use_sequence
        
        coords, plddt, residue_index = pdb_chain.to_structure_encoder_inputs(self.device) # [1, L, 37, 3], [1, L], [1, L]

        attention_mask = coords[:, :, 0, 0] == torch.inf # [1, L]

        sequence = pdb_chain.sequence
        sequence = sequence.replace(C.MASK_STR_SHORT, "<mask>")
        
        seq_ids = self.seq_tokenizer.encode(sequence, add_special_tokens=False)
        seq_ids = torch.tensor(seq_ids, dtype=torch.int64, device=self.device)
        assert len(seq_ids) == len(coords[0])
        
        input_list = (coords, attention_mask, residue_index, seq_ids, pdb_chain)
        quantized_reprs, quantized_indices, reprs = self.model(input_list, use_as_tokenizer=True) # [1, L, dim], [1, L], [1, L, dim]

        seqs = [Bio.PDB.Polypeptide.one_to_index(x) if x != "X" else 20 for x in pdb_chain.sequence] # total 20 standard AA in Bio

        if use_continuous:
            return reprs.squeeze(0), np.array(residue_index.squeeze(0).cpu()), seqs # [L, dim], [L]
        else:
            return quantized_indices.squeeze(0), np.array(residue_index.squeeze(0).cpu()), seqs # [L], [L]

if __name__ == "__main__":
    ckpt_name="AminoAseed"
    pretrained_ckpt_path="/data2/fanga5/StructTokenBench/struct_token_bench_release_ckpt/codebook_512x1024-1e+19-linear-fixed-last.ckpt/checkpoint/mp_rank_00_model_states.pt"
    quantizer_use_linear_project=True

    cfg = omegaconf.OmegaConf.load(os.path.join("/data2/fanga5/StructTokenBench/src/script/config/AminoAseed.yaml"))["model"]
    cfg.quantizer.freeze_codebook = True
    cfg.quantizer._need_init = False
    cfg.quantizer.use_linear_project = True

    encoder = WrappedOurPretrainedTokenizer(
        device="cuda:0",
        model_cfg=cfg,  # Replace with actual model configuration if needed
        pretrained_ckpt_path=pretrained_ckpt_path,  # Replace with actual path
        ckpt_name="AminoAseed"  # or "AminoAseed-2"
    )

    args = parse_args()
    DATA_DIR = "/data/fanga5/sabdab/"
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
        
        struc = bsio.load_structure(os.path.join(DATA_DIR + ab_fname))
        struc = struc[struc.chain_id == args.chain_id]
        pdb_chain = WrappedProteinChain.from_cif(
            path = None,
            chain_id = args.chain_id,
            id = ab_fname,
            is_predicted = False,
            atom_array = struc
        )
        continuous_structural_tokens, residue_index, seqs = encoder.encode_structure(pdb_chain, use_continuous=True, use_sequence=True)
        embedding_aminoaseed = continuous_structural_tokens[start:end].detach().cpu().numpy().mean(axis=0)
        embeddings.append(embedding_aminoaseed)
    
    embeddings = np.array(embeddings)
    np.save(args.output_path, embeddings)