import pandas as pd
import os
from tqdm import tqdm
import biotite.structure as bs
import biotite.structure.io as bsio
from multiprocessing import Pool
from biotite.sequence import ProteinSequence
import sys

num_workers = int(sys.argv[1]) if len(sys.argv) > 1 else 16

threetoone = ProteinSequence.convert_letter_3to1

SABDAB_DIR = "sabdab"

chains_df = pd.read_parquet("preprocessed_data/sabdab_2025-05-06-paired_chains.parquet")

temperatures = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
models = ['abmpnn', 'antifold', 'proteinmpnn', 'igbert_igloo', 'igloolm', 'igbert']
SAMPLED_DIR = 'sample_cdrs/'

def calculate_rmsd_of_loop(args):
    model, temp, ab_fname, loop_type, chain_id = args
    model_dir = os.path.join(SAMPLED_DIR, model, f'output/ibex_temp_{temp}')
    chain_info = chains_df[(chains_df['ab_fname'] == ab_fname) & (chains_df['chain_id'] == chain_id)].iloc[0]
    loop_start = chain_info[f'{loop_type}_start']
    loop_end = chain_info[f'{loop_type}_end']

    sabdab_struc = bsio.load_structure(os.path.join(SABDAB_DIR, ab_fname))
    sabdab_struc = sabdab_struc[sabdab_struc.chain_id == chain_id]
    sabdab_struc = sabdab_struc[bs.filter_peptide_backbone(sabdab_struc)]
    res_starts = bs.get_residue_starts(sabdab_struc)
    num_sabdab_residues = len(res_starts)
    sabdab_struc = sabdab_struc[res_starts[loop_start]:res_starts[loop_end]]
    sabdab_seq = "".join([threetoone(x) for x in bs.get_residues(sabdab_struc)[1]])

    rmsds = []
    for sample in range(11):
        fname = os.path.join(model_dir, f'{ab_fname.replace(".pdb", "")}_{chain_id}{loop_type[-1]}_{sample}.pdb')
        if not os.path.exists(fname):
            continue
        sampled_struc = bsio.load_structure(fname)
        sampled_struc = sampled_struc[sampled_struc.chain_id == chain_id]
        sampled_struc = sampled_struc[bs.filter_peptide_backbone(sampled_struc)]
        res_starts = bs.get_residue_starts(sampled_struc)
        num_sampled_residues = len(res_starts)
        if num_sampled_residues != num_sabdab_residues:
            print(f"Skipping {fname} due to residue count mismatch: {num_sampled_residues} vs {num_sabdab_residues}")
            continue
        sampled_struc = sampled_struc[res_starts[loop_start]:res_starts[loop_end]]
        sampled_seq = "".join([threetoone(x) for x in bs.get_residues(sampled_struc)[1]])

        superimposed, _ = bs.superimpose(sabdab_struc, sampled_struc)
        rmsd = bs.rmsd(sabdab_struc, superimposed)
        rmsds.append((model, temp, ab_fname, fname, loop_type, chain_id, sabdab_seq, sampled_seq, rmsd))
    return rmsds

all_results = []
args = []
for loop_type in ['H1', 'H2', 'H3', 'L1', 'L2', 'L3']:
    ab_fnames = pd.read_csv(f"sample_cdrs/antifold/input/antifold_input_{loop_type}.csv")['ab_fname'].to_list()
    for model in models:
        for temp in temperatures:
            for ab_fname in ab_fnames:
                args.append((model, temp, ab_fname, f"CDR{loop_type[1]}", loop_type[0]))
with Pool(processes=num_workers) as pool:
    results = list(tqdm(pool.imap(calculate_rmsd_of_loop, args), total=len(args)))
    results = [item for sublist in results for item in sublist]  # Flatten the list
df = pd.DataFrame(results, columns=['model', 'temperature', 'ab_fname', 'sampled_fname', 'loop_type', 'chain_id', 'sabdab_seq', 'sampled_seq', 'rmsd'])
df.to_csv('sample_cdrs/rmsd_results.csv', index=False)