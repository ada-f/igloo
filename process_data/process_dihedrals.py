import biotite.structure as struc
import biotite.structure.io.pdb as pdb_reader
import pandas as pd
import numpy as np
import json
from tqdm import tqdm
from multiprocessing import Pool
import argparse

AHO_CUTOFFS = {
    "LFW1": (0, 23),
    "L1": (23, 42),
    "LFW2": (42, 56),
    "L2": (56, 72),
    "LFW3": (72, 81),
    "L4": (81, 89),
    "LFW4": (89, 106),
    "L3": (106, 138),
    "LFW5": (138, None),
    "HFW1": (0, 23),
    "H1": (23, 42),
    "HFW2": (42, 56),
    "H2": (56, 69),
    "HFW3": (69, 81),
    "H4": (81, 89),
    "HFW4": (89, 106),
    "H3": (106, 138),
    "HFW5": (138, None),
}

def get_loop_structure(chain_id, start_res_id_chain, end_res_id_chain, pdb_path, keep_bfactor=False):
    pdb_file = pdb_reader.PDBFile.read(pdb_path)
    atom_array = pdb_file.get_structure()
    bfactor = pdb_reader.PDBFile.get_b_factor(pdb_file)[0]

    if len(bfactor) != atom_array.shape[1]:
        print(f"Warning: B-factor length ({bfactor.shape}) does not match atom array ({atom_array.shape}) for {pdb_path}.")
        return None
    bfactor = bfactor[atom_array.chain_id == chain_id]
    atom_array = atom_array[:, atom_array.chain_id == chain_id]

    phi, psi, omega = struc.dihedral_backbone(atom_array)
    phi = phi[0][start_res_id_chain:end_res_id_chain]
    psi = psi[0][start_res_id_chain:end_res_id_chain]
    omega = omega[0][start_res_id_chain:end_res_id_chain]

    if np.isnan(phi).any() or np.isnan(psi).any() or np.isnan(omega).any():
        print(f"Warning: NaN dihedral angles found in loop at {start_res_id_chain}:{end_res_id_chain} for {pdb_path}.")
        return None
    
    atom_array_Calpha = atom_array[:, atom_array.atom_name == "CA"]
    loop_coords = atom_array_Calpha[:, start_res_id_chain:end_res_id_chain].coord.reshape(-1, 3)
    stem_coords = np.concatenate((
        atom_array_Calpha[:, start_res_id_chain-5:start_res_id_chain].coord.reshape(-1, 3),
        atom_array_Calpha[:, end_res_id_chain:end_res_id_chain+5].coord.reshape(-1, 3)
    ))
    if stem_coords.shape[0] != 10:
        print(f"Warning: Stem coordinates for loop at {start_res_id_chain}:{end_res_id_chain} in {pdb_path} do not have exactly 10 atoms. Found {stem_coords.shape[0]} atoms.")
        return None

    output = {
        "loop_c_alpha_atoms": loop_coords.tolist(),
        "stem_c_alpha_atoms": stem_coords.tolist(),
        "phi": phi.tolist(),
        "psi": psi.tolist(),
        "omega": omega.tolist(),
    } 

    if keep_bfactor:
        output["bfactor"] = np.mean(bfactor[start_res_id_chain:end_res_id_chain]).item()
    return output

def process_one_loop(aho_seq, loop_type, pdb_path, keep_bfactor=False):
    loop_seq = aho_seq[AHO_CUTOFFS[loop_type][0]:AHO_CUTOFFS[loop_type][1]].replace('-', '')
    if len(loop_seq) == 0:
        return None

    start_res_id_chain = len(aho_seq[:AHO_CUTOFFS[loop_type][0]].replace('-', ''))
    end_res_id_chain = len(aho_seq[:AHO_CUTOFFS[loop_type][1]].replace('-', ''))
    chain_id = loop_type[0]
    
    output = get_loop_structure(chain_id, start_res_id_chain, end_res_id_chain, pdb_path, keep_bfactor)
    output['loop_sequence'] = loop_seq
    return output

def process_loops_from_one_entry(idx):
    entry = df.iloc[idx]
    loop_types = ['L1', 'L2', 'L3', 'L4', 'H1', 'H2', 'H3', 'H4']
    outputs = []
    for loop_type in loop_types:
        loop_data = {'loop_id': f"{entry[ID_KEY]}_{loop_type}", "loop_type": loop_type, ID_KEY: entry[ID_KEY]}
        seq = entry[AHO_LIGHT_KEY] if loop_type.startswith('L') else entry[AHO_HEAVY_KEY]
        dihedral_loop_data = process_one_loop(seq, loop_type, entry['pdb_path'], keep_bfactor=KEEP_BFACTOR)
        if dihedral_loop_data is None:
            return [] # Skip this entry if any loop data is None
        else:
            loop_data.update(dihedral_loop_data)
        outputs.append(loop_data)
    return outputs

def get_aho_chain_sections(fv_aho, chain):
    assert chain in ['L', 'H'], "Chain must be either 'L' for light or 'H' for heavy"
    # Retrieve start and end indices from AHO_CUTOFFS for heavy chain sections
    fwh1 = fv_aho[AHO_CUTOFFS[f"{chain}FW1"][0]:AHO_CUTOFFS[f"{chain}FW1"][1]].replace("-", "")
    cdrh1 = fv_aho[AHO_CUTOFFS[f"{chain}1"][0]:AHO_CUTOFFS[f"{chain}1"][1]].replace("-", "")
    fwh2 = fv_aho[AHO_CUTOFFS[f"{chain}FW2"][0]:AHO_CUTOFFS[f"{chain}FW2"][1]].replace("-", "")
    cdrh2 = fv_aho[AHO_CUTOFFS[f"{chain}2"][0]:AHO_CUTOFFS[f"{chain}2"][1]].replace("-", "")
    fwh3a = fv_aho[AHO_CUTOFFS[f"{chain}FW3"][0]:AHO_CUTOFFS[f"{chain}FW3"][1]].replace("-", "")
    cdrh4 = fv_aho[AHO_CUTOFFS[f"{chain}4"][0]:AHO_CUTOFFS[f"{chain}4"][1]].replace("-", "")
    fwh3b = fv_aho[AHO_CUTOFFS[f"{chain}FW4"][0]:AHO_CUTOFFS[f"{chain}FW4"][1]].replace("-", "")
    cdrh3 = fv_aho[AHO_CUTOFFS[f"{chain}3"][0]:AHO_CUTOFFS[f"{chain}3"][1]].replace("-", "")
    fwh4 = fv_aho[AHO_CUTOFFS[f"{chain}FW5"][0]:AHO_CUTOFFS[f"{chain}FW5"][1]].replace("-", "") # AHO_CUTOFFS["HFW5"][1] is None, meaning slice to end
    return pd.Series({
        f'{chain}FW1': fwh1,
        f'{chain}1': cdrh1,
        f'{chain}FW2': fwh2,
        f'{chain}2': cdrh2,
        f'{chain}FW3': fwh3a,
        f'{chain}4': cdrh4,
        f'{chain}FW4': fwh3b,
        f'{chain}3': cdrh3,
        f'{chain}FW5': fwh4
    })

def get_loop_regions(df, aho_light_key, aho_heavy_key):
    light_sections  = df[aho_light_key].apply(get_aho_chain_sections, chain='L')
    heavy_sections  = df[aho_heavy_key].apply(get_aho_chain_sections, chain='H')
    df = pd.concat([df, light_sections, heavy_sections], axis=1)
    return df

def parse_args():
    parser = argparse.ArgumentParser(description="Process loops and calculate dihedral angles.")
    parser.add_argument('--structure_dir', type=str, required=True, help="Directory containing PDB files.")
    parser.add_argument('--df_path', type=str, required=True, help="Path to the DataFrame file (CSV or Parquet).")
    parser.add_argument('--parquet_output_path', type=str, default=None, help="Path to save the output Parquet file.")
    parser.add_argument('--jsonl_output_path', type=str, default=None, help="Path to save the output JSONL file with loops.")
    parser.add_argument('--id_key', type=str, default='sequence_id', help="Column name for sequence ID in the DataFrame.")
    parser.add_argument('--aho_light_key', type=str, default='aho_light', help="Column name for AHO light chain in the DataFrame.")
    parser.add_argument('--aho_heavy_key', type=str, default='aho_heavy', help="Column name for AHO heavy chain in the DataFrame.")
    parser.add_argument('--num_workers', type=int, default=16, help="Number of parallel workers to use.")
    parser.add_argument('--chunk', type=int, default=None, help="Chunk index for processing")
    parser.add_argument('--chunk_total', type=int, default=None, help="Total number of chunks for processing")
    parser.add_argument('--bfactor', action='store_true', help="Calculate and include B-factors in the output.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    ID_KEY = args.id_key
    AHO_LIGHT_KEY = args.aho_light_key
    AHO_HEAVY_KEY = args.aho_heavy_key
    KEEP_BFACTOR = args.bfactor

    if args.df_path.endswith('.csv'):
        df = pd.read_csv(args.df_path)
    elif args.df_path.endswith('.parquet'):
        df = pd.read_parquet(args.df_path)
    else:
        raise ValueError("Unsupported file format for df_path. Use .csv or .parquet.")

    df['pdb_path'] = args.structure_dir + '/' + df[ID_KEY].astype(str) + '.pdb'

    if args.chunk is not None and args.chunk_total is not None:
        chunk_size = len(df) // args.chunk_total
        df = df[args.chunk * chunk_size : (args.chunk + 1) * chunk_size]
    with Pool(processes=args.num_workers) as pool:
        results = list(tqdm(pool.imap(process_loops_from_one_entry, range(len(df))), total=len(df), desc="Processing loops"))
    outputs = [item for sublist in results for item in sublist]

    if args.jsonl_output_path:
        with open(args.jsonl_output_path, 'w') as f:
            for loop in outputs:
                f.write(json.dumps(loop) + '\n')
        print(f"Processed {len(outputs)} loops and saved to {args.jsonl_output_path}")
    
    if args.parquet_output_path:
        outputs = pd.DataFrame(outputs)
        outputs.to_parquet(args.parquet_output_path, index=False)
        print(f"Processed {len(outputs)} loops and saved to {args.parquet_output_path}")