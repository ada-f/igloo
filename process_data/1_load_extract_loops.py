import pandas as pd
from typing import List
import argparse
import biotite.structure as struc
from biotoolkit import (
    load_annotated_structure,
    AnnotationSelector,
    multiprocess,
    sequence,
)

NUM_PAD_AA = 5  # Number of residues to pad for dihedral angle and stem calculation

def check_valid_angles(angles: List[float]) -> bool:
    if len(angles) == 0:
        return False
    if any(pd.isna(angle) for angle in angles):
        return False
    return True

def process_one_row(idx: int):
    entry = meta_df.loc[idx]        
    annotation_categories = ["aho_res_id", "aho_ins_code", "ab_label"]
    structure = load_annotated_structure(entry, ann_df[ann_df.structure_fname == entry.ab_fname], annotation_categories)
    loops = []
    for loop_type, selector in selectors.items():
        if entry.is_vhh:
            if loop_type.startswith("L"):
                continue # nanobodies don't have L chain
        loop_region = selector.extract(structure)
        if len(loop_region) == 0:
            print(f"WARNING: No {loop_type} region found for {entry.ab_fname}")
            continue
        loop_sequence = sequence(loop_region)
        try:
            phi, psi, omega = struc.dihedral_backbone(loop_region)

            backbone = loop_region[struc.filter_peptide_backbone(loop_region)]
            c_atoms = []
            c_alpha_atoms = []
            n_atoms = []
            for atom in backbone:
                if atom.atom_name == "C":
                    c_atoms.append(atom.coord.tolist())
                elif atom.atom_name == "CA":
                    c_alpha_atoms.append(atom.coord.tolist())
                elif atom.atom_name == "N":
                    n_atoms.append(atom.coord.tolist())

        except Exception as e:
            if isinstance(e, struc.BadStructureError):
                print(f"WARNING: Bad structure for {entry.ab_fname} with loop type {loop_type}: {e}")
                continue
            else:
                raise e
        loops.append({
            'sabdab_id': entry.sabdab_id,
            'ab_fname': entry.ab_fname,
            'loop_type': loop_type,
            'loop_sequence': loop_sequence[NUM_PAD_AA:-NUM_PAD_AA], # Exclude the padded residues required for dihedral calculation
            'phi': phi[NUM_PAD_AA:-NUM_PAD_AA],
            'psi': psi[NUM_PAD_AA:-NUM_PAD_AA],
            'omega': omega[NUM_PAD_AA:-NUM_PAD_AA],
            'c_atoms': c_atoms[NUM_PAD_AA:-NUM_PAD_AA],
            'c_alpha_atoms': c_alpha_atoms[NUM_PAD_AA:-NUM_PAD_AA],
            'n_atoms': n_atoms[NUM_PAD_AA:-NUM_PAD_AA],
            'stem_c_atoms': c_atoms[:NUM_PAD_AA] + c_atoms[-NUM_PAD_AA:],
            'stem_c_alpha_atoms': c_alpha_atoms[:NUM_PAD_AA] + c_alpha_atoms[-NUM_PAD_AA:],
            'stem_n_atoms': n_atoms[:NUM_PAD_AA] + n_atoms[-NUM_PAD_AA:],
            'stem_sequence': loop_sequence[:NUM_PAD_AA] + loop_sequence[-NUM_PAD_AA:],
            'loop_start': loop_region[NUM_PAD_AA].abs_res_id,
            'loop_end': loop_region[-(NUM_PAD_AA+1)].abs_res_id,
        })
    return loops


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata_parquet", type=str, required=True)
    parser.add_argument("--annotations_parquet", type=str, required=True)
    parser.add_argument("--ncpu", type=int, default=1)
    args = parser.parse_args()

    meta_df = pd.read_parquet(args.metadata_parquet)
    ann_df = pd.read_parquet(args.annotations_parquet)
 
    selectors = {loop_type: AnnotationSelector("ab_label", loop_type, pad_sel_by=NUM_PAD_AA) for loop_type in ["H1", "H2", "H3", "H4", "L1", "L2", "L3", "L4"]}
    
    loops = multiprocess(
        process_one_row,
        meta_df.index,
        ncpu=16,
    )
    loops = [loop for sublist in loops for loop in sublist]
    loops_df = pd.DataFrame(loops)

    loops_df['is_valid_phi'] = loops_df['phi'].apply(lambda x: check_valid_angles(x))
    loops_df['is_valid_psi'] = loops_df['psi'].apply(lambda x: check_valid_angles(x))
    loops_df['is_valid_omega'] = loops_df['omega'].apply(lambda x: check_valid_angles(x))

    loops_df['is_valid_loop'] = loops_df['is_valid_phi'] & loops_df['is_valid_psi'] & loops_df['is_valid_omega']
    print(f"Number of invalid loops: {sum(~loops_df['is_valid_loop'])}")
    loops_df = loops_df[loops_df['is_valid_loop']].copy()
    loops_df = loops_df.drop(columns=['is_valid_phi', 'is_valid_psi', 'is_valid_omega', 'is_valid_loop'])
    loops_df = loops_df.reset_index(drop=True)

    outfname = args.metadata_parquet.replace("metadata.parquet", "loops.parquet")
    loops_df.to_parquet(outfname, index=False)
    print(f"Saved loops ({len(loops_df)}) to {outfname}")
