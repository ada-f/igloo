import argparse
import numpy as np
import pandas as pd
import fastpdb
import biotite.structure as struc
from biotite.structure import AtomArray
from biotite.structure.filter import filter_amino_acids, filter_polymer
from pathlib import Path

# AtomArray selector + functions
from biotoolkit import (
    AnnotationSelector,
    annotate_residue_wise,
    spread_abs_residue_wise,
    size,
    sequence,
    chain_sequences,
)
# Alignment + antibody labeling functions
from biotoolkit import (
    check_aho_alignment_from_seq,
    realign,
    update_realigned_metadata,
    populate_ab_resinfo_dict,
    add_ab_labels,
)
# Annotations output
from biotoolkit import (
    atom_array_to_record_batch,
    write_batches_parquet,
)
from biotoolkit import multiprocess
import os

PDB_DIR = Path(os.environ['PDB_DIR']).resolve()

def load_structure(entry: pd.Series) -> AtomArray:
    pdb_path = PDB_DIR / entry.ab_fname
    array = fastpdb.PDBFile.read(pdb_path).get_structure(
        altloc="occupancy",
        model=1,
        extra_fields=["b_factor", "occupancy"]
    )
    # Filter to only L-peptide-linking residues w/ at least 1 peptide bond
    array = array[
        filter_amino_acids(array) &
        filter_polymer(array)
    ]

    # Add column with absolute numbering
    resids = np.array(range(1, size(array)+1), dtype=np.int32)
    array.set_annotation("abs_res_id", struc.create_continuous_res_ids(array, restart_each_chain=False))
    assert all(array.abs_res_id == spread_abs_residue_wise(array, resids))

    #annotate_residue_wise(array, "abs_res_id", resids)
    # Just in case there's ever any weirdness w/ insertion codes
    #assert all(array.abs_res_id == struc.create_continuous_res_ids(array, restart_each_chain=False))


    # Check that fv_heavy/light sequence corresponds exactly to what we have in structure
    chainseqs = chain_sequences(array)
    if "H" in chainseqs.keys() and chainseqs["H"] != entry.fv_heavy:
        print(f"WARNING: Mismatched fv_heavy and resolved for {entry.ab_fname}")
    if "L" in chainseqs.keys() and chainseqs["L"] != entry.fv_light:
        print(f"WARNING: Mismatched fv_light and resolved for {entry.ab_fname}")

    # Initialize "dummy" columns for AHo information + ab_labels
    array.set_annotation("aho_ins_code", [""]*array.shape[0])
    array.set_annotation("aho_res_id", np.where(np.isin(array.chain_id, ["H","L"], invert=True), 0, -1))
    array.set_annotation("ab_label", np.where(np.isin(array.chain_id, ["H","L"], invert=True), "antigen", "antibody"))

    return array

def parallel_process_entries(idx: int):
    entry = df.loc[idx]
    # Define {ab_chain: (aho_resid, aho_icodes)} for current index
    chain_resinfo = {}
    if idx in hc_resinfo_dict.keys():
        chain_resinfo["H"] = hc_resinfo_dict[idx]
    if idx in lc_resinfo_dict.keys():
        chain_resinfo["L"] = lc_resinfo_dict[idx]

    # Load structure and apply per-residue AHo + ab_label annotations
    array = load_structure(entry)
    add_ab_labels(array, chain_resinfo)

    # Store the annotations RecordBatch
    batch = atom_array_to_record_batch(entry, array, resolution=args.annotation_level)

    # Add CDR sequences to dataframe
    cdr_seqs = [sequence(sel.extract(array)) for sel in sels]
    ###df.loc[idx, _cdr_cols]
    return batch, cdr_seqs

# ex: python 0_process_annotate_ibex_data.py --parquet_in "sabdab_2025-05-06-paired-matched-pdbs.parquet" --ncpu 16 --annotation_level "residue"
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet_in", type=str, required=True)
    parser.add_argument("--ncpu", type=int, default=1)
    parser.add_argument("--annotation_level", type=str, choices=["atom","residue","chain"], default="residue")
    args = parser.parse_args()

    print(f"Loading paired-matched-pdbs parquet file", args.parquet_in)
    df = pd.read_parquet(args.parquet_in)
    df.reset_index(drop=True, inplace=True) # sabdab indexes are not unique
    df['sabdab_id'] = range(len(df))

    # Include any filters for df
    # df = df[df['resolution'] < 3.5]

    ##### MANUAL PATCH #####
    ''' Many other mismatches (likely noncanonicals but same length so no issue for numbering) '''
    ### 5iwl (15616,15617), 7sen, 6w5a all have PCA (pyroglutamic acid) at H:1
    # online all their SEQRES lists it as Q so that's what I cast it to
    def _patch_pca(pdb_code: str):
        df.loc[df.pdb_code==pdb_code, "fv_heavy_aho"] = df.loc[df.pdb_code==pdb_code, "fv_heavy_aho"].apply(lambda x: "Q"+x[1:])
        df.loc[df.pdb_code==pdb_code, "fv_heavy"] = df.loc[df.pdb_code==pdb_code, "fv_heavy"].apply(lambda x: "Q"+x)
    for pdbid in ["5iwl", "7sen", "6w5a"]: _patch_pca(pdbid)

    ### 7sgm idx 15876-15878 has DV7 residue (X in pdb but Gly); G gets added to end of H3 gap according to pd_anarci_numbering
    # indices 15879,15880,15881,15882,15884 also have exact same issue (15883 missing D:1 so use contains)
    _mask_swepp = (~df.fv_light.isnull()) & (df.fv_light.str.contains("SWEPP"))
    df.loc[_mask_swepp, "fv_light"] = df.loc[_mask_swepp, "fv_light"].str.replace("SWEPP","SWEGPP")
    df.loc[_mask_swepp, "fv_light_aho"] = df.loc[_mask_swepp, "fv_light_aho"].str.replace("SWE------------------------PP", "SWE-----------------------GPP")

    ### 7t74 (15882) has pTyr in fv_heavy
    df.loc[df.pdb_code=="7t74", "fv_heavy"] = df.loc[df.pdb_code=="7t74", "fv_heavy"].str.replace("SDDSQH","SDDYSQH")
    df.loc[df.pdb_code=="7t74", "fv_heavy_aho"] = df.loc[df.pdb_code=="7t74", "fv_heavy_aho"].str.replace("SDD--------SQH","SDD-------YSQH")


    ### Get rid of lower-case
    for col in ["fv_heavy", "fv_heavy_aho", "fv_light", "fv_light_aho"]:
        df[col] = df[col].str.upper()

    ### Left-pad to full length
    df.loc[~df.fv_heavy_aho.isna(), "fv_heavy_aho"] = df.loc[~df.fv_heavy_aho.isna(), "fv_heavy_aho"].apply(lambda x: x.ljust(149,"-"))
    df.loc[~df.fv_light_aho.isna(), "fv_light_aho"] = df.loc[~df.fv_light_aho.isna(), "fv_light_aho"].apply(lambda x: x.ljust(148,"-"))
    ### NOTE: undoing this truncation changed mismatch_mask.value_counts() to ALL FALSE
    #df.loc[df.fv_light_aho.str.len()==149, "fv_light_aho"] = df.loc[df.fv_light_aho.str.len()==149, "fv_light_aho"].apply(lambda x: x[:148])

    # 18506 len=149 chains (18815 total) df.fv_heavy_aho.str.len().value_counts()
    # 15535 len=148 chains (15673 total) df.fv_light_aho.str.len().value_counts()
    # 11213 len=148 + 4322 len=149 if skip truncation

    ### Initial sequence-level check (cysteine + len < 150)
    df["hc_alignment_okay"] = df["fv_heavy_aho"].apply(check_aho_alignment_from_seq) # 339 false
    df["lc_alignment_okay"] = df["fv_light_aho"].apply(check_aho_alignment_from_seq) # 196 false

    # Realign entries where alignment_okay==False & update df metadata
    print(f"Realigning extra long sequences using {args.ncpu} cpu(s)...")
    realn_hc = update_realigned_metadata(df, realign(df, "H"))
    realn_lc = update_realigned_metadata(df, realign(df, "L"))

    ### OPTIONAL filters for misaligned chains
    filter_hc_bad = (df.hc_alignment_okay) # 26
    filter_lc_bad = (df.lc_alignment_okay) # 27
    filter_any_bad = ((df.hc_alignment_okay) & (df.lc_alignment_okay)) # 50
    df = df[filter_any_bad]

    ### Populate ab_resids/icodes for all alignable chains
    hc_resinfo_dict, lc_resinfo_dict = populate_ab_resinfo_dict(df, realn_hc, realn_lc)

    ### Initialize Selectors + df columns for fetching/storing CDR seqs directly from structures
    CDR_NAMES = ["H1", "H2", "H3", "H4", "L1", "L2", "L3", "L4"]
    sels = [AnnotationSelector("ab_label", cdr) for cdr in CDR_NAMES]
    _cdr_cols = [f"{cdr}_seq" for cdr in CDR_NAMES]
    df[_cdr_cols] = None

    ### Parse data, post-process, save to parquet
    print(f"LOGGING: Recording annotations at {args.annotation_level.upper()}-level detail")
    print(f"Processing {len(df)} entries...")

    data = multiprocess(
        parallel_process_entries,
        df.index,
        ncpu=args.ncpu
    )
    batches = [d[0] for d in data]
    for idx,d in zip(df.index, data):
        df.loc[idx, _cdr_cols] = d[1]

    print(f"Writing metadata for {args.parquet_in}")
    df.to_parquet(args.parquet_in.replace(".parquet", "_metadata.parquet"))
    print(f"Writing annotations for {args.parquet_in}")
    write_batches_parquet(batches, args.parquet_in.replace(".parquet", "_annotations.parquet"))