import argparse
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import fastpdb
import os
import biotite.structure as struc
from biotite.structure import AtomArray
from biotite.structure.filter import filter_amino_acids, filter_polymer
from biotite.structure.info import one_letter_code, amino_acid_names
from pathlib import Path
from typing import Literal, Callable, Iterable
from multiprocessing import Pool
from typing import Callable, Iterable
from functools import wraps
from operator import attrgetter
import anarci_numbering

########## Constants ##########
PDB_DIR = Path(os.environ['PDB_DIR']).resolve()

_aa_3to1 = {res: one_letter_code(res) for res in amino_acid_names() if one_letter_code(res) is not None} | {
    "HOH": "0",
    "SOL": "0",
    "SLL": "K", # 5veb chain A
    "DV7": "G", # 6w9g LC (LMY), in fasta as [...SWE]X[PPT...]
    "PCA": "Q", # 5iwl HC auth B, in fasta as Q, technically ambiguous with E
}

def aa_3to1(res: str) -> str:
    return _aa_3to1.get(res, "X")

LC_AHO_UPPER_BOUNDS = [
    (23, "LFW1"),
    (42, "L1"),
    (56, "LFW2"),
    (72, "L2"),
    (81, "LFW3"),
    (89, "L4"),
    (106, "LFW4"),
    (138, "L3"),
    (148, "LFW5")
]
HC_AHO_UPPER_BOUNDS = [
    (23, "HFW1"),
    (42, "H1"),
    (56, "HFW2"),
    (69, "H2"),
    (81, "HFW3"),
    (89, "H4"),
    (106, "HFW4"),
    (138, "H3"),
    (149, "HFW5")
]

########## Antibody alignment ##########
def pd_aho_alignment(seqs: list[str], ncpu: int = 1) -> tuple:
    def _split_index(idx: str) -> tuple[int, str]:
        if idx[-1].isalpha():
            return int(idx[:-1]), idx[-1]
        return int(idx), ""

    def _split_index_list(indexes: list[str]) -> tuple[list[int], list[str]]:
        resids = []
        icodes = []
        for res, ins in [_split_index(i) for i in indexes]:
            resids.append(res)
            icodes.append(ins)
        return resids, icodes

    def _aho_flag(fv_seq: str, resids: list[int]) -> bool:
        # Return False because no output from alignment to judge
        if fv_seq is None:
            return False
        # Otherwise will throw ValueError
        if not (23 in resids and 106 in resids):
            return False
        return fv_seq[resids.index(23)] == "C" and fv_seq[resids.index(106)] == "C"

    # Batch alignment
    aho_seqs, meta = anarci_numbering(
        sequences=seqs,
        scheme="aho",
        allow_fix=False,
        return_metadata=True,
        ncpu=ncpu
    )

    chain_types = [mdata["chain_type"] if mdata else None for mdata in meta]
    # Trim LC to 148 _if_ the last position is already a gap (otherwise that means it's resolved in structure)
    aho_seqs = [seq[:148] if (seq and ct=="L" and seq[-1]=="-") else seq for seq, ct in zip(aho_seqs, chain_types)]
    # Pad aho_seqs to full length based on chain type
    aho_seqs = [seq.ljust((148 if ct=="L" else 149), "-") if seq else None for seq, ct in zip(aho_seqs, chain_types)]

    # Parse out raw residue numbers + insertion codes
    # (tupled as "resinfo") for detailed checks
    aho_resids, aho_icodes = [], []
    for seq, mdata in zip(aho_seqs, meta):
        if mdata is None:
            _resids, _icodes = None, None
        else:
            _resids, _icodes = _split_index_list(mdata["scheme_indexes"])
            # Match resinfo arrays to non-gap positions only
            _resids = [r for s,r in zip(seq, _resids) if s != "-"]
            _icodes = [i for s,i in zip(seq, _icodes) if s != "-"]
        aho_resids.append(_resids)
        aho_icodes.append(_icodes)

    # Check cysteine positions based on lists
    aho_alignment_flags = [
        _aho_flag(fv_seq, resids)
        for fv_seq, resids in zip(aho_seqs, aho_resids)
    ]

    return chain_types, aho_seqs, aho_resids, aho_icodes, aho_alignment_flags

### We only have to realign the truly bad ones
def realign(df: pd.DataFrame, chain: Literal["H","L"]) -> dict[int, tuple]:
    match chain:
        case "H":
            bad_df = df[~df.hc_alignment_okay]
            seq_col = "fv_heavy"
        case "L":
            bad_df = df[~df.lc_alignment_okay]
            seq_col = "fv_light"
    bad_idx = bad_df.index
    bad_fnames = bad_df.ab_fname
    chain_types, aho_seqs, aho_resids, aho_icodes, align_flags = pd_aho_alignment(bad_df[seq_col].values)

    return dict(zip(bad_idx, list(zip(*(bad_fnames, chain_types, aho_seqs, aho_resids, aho_icodes, align_flags)))))

### Update the metadata, keep the aho resid/icode lists for later
def update_realigned_metadata(df: pd.DataFrame, realn: dict[int, tuple]) -> dict[int, tuple]:
    for idx in realn.keys():
        bad_fname, chain_type, aho_seq, aho_resids, aho_icodes, align_flag = realn[idx]
        assert df.at[idx, "ab_fname"] == bad_fname
        # Only update if we actually fixed it
        if align_flag:
            if chain_type == "H":
                if df.at[idx, "fv_heavy_aho"].replace("-","") != df.at[idx, "fv_heavy"]:
                    print(f"WARNING: residue mismatch for {idx=} and {aho_seq=}")
                df.at[idx, "fv_heavy_aho"] = aho_seq
                df.at[idx, "hc_alignment_okay"] = align_flag
            elif chain_type == "L":
                if df.at[idx, "fv_light_aho"].replace("-","") != df.at[idx, "fv_light"]:
                    print(f"WARNING: residue mismatch for {idx=} and {aho_seq=}")
                df.at[idx, "fv_light_aho"] = aho_seq
                df.at[idx, "lc_alignment_okay"] = align_flag
    return realn

########## Antibody annotations ##########
def check_aho_alignment_from_seq(aho_seq: str) -> bool:
    # Empty sequence gets auto-approved
    if pd.isna(aho_seq):
        return True
    # Since this is a shortcut, assume valid ONLY for normal length sequences
    if len(aho_seq) > 149:
        return False
    return aho_seq[22] == "C" and aho_seq[105] == "C"

def aho_labels_from_resids(chain_type: Literal["H","L"], aho_resids: list[int]):
    match chain_type:
        case "H": aho_bounds = HC_AHO_UPPER_BOUNDS
        case "L": aho_bounds = LC_AHO_UPPER_BOUNDS
        case _: raise ValueError("chain_type must be 'H' or 'L'")
    all_labels = []
    max_idx = aho_bounds[-1][0]-1
    # HACK: fix the off-by-one mismatch for fv_light
    if chain_type=="L" and aho_resids[-1]==149:
        max_idx += 1
    idx = 0
    bound,label = aho_bounds[idx]
    for res in aho_resids:
        if res > bound and res < max_idx:
            idx += 1
            bound,label = aho_bounds[idx]
        all_labels.append(label)

    return all_labels

def aho_resinfo_from_aho_seq(aho_seq: str) -> tuple[list[int], list[str]]:
    """
    Converts a "NORMAL" (aka no insertion codes) AHo aligned
    fv sequence to a resinfo tuple (resids, icodes).
    """
    # Empty sequence gets empty response (align flag okay though)
    if pd.isna(aho_seq):
        return None
    assert len(aho_seq) < 150, "Passed aho_seq is too long! Must be a canonical 148 or 149 length fv sequence."
    aho_resids = [i for i,s in enumerate(aho_seq,1) if s != "-"]
    aho_icodes = [""]*len(aho_resids)
    assert len(aho_resids) == len(aho_seq.replace("-",""))

    return aho_resids, aho_icodes

def populate_ab_resinfo_dict(df: pd.DataFrame, realn_hc: dict[int, tuple], realn_lc: dict[int, tuple]) -> tuple[dict[int, tuple]]:
    hc_resinfo = {}
    lc_resinfo = {}
    hc_keys = set(realn_hc.keys())
    lc_keys = set(realn_lc.keys())
    # idx won't be in keys if we could align already
    # so miss either criterion means don't add it
    for idx in df.index:
        # Only add idx:(resid,icode) if align_flag==True
        if idx in hc_keys and (data := realn_hc[idx])[-1]:
            hc_resinfo[idx] = (data[3], data[4])
        elif (resinfo := aho_resinfo_from_aho_seq(df.loc[idx].fv_heavy_aho)) is not None:
            hc_resinfo[idx] = resinfo
        # else resinfo is None due to lack of sequence, so don't do anything

        if idx in lc_keys and (data := realn_lc[idx])[-1]:
            lc_resinfo[idx] = (data[3], data[4])
        elif (resinfo := aho_resinfo_from_aho_seq(df.loc[idx].fv_light_aho)) is not None:
            lc_resinfo[idx] = resinfo

    assert len(hc_resinfo) == len(df[~df.fv_heavy_aho.isnull()])
    assert len(lc_resinfo) == len(df[~df.fv_light_aho.isnull()])

    return hc_resinfo, lc_resinfo

def add_ab_labels(array: AtomArray, chain_resinfo: dict[str, tuple[list[int], list[str]]]):
    ### chain_resinfo is {chain:(aho_resids, aho_icodes)}
    # Overwrite the ab_label column in separate poses for convenience
    split_pose = {c: array[array.chain_id==c] for c in struc.get_chains(array)}
    for chain, (resids, icodes) in chain_resinfo.items():
        annotate_residue_wise(split_pose[chain], "aho_res_id", resids)
        annotate_residue_wise(split_pose[chain], "aho_ins_code", icodes)
        annotate_residue_wise(split_pose[chain], "ab_label", aho_labels_from_resids(chain, resids))
    joined_pose = struc.concatenate(split_pose.values())

    array.set_annotation("aho_res_id", joined_pose.aho_res_id)
    array.set_annotation("aho_ins_code", joined_pose.aho_ins_code)
    array.set_annotation("ab_label", joined_pose.ab_label)

    # Confirm that we removed all "antibody" dummy label for all _passed_ chains
    assert np.isin("antibody", array[np.isin(array.chain_id, list(chain_resinfo.keys()))], invert=True)


########## AtomArray functions ##########
def size(array: AtomArray) -> int:
    return struc.get_residue_count(array)

def sequence(array: AtomArray) -> str:
    return "".join(aa_3to1(res) for res in struc.get_residues(array)[1])

def chain_sequences(array: AtomArray) -> dict[str, str]:
    return {
        chain.item(): "".join(
            aa_3to1(res) for res in struc.get_residues(array[array.chain_id==chain])[1]
        )
        for chain in np.unique(struc.get_chains(array))
    }

def spread_abs_residue_wise(array: AtomArray, values: list | np.ndarray):
    _abs_res_id, abs_res_idx = np.unique(array.abs_res_id, return_index=True)
    assert len(_abs_res_id) == len(values), "Number of values and residues (`size(array)`) in array must match!"
    # Have to include the last atom to capture the last residue
    abs_res_idx = np.append(abs_res_idx, array.shape[0])
    natoms = abs_res_idx[1:] - abs_res_idx[:-1]
    return np.repeat(values, natoms)

def annotate_residue_wise(array: AtomArray, label: str, values: list | np.ndarray):
    #vals = struc.spread_residue_wise(array, values)
    vals = spread_abs_residue_wise(array, values)
    assert len(vals) == len(array)
    array.set_annotation(label, vals)

def annotate_broadcast(array: AtomArray, label: str, value: str | float | int):
    vals = np.array([value]*array.shape[0])
    array.set_annotation(label, vals)

def annotate_array(array: AtomArray, array_annotations: pd.DataFrame, annotation_categories: list[str]):
    for cat in annotation_categories:
        annotate_residue_wise(array, cat, array_annotations[cat])
        #array.set_annotation(cat, spread_abs_residue_wise(array, annotation_categories[ann]))
    assert all(ann in array.get_annotation_categories() for ann in annotation_categories)


########## AtomArray loading ##########
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
    # Add column with absolute residue numbering
    resids = np.array(range(1, size(array)+1), dtype=np.int32)
    array.set_annotation("abs_res_id", struc.create_continuous_res_ids(array, restart_each_chain=False))
    assert all(array.abs_res_id == spread_abs_residue_wise(array, resids))

    return array

def load_annotated_structure(entry: pd.Series, ann_df: pd.DataFrame, annotation_categories: list[str] = ["aho_res_id", "aho_ins_code", "ab_label"]) -> AtomArray:
    array = load_structure(entry)
    _anns = ann_df[ann_df.structure_fname == entry.ab_fname]
    annotate_array(array, _anns, annotation_categories)

    return array

########## Selectors ##########
""" Decorator for residue selection padding functionality """
def expandable(get_value):
    def dec(fn):
        @wraps(fn)
        def wrapper(self, *args, **kwargs):
            result = fn(self, *args, **kwargs)
            # No pad if empty selection e.g. AnnotationSelector("ab_label", cdr) on a VHH
            if result.size == 0:
                return result
            n = get_value(self)
            return np.pad(result, pad_width=n, mode="linear_ramp", end_values=(result[0]-n, result[-1]+n))
        return wrapper
    return dec

class Selector:
    """ Selector base class
    
    pad_sel_by : int
        Expands residue selection by this amount equally on both sides
    """
    def __init__(self, pad_sel_by: int = 0):
        self.pad = pad_sel_by

    # Return per-atom boolean array of Selector 
    def atom_mask(self, array: AtomArray) -> np.ndarray[bool]:
        return np.isin(array.abs_res_id, self.get_residues(array))

    # Return per-residue boolean array of Selector
    def residue_mask(self, array: AtomArray) -> np.ndarray[bool]:
        return np.isin(np.unique(array.abs_res_id), self.get_residues(array))

    # Returns a new AtomArray corresponding to the Selector
    def extract(self, array: AtomArray) -> AtomArray:
        #return array[self.apply(array)]
        return array[self.atom_mask(array)]

class ChainSelector(Selector):
    """ Selects all atoms found in `which_chains` """
    def __init__(self, which_chains: str | list[str]):
        self.which_chains = which_chains
        if type(self.which_chains) == str:
            self.which_chains = [self.which_chains]

    # Return an int array of abs_res_id corresonding to the selection
    def get_residues(self, array: AtomArray) -> np.ndarray[np.int32]:
        return np.unique(array[np.isin(array.chain_id, self.which_chains)].abs_res_id)

class AnnotationSelector(Selector):
    """ Selects atoms matching `labels` in the AtomArray field `annotation` """
    def __init__(self, annotation: str, labels: str | list[str], pad_sel_by: int = 0):
        super().__init__(pad_sel_by)
        self.annotation = annotation
        self.labels = labels
        if type(self.labels) == str:
            self.labels = [self.labels]

    @expandable(attrgetter('pad'))
    def get_residues(self, array: AtomArray) -> np.ndarray[np.int32]:
        assert self.annotation in array.get_annotation_categories()
        return np.unique(array[np.isin(array.get_annotation(self.annotation), self.labels)].abs_res_id)


########## Annotations output ##########
LABEL_SCHEMA = pa.schema(
    [
        ("structure_fname", pa.string()),
        ("pdb_code", pa.string()),
        ("chain_id", pa.string()),
        ("res_id", pa.int64()),
        ("ins_code", pa.string()),
        ("res_name", pa.string()),
        ("abs_res_id", pa.int64()),
        ("aho_res_id", pa.int64()),
        ("aho_ins_code", pa.string()),
        ("ab_label", pa.string())
    ]
)

def atom_array_to_record_batch(entry: pd.Series, array: AtomArray, resolution: Literal["atom", "residue", "chain"] = "residue") -> pa.RecordBatch:
    match resolution:
        case "atom":
            pass
        case "residue":
            _abs_res_id, abs_res_idx = np.unique(array.abs_res_id, return_index=True)
            assert all(array[abs_res_idx].abs_res_id == _abs_res_id)
            array = array[abs_res_idx]
        case "chain":
            _chain_id, chain_idx = np.unique(array.chain_id, return_index=True)
            assert all(array[chain_idx].chain_id == _chain_id)
            array = array[chain_idx]
        case _:
            raise ValueError(f"{resolution=} not supported! Please use 'atom', 'residue', or 'chain'.")

    fname = entry.ab_fname
    pdb_code = entry.pdb_code
    record_batch = pa.RecordBatch.from_arrays(
        [
            pa.array([fname] * len(array)),
            pa.array([pdb_code] * len(array)),
            pa.array(array.chain_id),
            pa.array(array.res_id),
            pa.array(array.ins_code),
            pa.array(array.res_name),
            pa.array(array.abs_res_id),
            pa.array(array.aho_res_id),
            pa.array(array.aho_ins_code),
            pa.array(array.ab_label),
        ],
        schema=LABEL_SCHEMA,
    )

    return record_batch

def write_batches_parquet(batches: list[pa.RecordBatch], outname: str):
    # Writer expects list of batches so no need to pa.concat_batches first
    table = pa.Table.from_batches(batches)
    pq.write_table(table, outname)

########## Gotta go fast ##########
def multiprocess(func: Callable, vals: Iterable, ncpu: int = 1, worker_init_fn: Callable = None) -> list:
    # Create a pool of worker processes
    with Pool(processes=ncpu, initializer=worker_init_fn) as pool:
        # Use the pool to map the function across the iterable
        result = pool.map(func, vals)
    return result