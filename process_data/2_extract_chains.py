import pandas as pd
import argparse
from biotoolkit import (
    load_annotated_structure,
    AnnotationSelector,
    multiprocess,
    sequence,
)

H_SEGMENTS = ["HFW1", "H1", "HFW2", "H2", "HFW3", "H4", "HFW4", "H3", "HFW5"]
L_SEGMENTS = ["LFW1", "L1", "LFW2", "L2", "LFW3", "L4", "LFW4", "L3", "LFW5"]
H_SELECTORS = [AnnotationSelector("ab_label", loop_type) for loop_type in H_SEGMENTS]
L_SELECTORS = [AnnotationSelector("ab_label", loop_type) for loop_type in L_SEGMENTS]

def process_one_row(idx: int):
    entry = meta_df.loc[idx]        
    annotation_categories = ["aho_res_id", "aho_ins_code", "ab_label"]
    structure = load_annotated_structure(entry, ann_df[ann_df.structure_fname == entry.ab_fname], annotation_categories)
    chains = []
    for chain_id, selectors, segments in [("H", H_SELECTORS, H_SEGMENTS), ("L", L_SELECTORS, L_SEGMENTS)]:
        if entry.is_vhh:
            if chain_id == "L":
                continue
        sequences = [sequence(selector.extract(structure)) for selector in selectors]
        curr_seq = ""
        curr_chain = {
            'sabdab_id': entry.sabdab_id,
            'ab_fname': entry.ab_fname,
            'chain_id': chain_id,
        }
        for segment, seq in zip(segments, sequences):
            start = len(curr_seq)
            curr_seq += seq
            end = len(curr_seq)
            if 'FW' in segment:
                segment_name = segment[1:]
            else:
                segment_name = "CDR" + segment[1:]
            curr_chain[f"{segment_name}_start"] = start
            curr_chain[f"{segment_name}_end"] = end
        curr_chain["sequence"] = curr_seq
        chains.append(curr_chain)
    return chains


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata_parquet", type=str, required=True)
    parser.add_argument("--annotations_parquet", type=str, required=True)
    parser.add_argument("--ncpu", type=int, default=1)
    args = parser.parse_args()

    meta_df = pd.read_parquet(args.metadata_parquet)
    ann_df = pd.read_parquet(args.annotations_parquet)
 
    chains = multiprocess(
        process_one_row,
        meta_df.index,
        ncpu=16,
    )
    chains = [loop for sublist in chains for loop in sublist]
    chains_df = pd.DataFrame(chains)
    chains_df = chains_df[chains_df['sequence'].str.len() > 0]  # Filter out empty sequences

    outfname = args.metadata_parquet.replace("metadata.parquet", "chains.parquet")
    chains_df.to_parquet(outfname, index=False)
    print(f"Saved chains ({len(chains_df)}) to {outfname}")