import pandas as pd
import sys
import numpy as np
import sys
import os
import random
random.seed(42)

# sys.path.append(os.path.join(os.path.dirname(__file__), "../process_data"))
# from process_dihedrals import get_loop_regions

# df = pd.read_parquet("/data2/fanga5/paired_OAS/paired_OAS_index.parquet")
# df = get_loop_regions(df, aho_light_key='fv_light_aho', aho_heavy_key='fv_heavy_aho')
# df.to_parquet("/data2/fanga5/paired_OAS/paired_OAS_index_with_loops.parquet", index=False)

df = pd.read_parquet("/data2/fanga5/paired_OAS/igloo/input/paired_OAS_loops_data_with_angles.parquet")

mask = np.ones(len(df), dtype=bool)
for loop_section in ["LFW1", "L1", "LFW2", "L2", "LFW3", "L3", "LFW4", "L4", "LFW5", "HFW1", "H1", "HFW2", "H2", "HFW3", "H3", "HFW4", "H4", "HFW5"]:
    if "FW" in loop_section:
        min_loop_length = 5
    else:
        min_loop_length = 1
    empty_loops = df[loop_section].str.len() < min_loop_length
    num_empty_loops = empty_loops.sum()
    print(f"{loop_section} empty loops: {num_empty_loops} ({num_empty_loops / len(df) * 100:.2f}%)")
    mask[empty_loops] = False
print(f"Total valid sequences: {mask.sum()}, {mask.sum() / len(df) * 100:.2f}%")

df = df[mask].reset_index(drop=True).copy()

df['is_rep_light'] = df.seqid == df.clusters_light_seqid09_cov08
df['is_rep_heavy'] = df.seqid == df.clusters_heavy_seqid09_cov08

light_indexes = list(range(df[df['is_rep_light']]['clusters_light_seqid09_cov08'].nunique()))
heavy_indexes = list(range(df[df['is_rep_heavy']]['clusters_heavy_seqid09_cov08'].nunique()))

random.shuffle(light_indexes)
random.shuffle(heavy_indexes)
light_val_indexes = light_indexes[:10000]
heavy_val_indexes = heavy_indexes[:20000]
light_test_indexes = light_indexes[10000:20000]
heavy_test_indexes = heavy_indexes[20000:40000]
light_train_indexes = light_indexes[20000:]
heavy_train_indexes = heavy_indexes[40000:]

light_clusters = df[df['is_rep_light']]['clusters_light_seqid09_cov08'].unique()
heavy_clusters = df[df['is_rep_heavy']]['clusters_heavy_seqid09_cov08'].unique()
print("Total light clusters:", len(light_clusters), "Total heavy clusters:", len(heavy_clusters))

light_clusters_val = light_clusters[light_val_indexes]
heavy_clusters_val = heavy_clusters[heavy_val_indexes]
light_clusters_test = light_clusters[light_test_indexes]
heavy_clusters_test = heavy_clusters[heavy_test_indexes]
light_clusters_train = light_clusters[light_train_indexes]
heavy_clusters_train = heavy_clusters[heavy_train_indexes]

print("Light chain:", len(light_clusters_train), len(light_clusters_val), len(light_clusters_test))
print("Heavy chain:", len(heavy_clusters_train), len(heavy_clusters_val), len(heavy_clusters_test))

for split in ['train', 'val', 'test']:
    if split == 'train':
        split_df_light = df[df['clusters_light_seqid09_cov08'].isin(eval(f'light_clusters_{split}'))]
        split_df_heavy = df[df['clusters_heavy_seqid09_cov08'].isin(eval(f'heavy_clusters_{split}'))]
    else:
        split_df_light = df[df['is_rep_light'] & df['clusters_light_seqid09_cov08'].isin(eval(f'light_clusters_{split}'))]
        split_df_heavy = df[df['is_rep_heavy'] & df['clusters_heavy_seqid09_cov08'].isin(eval(f'heavy_clusters_{split}'))]
    
    light_chains = split_df_light[['seqid', "LFW1", "L1", "LFW2", "L2", "LFW3", "L4", "LFW4", "L3", "LFW5", "L_angles"]].rename(columns={
        "LFW1": "FW1",
        "L1": "CDR1",
        "LFW2": "FW2",
        "L2": "CDR2",
        "LFW3": "FW3",
        "L4": "CDR4",
        "LFW4": "FW4",
        "L3": "CDR3",
        "LFW5": "FW5",
        "L_angles": "angles",
    })
    heavy_chains = split_df_heavy[['seqid', "HFW1", "H1", "HFW2", "H2", "HFW3", "H4", "HFW4", "H3", "HFW5", "H_angles"]].rename(columns={
        "HFW1": "FW1",
        "H1": "CDR1",
        "HFW2": "FW2",
        "H2": "CDR2",
        "HFW3": "FW3",
        "H4": "CDR4",
        "HFW4": "FW4",
        "H3": "CDR3",
        "HFW5": "FW5",
        "H_angles": "angles",
    })
    df_chains = pd.concat([light_chains, heavy_chains], axis=0)
    print(f"Total chains in {split} split: {len(df_chains)}")
    df_chains.to_parquet(f"/data2/fanga5/paired_OAS/paired_OAS_index_with_loops_and_angles_single_chain_{split}_with_seqid.parquet", index=False)

    if split == 'test':
        light_chains.to_parquet(f"/data2/fanga5/paired_OAS/paired_OAS_index_with_loops_and_angles_light_chain_{split}_with_seqid.parquet", index=False)
        heavy_chains.to_parquet(f"/data2/fanga5/paired_OAS/paired_OAS_index_with_loops_and_angles_heavy_chain_{split}_with_seqid.parquet", index=False)

    print(f"Saved {split} split with {len(df_chains)} chains to /data2/fanga5/paired_OAS/paired_OAS_index_with_loops_and_angles_single_chain_{split}_with_seqid.parquet")