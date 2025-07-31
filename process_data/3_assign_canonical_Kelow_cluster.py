import numpy as np
import pandas as pd
import argparse
from multiprocessing import Pool
from tqdm import tqdm

CLUSTERS = pd.read_parquet("Kelow_cluster_center_dihedrals.parquet")

def get_closest_cluster(
    query_phi_arr: np.ndarray,
    query_psi_arr: np.ndarray,
    query_omega_arr: np.ndarray,
    cdr_type: str,
    return_distance: bool = False
) -> tuple[str, float]:
    assert query_phi_arr.shape == query_psi_arr.shape == query_omega_arr.shape, "phi, psi, and omega arrays must have the same shape"
    length = len(query_phi_arr)

    matching_clusters = CLUSTERS[(CLUSTERS["cdr"] == cdr_type) & (CLUSTERS["length"] == length)]

    if not matching_clusters.empty:
        cluster_to_dist_dict = dict()
        for index in matching_clusters.index.values:
            cluster = matching_clusters.at[index, "cluster"]
            cluster_phi_arr = np.array(matching_clusters.at[index, "phi_arr"])
            cluster_psi_arr = np.array(matching_clusters.at[index, "psi_arr"])
            cluster_omega_arr = np.array(matching_clusters.at[index, "omega_arr"])

            phi_diff = np.mean(2 * (1 - np.cos(query_phi_arr - cluster_phi_arr)))
            psi_diff = np.mean(2 * (1 - np.cos(query_psi_arr - cluster_psi_arr)))
            omega_diff = np.mean(2 * (1 - np.cos(query_omega_arr - cluster_omega_arr)))
            dih_dist = float(np.mean(np.array([phi_diff, psi_diff, omega_diff])))
            cluster_to_dist_dict[cluster] = dih_dist
        
        closest_cluster, min_dih_dist = min(cluster_to_dist_dict.items(), key=lambda x: x[1])
        
        if min_dih_dist > THRESHOLD:
            # If the closest cluster is too far, we assign it to a wildcard cluster
            assigned_cluster = cdr_type.upper() + "-" + str(length) + "-*"
            min_dih_dist = None if not return_distance else min_dih_dist
        else:
            # If the closest cluster is within the threshold, we assign it to that cluster
            assigned_cluster = closest_cluster
            min_dih_dist = min_dih_dist
        return assigned_cluster, min_dih_dist
    else:
        return cdr_type + "-" + str(length) + "-*", None

def process_row(idx):
    row = df.iloc[idx]
    phi_arr = np.array(row["phi"])
    psi_arr = np.array(row["psi"])
    omega_arr = np.array(row["omega"])
    cdr_type = row["loop_type"]
    assigned_cluster, min_dih_dist = get_closest_cluster(phi_arr, psi_arr, omega_arr, cdr_type, return_distance=args.return_distance)

    col1_name = f'assigned_cluster_D={THRESHOLD}' if THRESHOLD != 0.47 else 'assigned_cluster'
    col2_name = f'min_dih_dist_D={THRESHOLD}' if THRESHOLD != 0.47 else 'min_dih_dist'
    return pd.Series({col1_name: assigned_cluster, col2_name: min_dih_dist})

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet_in", type=str, required=True)
    parser.add_argument("--threshold", type=float, default=0.47, help="Distance threshold for cluster assignment")
    parser.add_argument("--return_distance", action="store_true", help="Return the distance to the nearest cluster")
    args = parser.parse_args()
    THRESHOLD = args.threshold
    df = pd.read_parquet(args.parquet_in)

    with Pool(processes=16) as pool:
        clusters_df = list(tqdm(pool.imap(process_row, range(len(df))), total=len(df), desc="Processing loops"))
    clusters_df = pd.DataFrame(clusters_df)
    df = pd.concat([df, clusters_df], axis=1)
    df.to_parquet(args.parquet_in)