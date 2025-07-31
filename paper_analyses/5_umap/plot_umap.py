import pandas as pd
import numpy as np
import umap
from tqdm import tqdm

embedding_files = [
    "/data2/fanga5/embeddings/embeddings_test_loop_len_all_seed_42.parquet",
    "/data2/fanga5/embeddings/embeddings_val_loop_len_all_seed_42.parquet",
    "/data2/fanga5/embeddings/embeddings_train_loop_len_all_seed_42.parquet",
]

dfs = []
for fname in embedding_files:
    df = pd.read_parquet(fname)
    dfs.append(df)
df = pd.concat(dfs, ignore_index=True)

df['loop_type'] = df['loop_id'].str.slice(-2,)
embedding_matrix = np.vstack(df['encoded'].values)

df_umap = pd.DataFrame({
    'loop_id': df['loop_id'],
    'loop_type': df['loop_type'],
    'loop_length': df['loop_length'],
})


for n_neighbors in [5, 20, 50]:
    for min_dist in [0.01, 0.1, 0.5]:
        print(f"Running UMAP with n_neighbors={n_neighbors}, min_dist={min_dist}")
        umap_model = umap.UMAP(n_components=2, n_jobs=8, n_neighbors=n_neighbors, min_dist=min_dist)
        embedding_2d = umap_model.fit_transform(embedding_matrix)

        df_umap[f'x_{n_neighbors}_{min_dist}'] = embedding_2d[:, 0]
        df_umap[f'y_{n_neighbors}_{min_dist}'] = embedding_2d[:, 1]

        df_umap.to_parquet(f"/data2/fanga5/embeddings/umap_sabdab_embeddings_{n_neighbors}_{min_dist}.parquet", index=False)
