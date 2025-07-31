import pandas as pd
import numpy as np
import importlib
import ridge_regression_task
importlib.reload(ridge_regression_task)
from ridge_regression_task import train_regression_model
from sklearn.preprocessing import StandardScaler
import argparse

parser = argparse.ArgumentParser(description="Run regression on AbBiBench data")
parser.add_argument('--item_id', type=str, required=True, help='Item ID for the dataset')
args = parser.parse_args()

item_id = args.item_id

DATA_DIR = "/data2/fanga5/benchmarking_data/AbBiBench"

binding_scores_path = f'{DATA_DIR}/{item_id}_benchmarking_data_loops.csv'
binding_scores_df = pd.read_csv(binding_scores_path)

esmc_path = f"{DATA_DIR}/ESMC/{item_id}_esmc_embeddings.npy"
esmc_embeddings = np.load(esmc_path)

saprot_path = f"{DATA_DIR}/saprot/{item_id}_embeddings.npy"
saprot_embeddings = np.load(saprot_path)

prostt5_path = f"{DATA_DIR}/prostt5/{item_id}_embeddings.npy"
prostt5_embeddings = np.load(prostt5_path)

igbert_embeddings_path = f'{DATA_DIR}/IgBert/baseline/{item_id}_benchmarking_data_igbert.npy'
igbert_embeddings = np.load(igbert_embeddings_path)

ablang2_embeddings_path = f'{DATA_DIR}/ablang2/{item_id}_embeddings.npy'
ablang2_embeddings = np.load(ablang2_embeddings_path)

esm2_embeddings_path = f'{DATA_DIR}/ESM2/{item_id}_embeddings.npy'
esm2_embeddings = np.load(esm2_embeddings_path)

with_igloo_embeddings_path = f'{DATA_DIR}/IgBert/with_igloo_angles_v13/{item_id}_benchmarking_data_igbert.npy'
with_igloo_embeddings = np.load(with_igloo_embeddings_path)

with_igloo_all_angles_embeddings_path = f'{DATA_DIR}/IgBert/igbert_all_angles_v4/{item_id}_benchmarking_data_igbert.npy'
with_igloo_all_angles_embeddings = np.load(with_igloo_all_angles_embeddings_path)

y_full = binding_scores_df['binding_score'].values
y_full = StandardScaler().fit_transform(y_full.reshape(-1, 1)).flatten()

# PCA is not applied on any of the models
print()
print(f"{item_id} with IgBert")
print(f"X_full shape: {igbert_embeddings.shape}, y_full shape: {y_full.shape}")
train_regression_model(igbert_embeddings, y_full, pca_components=igbert_embeddings.shape[1], verbose=False)
print()

print()
print(f"{item_id} with ESMC")
print(f"X_full shape: {esmc_embeddings.shape}, y_full shape: {y_full.shape}")
train_regression_model(esmc_embeddings, y_full, pca_components=igbert_embeddings.shape[1], verbose=False)
print()

print()
print(f"{item_id} with Saprot")
print(f"X_full shape: {saprot_embeddings.shape}, y_full shape: {y_full.shape}")
train_regression_model(saprot_embeddings, y_full, pca_components=igbert_embeddings.shape[1], verbose=False)
print()

print()
print(f"{item_id} with ProstT5")
print(f"X_full shape: {prostt5_embeddings.shape}, y_full shape: {y_full.shape}")
train_regression_model(prostt5_embeddings, y_full, pca_components=igbert_embeddings.shape[1], verbose=False)
print()

print()
print(f"{item_id} with AbLang2")
print(f"X_full shape: {ablang2_embeddings.shape}, y_full shape: {y_full.shape}")
train_regression_model(ablang2_embeddings, y_full, pca_components=igbert_embeddings.shape[1], verbose=False)
print()

print()
print(f"{item_id} with ESM2")
print(f"X_full shape: {esm2_embeddings.shape}, y_full shape: {y_full.shape}")
train_regression_model(esm2_embeddings, y_full, pca_components=esm2_embeddings.shape[1], verbose=False)
print()

print()
print(f"{item_id} with Igloo tokens")
print(f"X_full shape: {with_igloo_embeddings.shape}, y_full shape: {y_full.shape}")
train_regression_model(with_igloo_embeddings, y_full, pca_components=igbert_embeddings.shape[1], verbose=False)
print()

print()
print(f"{item_id} with Igloo all angles v1")
print(f"X_full shape: {with_igloo_all_angles_embeddings.shape}, y_full shape: {y_full.shape}")
train_regression_model(with_igloo_all_angles_embeddings, y_full, pca_components=igbert_embeddings.shape[1], verbose=False)
print()