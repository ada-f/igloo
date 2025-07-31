import pandas as pd
from scipy.stats import pearsonr, spearmanr
import wandb
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, log_loss, auc, precision_recall_curve
from sklearn.decomposition import PCA # Import PCA
from tqdm import tqdm
import numpy as np


def train_regression_model(
    X_full, y_full,
    seed=42,
    use_wandb=False,
    pca_components=1024, # New argument for PCA components
    verbose=True,
):
    # --- Hyperparameter Search Space for L2 Regularization (alpha in scikit-learn) ---
    lambda_values = [1, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001, 0] 

    # --- 10-Fold Outer Cross-Validation ---
    kf_outer = KFold(n_splits=10, shuffle=True, random_state=seed)
    
    fold_results = {
        'test_loss': [],
        'test_pearson': [],
        'test_spearman': [],
        'test_r2': [],
    }

    for fold_idx, (train_val_idx, test_idx) in tqdm(enumerate(kf_outer.split(X_full)), total=10, desc="Outer CV Folds"):
        if verbose:
            print(f"--- Outer Fold {fold_idx + 1}/10 ---")
        
        X_train_val_orig, y_train_val = X_full[train_val_idx], y_full[train_val_idx]
        X_test_orig, y_test = X_full[test_idx], y_full[test_idx]

        # --- PCA Section ---
        # Fit PCA only on the training data of the current outer fold
        # If the original dimension is already <= pca_components, skip PCA
        if X_train_val_orig.shape[1] > pca_components:
            pca = PCA(n_components=pca_components, random_state=seed)
            pca.fit(X_train_val_orig) # Fit PCA on the training data only

            # Transform all splits using the fitted PCA
            X_train_val = pca.transform(X_train_val_orig)
            X_test = pca.transform(X_test_orig)
            if verbose:
                print(f"  PCA applied. Original dim: {X_train_val_orig.shape[1]}, Projected dim: {X_train_val.shape[1]}")
        else:
            if verbose:
                print(f"  Original dimension ({X_train_val_orig.shape[1]}) <= PCA components ({pca_components}), skipping PCA.")
            X_train_val = X_train_val_orig
            X_test = X_test_orig

        # --- 5-Fold Inner Cross-Validation for Hyperparameter Selection ---
        kf_inner = KFold(n_splits=5, shuffle=True, random_state=seed)
        
        best_lambda_for_fold = None
        lambda_avg_val_losses = {lam: [] for lam in lambda_values}

        for inner_fold_idx, (train_idx_inner, val_idx_inner) in enumerate(kf_inner.split(X_train_val)):
            X_train_inner, y_train_inner = X_train_val[train_idx_inner], y_train_val[train_idx_inner]
            X_val_inner, y_val_inner = X_train_val[val_idx_inner], y_train_val[val_idx_inner]

            for current_lambda in lambda_values:
                # Use scikit-learn Ridge model
                model_inner = Ridge(alpha=current_lambda, random_state=seed)
                model_inner.fit(X_train_inner, y_train_inner)
                
                y_pred_inner = model_inner.predict(X_val_inner)
                val_loss_inner = mean_squared_error(y_val_inner, y_pred_inner)
                lambda_avg_val_losses[current_lambda].append(val_loss_inner)

        # After inner folds, select the best lambda for this outer fold
        avg_losses_for_lambdas = {lam: np.mean(losses) for lam, losses in lambda_avg_val_losses.items()}
        best_lambda_for_fold = min(avg_losses_for_lambdas, key=avg_losses_for_lambdas.get)
        if verbose:
            print(f"  Best lambda for Outer Fold {fold_idx + 1}: {best_lambda_for_fold:.6f} (Avg Inner Val Loss: {avg_losses_for_lambdas[best_lambda_for_fold]:.4f})")

        # --- Train on full outer train_val data (PCA transformed) with best lambda ---
        if verbose:
            print(f"  Training final model for Outer Fold {fold_idx + 1} with lambda={best_lambda_for_fold:.6f}")
        final_model_outer = Ridge(alpha=best_lambda_for_fold, random_state=seed)
        final_model_outer.fit(X_train_val, y_train_val)
        
        # --- Evaluate on outer test fold (PCA transformed) ---
        y_pred_test = final_model_outer.predict(X_test)
        test_loss = mean_squared_error(y_test, y_pred_test)
        pearson_corr = pearsonr(y_pred_test, y_test)[0]
        spearman_corr = spearmanr(y_pred_test, y_test).correlation
        r2_score_value = r2_score(y_test, y_pred_test)

        if verbose:
            print(f"  Test Loss: {test_loss:.4f}, Pearson: {pearson_corr:.4f}, Spearman: {spearman_corr:.4f}, R2: {r2_score_value:.4f}")

        fold_results['test_loss'].append(test_loss)
        fold_results['test_pearson'].append(pearson_corr)
        fold_results['test_spearman'].append(spearman_corr)
        fold_results['test_r2'].append(r2_score_value)

        if use_wandb:
            wandb.log({
                f'fold_{fold_idx + 1}_test_loss': test_loss,
                f'fold_{fold_idx + 1}_test_pearson_corr': pearson_corr,
                f'fold_{fold_idx + 1}_test_spearman_corr': spearman_corr,
                f'fold_{fold_idx + 1}_test_r2_score': r2_score_value,
                f'fold_{fold_idx + 1}_best_lambda': best_lambda_for_fold
            })
    print("\n--- Cross-Validation Complete ---")
    avg_test_loss = np.mean(fold_results['test_loss'])
    avg_test_pearson = np.mean(fold_results['test_pearson'])
    avg_test_spearman = np.mean(fold_results['test_spearman'])
    avg_test_r2 = np.mean(fold_results['test_r2'])

    print(f"Average Test Loss across 10 folds: {avg_test_loss:.4f}, std: {np.std(fold_results['test_loss']):.4f}")
    print(f"Average R2 Score across 10 folds: {avg_test_r2:.4f}, std: {np.std(fold_results['test_r2']):.4f}")
    print(f"Average Test Pearson Correlation across 10 folds: {avg_test_pearson:.4f}, std: {np.std(fold_results['test_pearson']):.4f}")
    print(f"Average Test Spearman Correlation across 10 folds: {avg_test_spearman:.4f}, std: {np.std(fold_results['test_spearman']):.4f}")

    if use_wandb:
        wandb.log({
            'avg_test_loss': avg_test_loss,
            'avg_test_pearson_corr': avg_test_pearson,
            'avg_test_spearman_corr': avg_test_spearman,
            'avg_test_r2_score': avg_test_r2,
        })

    return pd.DataFrame({
        'metric': ['mse_loss', 'pearson', 'spearman', 'r2'],
        'mean': [avg_test_loss, avg_test_pearson, avg_test_spearman, avg_test_r2],
        'std': [
            np.std(fold_results['test_loss']),
            np.std(fold_results['test_pearson']),
            np.std(fold_results['test_spearman']),
            np.std(fold_results['test_r2'])
        ]
    })

def train_classification_model(
    X_full, y_full,
    seed=42,
    use_wandb=False,
    pca_components=1024,
    verbose=True,
    regularization_coeffs=[100, 10, 1, 0.1, 0.01],
    k_folds=10,
    inner_k_folds=5,
    multinomial=False, # If True, use multinomial logistic regression
):
    # --- Hyperparameter Search Space for C (Inverse of L2 Regularization) ---
    # Smaller C means stronger regularization.
    # We'll use values that are reciprocals of your original lambda_values
    c_values = regularization_coeffs
    # If lambda was 0, C effectively becomes very large (no regularization)
    # Consider adjusting this range based on typical LogisticRegression C values.

    # --- 10-Fold Outer Cross-Validation ---
    kf_outer = KFold(n_splits=k_folds, shuffle=True, random_state=seed)

    fold_results = {
        'test_log_loss': [],
        'test_accuracy': [],
        'test_precision': [],
        'test_recall': [],
        'test_f1': [],
        'test_auc_roc': [],
        'test_auprc': [],
    }

    for fold_idx, (train_val_idx, test_idx) in tqdm(enumerate(kf_outer.split(X_full)), total=k_folds, desc="Outer CV Folds"):
        if verbose:
            print(f"--- Outer Fold {fold_idx + 1}/10 ---")

        X_train_val_orig, y_train_val = X_full[train_val_idx], y_full[train_val_idx]
        X_test_orig, y_test = X_full[test_idx], y_full[test_idx]

        # --- PCA Section ---
        if X_train_val_orig.shape[1] > pca_components:
            pca = PCA(n_components=pca_components, random_state=seed)
            pca.fit(X_train_val_orig)

            X_train_val = pca.transform(X_train_val_orig)
            X_test = pca.transform(X_test_orig)
            if verbose:
                print(f"  PCA applied. Original dim: {X_train_val_orig.shape[1]}, Projected dim: {X_train_val.shape[1]}")
        else:
            if verbose:
                print(f"  Original dimension ({X_train_val_orig.shape[1]}) <= PCA components ({pca_components}), skipping PCA.")
            X_train_val = X_train_val_orig
            X_test = X_test_orig

        if len(c_values) > 1:
            # --- 5-Fold Inner Cross-Validation for Hyperparameter Selection ---
            kf_inner = KFold(n_splits=inner_k_folds, shuffle=True, random_state=seed)

            best_c_for_fold = None
            c_avg_val_losses = {c: [] for c in c_values}

            for inner_fold_idx, (train_idx_inner, val_idx_inner) in enumerate(kf_inner.split(X_train_val)):
                X_train_inner, y_train_inner = X_train_val[train_idx_inner], y_train_val[train_idx_inner]
                X_val_inner, y_val_inner = X_train_val[val_idx_inner], y_train_val[val_idx_inner]

                for current_c in c_values:
                    # Use scikit-learn LogisticRegression model
                    # solver='liblinear' is good for small datasets and L1/L2 regularization
                    # For larger datasets, 'lbfgs' or 'saga' might be better, but ensure max_iter is sufficient.
                    model_inner = LogisticRegression(C=current_c, random_state=seed, solver='liblinear', multi_class='multinomial' if multinomial else 'auto')
                    model_inner.fit(X_train_inner, y_train_inner)

                    # Predict probabilities for log loss
                    y_pred_proba_inner = model_inner.predict_proba(X_val_inner)[:, 1]
                    val_loss_inner = log_loss(y_val_inner, y_pred_proba_inner)
                    c_avg_val_losses[current_c].append(val_loss_inner)

            # After inner folds, select the best C for this outer fold
            avg_losses_for_cs = {c: np.mean(losses) for c, losses in c_avg_val_losses.items()}
            best_c_for_fold = min(avg_losses_for_cs, key=avg_losses_for_cs.get)
            if verbose:
                print(f"  Best C for Outer Fold {fold_idx + 1}: {best_c_for_fold:.6f} (Avg Inner Val Log Loss: {avg_losses_for_cs[best_c_for_fold]:.4f})")
        else:
            # If only one C value is provided, use it directly
            best_c_for_fold = c_values[0]
            if verbose:
                print(f"  Only one C value provided: {best_c_for_fold:.6f}, skipping inner CV.")

        # --- Train on full outer train_val data (PCA transformed) with best C ---
        if verbose:
            print(f"  Training final model for Outer Fold {fold_idx + 1} with C={best_c_for_fold:.6f}")
        final_model_outer = LogisticRegression(C=best_c_for_fold, random_state=seed, solver='liblinear')
        final_model_outer.fit(X_train_val, y_train_val)

        # --- Evaluate on outer test fold (PCA transformed) ---
        y_pred_test_proba = final_model_outer.predict_proba(X_test)[:, 1] # Probabilities for AUC and log loss
        y_pred_test_class = final_model_outer.predict(X_test) # Predicted classes for accuracy, precision, etc.

        test_log_loss = log_loss(y_test, y_pred_test_proba)
        test_accuracy = accuracy_score(y_test, y_pred_test_class)
        if np.all(y_test == 0):
            print("Warning: All test labels are 0, precision and recall will be undefined.")
        test_precision = precision_score(y_test, y_pred_test_class, zero_division=0) # zero_division=0 to handle cases with no positive predictions
        test_recall = recall_score(y_test, y_pred_test_class, zero_division=0)
        test_f1 = f1_score(y_test, y_pred_test_class, zero_division=0)
        test_auc_roc = roc_auc_score(y_test, y_pred_test_proba)
        precision, recall, _ = precision_recall_curve(y_test, y_pred_test_proba)
        test_auprc = auc(recall, precision)

        if verbose:
            print(f"  Test Log Loss: {test_log_loss:.4f}, Accuracy: {test_accuracy:.4f}, Precision: {test_precision:.4f}, Recall: {test_recall:.4f}, F1-Score: {test_f1:.4f}, AUC-ROC: {test_auc_roc:.4f}, AUPRC: {test_auprc:.4f}")

        fold_results['test_log_loss'].append(test_log_loss)
        fold_results['test_accuracy'].append(test_accuracy)
        fold_results['test_precision'].append(test_precision)
        fold_results['test_recall'].append(test_recall)
        fold_results['test_f1'].append(test_f1)
        fold_results['test_auc_roc'].append(test_auc_roc)
        fold_results['test_auprc'].append(test_auprc)

        if use_wandb:
            wandb.log({
                f'fold_{fold_idx + 1}_test_log_loss': test_log_loss,
                f'fold_{fold_idx + 1}_test_accuracy': test_accuracy,
                f'fold_{fold_idx + 1}_test_precision': test_precision,
                f'fold_{fold_idx + 1}_test_recall': test_recall,
                f'fold_{fold_idx + 1}_test_f1_score': test_f1,
                f'fold_{fold_idx + 1}_test_auc_roc': test_auc_roc,
                f'fold_{fold_idx + 1}_best_C': best_c_for_fold,
                f'fold_{fold_idx + 1}_test_auprc': test_auprc
            })
    print("\n--- Cross-Validation Complete ---")
    avg_test_log_loss = np.mean(fold_results['test_log_loss'])
    avg_test_accuracy = np.mean(fold_results['test_accuracy'])
    avg_test_precision = np.mean(fold_results['test_precision'])
    avg_test_recall = np.mean(fold_results['test_recall'])
    avg_test_f1 = np.mean(fold_results['test_f1'])
    avg_test_auc_roc = np.mean(fold_results['test_auc_roc'])
    avg_test_auprc = np.mean(fold_results['test_auprc'])


    print(f"Average Test Log Loss across 10 folds: {avg_test_log_loss:.4f}, std: {np.std(fold_results['test_log_loss']):.4f}")
    print(f"Average Test Accuracy across 10 folds: {avg_test_accuracy:.4f}, std: {np.std(fold_results['test_accuracy']):.4f}")
    print(f"Average Test Precision across 10 folds: {avg_test_precision:.4f}, std: {np.std(fold_results['test_precision']):.4f}")
    print(f"Average Test Recall across 10 folds: {avg_test_recall:.4f}, std: {np.std(fold_results['test_recall']):.4f}")
    print(f"Average Test F1-Score across 10 folds: {avg_test_f1:.4f}, std: {np.std(fold_results['test_f1']):.4f}")
    print(f"Average Test AUC-ROC across 10 folds: {avg_test_auc_roc:.4f}, std: {np.std(fold_results['test_auc_roc']):.4f}")
    print(f"Average Test AUPRC across 10 folds: {avg_test_auprc:.4f}, std: {np.std(fold_results['test_auprc']):.4f}")


    if use_wandb:
        wandb.log({
            'avg_test_log_loss': avg_test_log_loss,
            'avg_test_accuracy': avg_test_accuracy,
            'avg_test_precision': avg_test_precision,
            'avg_test_recall': avg_test_recall,
            'avg_test_f1_score': avg_test_f1,
            'avg_test_auc_roc': avg_test_auc_roc,
            'avg_test_auprc': avg_test_auprc,
        })

    return pd.DataFrame(fold_results)