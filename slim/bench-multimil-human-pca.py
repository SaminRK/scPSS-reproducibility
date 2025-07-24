import sys

import anndata as ad
import multimil as mtm
import numpy as np
import scanpy as sc
import scvi
import warnings
import traceback
from anndata import AnnData

warnings.filterwarnings("ignore")

print("Last run with scvi-tools version:", scvi.__version__)


import pandas as pd
import numpy as np
import scanpy as sc

## Modify to the source directory of single cell data
DATA_PATH = "data/human-myocardial-infarction.h5ad"

adata = sc.read_h5ad(DATA_PATH)
sc.pp.highly_variable_genes(adata, n_top_genes=5000, subset=True)


import anndata as AD
import numpy as np

def get_ad(PROBLEM_KEY, n_obs=20_000, random_state=100):

    ad = adata[adata.obs["cell_type"] == PROBLEM_KEY]
    sc.pp.subsample(ad, n_obs=n_obs, random_state=random_state)
    ad.obs["dataset"] = ad.obs.apply(
        lambda row: "reference" if row["major_labl"] == "CTRL" else "query",
        axis=1
    )
    if "X_pca" in adata.obsm: del adata.obsm["X_pca"]
    if "X_pca_harmony" in adata.obsm: del adata.obsm["X_pca_harmony"]
    return ad

with open('bench-multimil-human-pca.csv', 'w') as f:

    print('PROBLEM_KEY', 'seed', 'accuracy', 'precision', 'recall', 'f1', 'auc', 'aupr', flush=True, sep=',', file=f)

    for PROBLEM_KEY in ['cardiac endothelial cell', 'immature innate lymphoid cell', 'pericyte', 'fibroblast of cardiac tissue', 'cardiac muscle myoblast', ]:
        completed = 0

        for i in range(50):
            try:
                seed = i * 100
                scvi.settings.seed = seed

                ad_ = get_ad(PROBLEM_KEY, random_state=seed)
                print(ad_)

                sc.pp.pca(ad_)
                latent = ad_.obsm["X_pca"]
                ad = AnnData(X=latent, obs=ad_.obs.copy())

                ad.obs["disease"] = ad.obs["dataset"]

                sample_key = "sample"

                classification_keys = ["disease"]
                z_dim = ad.shape[1]
                categorical_covariate_keys = classification_keys + [sample_key]

                idx = ad.obs[sample_key].sort_values().index
                ad = ad[idx].copy()

                mtm.model.MILClassifier.setup_anndata(
                    ad,
                    categorical_covariate_keys=categorical_covariate_keys,
                )

                mil = mtm.model.MILClassifier(
                    ad,
                    classification=classification_keys,
                    z_dim=z_dim,
                    n_layers_cell_aggregator=2,
                    sample_key=sample_key,
                    class_loss_coef=0.1,
                )

                mil.train(lr=1e-3)

                mil.get_model_output()

                print(ad)

                ad_que = ad[ad.obs['major_labl'].isin(['RZ', 'IZ'])]

                true_labels = ad_que.obs['major_labl'].isin(['IZ']).to_numpy()
                predicted_labels = ad_que.obs['cell_attn'].to_numpy() > 0.1

                from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

                accuracy = accuracy_score(true_labels, predicted_labels)
                precision = precision_score(true_labels, predicted_labels)
                recall = recall_score(true_labels, predicted_labels)
                f1 = f1_score(true_labels, predicted_labels)

                print("Accuracy:", accuracy)
                print("Precision:", precision)
                print("Recall:", recall)
                print("F1-score:", f1)

                from sklearn.metrics import roc_auc_score, average_precision_score
                import matplotlib.pyplot as plt
                from sklearn.metrics import roc_curve, precision_recall_curve

                y_true = true_labels
                y_scores = ad_que.obs['cell_attn']

                # Calculate AUC
                auc = roc_auc_score(y_true, y_scores)
                print("AUC:", auc)

                # Calculate AUPR
                aupr = average_precision_score(y_true, y_scores)
                print("AUPR:", aupr)

                print(PROBLEM_KEY, seed, accuracy, precision, recall, f1, auc, aupr, flush=True, sep=',', file=f)
                completed += 1
                if completed >= 25: break
            except Exception as e:
                traceback.print_exc()
