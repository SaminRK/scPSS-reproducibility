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

ROOT_PATH = 'data/GSE214611_RAW'

import pandas as pd
import numpy as np
import scanpy as sc
from preprocess_calcagno import get_preprocessed_anndata


with open('bench-multimil-vae.csv', 'w') as f:

    print('PROBLEM_KEY', 'seed', 'accuracy', 'precision', 'recall', 'f1', 'auc', 'aupr', flush=True, sep=',', file=f)

    for PROBLEM_KEY in ('ref_0_que_1', 'ref_0_que_3', 'ref_0_que_7', 'ref_0_que_1-7', 'ref_0_que_1hr', 'ref_0_que_4hr'):
        completed = 0

        for i in range(50):
#            try:
                seed = i * 100
                scvi.settings.seed = seed

                adata_ = get_preprocessed_anndata(PROBLEM_KEY, ROOT_PATH)

                scvi.model.SCVI.setup_anndata(adata_, layer="counts")
                vae = scvi.model.SCVI(adata_, n_latent=100)
                vae.train(early_stopping=True)

                latent = vae.get_latent_representation()

                adata = AnnData(X=latent, obs=adata_.obs.copy())

                print(adata)

                adata.obs["disease"] = adata.obs["dataset"] == "query"
                print(adata.obs["disease"])
                print(np.mean(adata.obs["disease"].to_numpy()))

                sample_key = "sample"

                classification_keys = ["disease"]
                z_dim = adata.shape[1]
                categorical_covariate_keys = classification_keys + [sample_key]

                idx = adata.obs[sample_key].sort_values().index
                adata = adata[idx].copy()

                mtm.model.MILClassifier.setup_anndata(
                    adata,
                    categorical_covariate_keys=categorical_covariate_keys,
                )


                mil = mtm.model.MILClassifier(
                    adata,
                    classification=classification_keys,
                    z_dim=z_dim,
                    sample_key=sample_key,
                    class_loss_coef=0.1,
                )

                mil.train(lr=1e-3)

                mil.get_model_output()

                print(adata)

                ad_que = adata[adata.obs['dataset'] == 'query']

                true_labels = ad_que.obs['zone'].isin(['BZ1', 'BZ2']).to_numpy()
                threshold_idx = int(len(ad_que) * 0.5)
                threshold_value = sorted(ad_que.obs["cell_attn"])[threshold_idx]
                print(threshold_value)
                predicted_labels = ad_que.obs['cell_attn'].to_numpy() > threshold_value

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
 #           except Exception as e:
  #              traceback.print_exc()
   #             sys.exit(-1)


