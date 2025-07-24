import sys

import anndata as ad
import multimil as mtm
import numpy as np
import scanpy as sc
import scvi
import warnings
import traceback

warnings.filterwarnings("ignore")

print("Last run with scvi-tools version:", scvi.__version__)

ROOT_PATH = 'data/GSE214611_RAW'

import pandas as pd
import numpy as np
import scanpy as sc

metadata_sc = pd.read_csv(f'{ROOT_PATH}/sn_wc_object_integrated@meta.data.csv', index_col=0)
metadata_sc_t = metadata_sc.rename(index=lambda x: x.split('_')[0])
metadata_sc_t['obs_names'] = metadata_sc_t.index

def get_anndata(mtx_path_id, metadata_orig_id):
    PATH = f'{ROOT_PATH}/{mtx_path_id}'
    ad = sc.read_10x_mtx(PATH)
    ad.obs = ad.obs.rename(index=lambda x: x.split('-')[0])
    A = metadata_sc_t[metadata_sc_t['orig.ident'] == metadata_orig_id]

    ad.obs = ad.obs.merge(A, left_index=True, right_index=True, how='left')

    return ad

def pp_anndata(ad, n_top_genes=None, plot=False):
    ad.var_names_make_unique()

    sc.pp.filter_cells(ad, min_genes=200)
    sc.pp.filter_genes(ad, min_cells=3)

    ad.var['mt'] = ad.var_names.str.startswith('mt-')
    sc.pp.calculate_qc_metrics(ad, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
    if plot:
        sc.pl.violin(ad, ['n_genes_by_counts', 'total_counts', 'pct_counts_mt'],
                jitter=0.4, multi_panel=True)
        sc.pl.scatter(ad, x='total_counts', y='pct_counts_mt')
        sc.pl.scatter(ad, x='total_counts', y='n_genes_by_counts')

    ad = ad[ad.obs.pct_counts_mt <= 5, :]
    ad.layers['counts'] = ad.X.copy()

    return ad

def get_mtx_path_id_metadata_orig_id(problem_key):
    exp_meta_keys = {
        'ref_0_que_1hr': {
            'reference': [
                ('snd0_1', 'D0'),
                ('snd0_2', 'D0'),
                ('snd0_3', 'NoMI_Nuclei'),
            ],
            'query': [
                ('sn1hr_1', 'sn1_1'),
                ('sn1hr_2', 'sn1_2'),
            ]
        },
        'ref_0_que_4hr': {
            'reference': [
                ('snd0_1', 'D0'),
                ('snd0_2', 'D0'),
                ('snd0_3', 'NoMI_Nuclei'),
            ],
            'query': [
                ('sn4hr', '4HR'),
            ]
        },
        'ref_0_que_1': {
            'reference': [
                ('snd0_1', 'D0'),
                ('snd0_2', 'D0'),
                ('snd0_3', 'NoMI_Nuclei'),
            ],
            'query': [
                ('snd1_1', 'D1_MI'),
                ('snd1_2', 'D1_MI_Hrt_nuclei'),
                ('snd1_3', 'D1_IR30_Hrt_nuclei'),
                ('snd1_4', 'WT_IR30_D1_2'),
                ('snd1_5', 'WT_IR30_D1_3'),
            ]
        },
        'ref_0_que_3': {
            'reference': [
                ('snd0_1', 'D0'),
                ('snd0_2', 'D0'),
                ('snd0_3', 'NoMI_Nuclei'),
            ],
            'query': [
                ('snd3_1', 'WT_IR30_D3'),
                ('snd3_2', 'D3F_M_rep1'),
                ('snd3_3', 'D3F_M_rep2'),
            ]
        },
        'ref_0_que_7': {
            'reference': [
                ('snd0_1', 'D0'),
                ('snd0_2', 'D0'),
                ('snd0_3', 'NoMI_Nuclei'),
            ],
            'query': [
                ('snd7_1', 'snD7_1'),
                ('snd7_2', 'snD7_1'),
                ('snd7_3', 'D5_D7'),
            ]
        },
        'ref_0_que_1-7': {
            'reference': [
                ('snd0_1', 'D0'),
                ('snd0_2', 'D0'),
                ('snd0_3', 'NoMI_Nuclei'),
            ],
            'query': [
                ('sn1hr_1', 'sn1_1'),
                ('sn1hr_2', 'sn1_2'),
                ('sn4hr', '4HR'),
                ('snd1_1', 'D1_MI'),
                ('snd1_2', 'D1_MI_Hrt_nuclei'),
                ('snd1_3', 'D1_IR30_Hrt_nuclei'),
                ('snd1_4', 'WT_IR30_D1_2'),
                ('snd1_5', 'WT_IR30_D1_3'),
                ('snd3_1', 'WT_IR30_D3'),
                ('snd3_2', 'D3F_M_rep1'),
                ('snd3_3', 'D3F_M_rep2'),
                ('snd7_1', 'snD7_1'),
                ('snd7_2', 'snD7_1'),
                ('snd7_3', 'D5_D7'),
            ]
        }
    }

    return exp_meta_keys[problem_key]

import anndata as AD

imp_celltypes = ['Ankrd1', 'Xirp2', 'Myh6']


def get_ad(PROBLEM_KEY, n_top_genes=5000):

    exp_meta_keys = get_mtx_path_id_metadata_orig_id(PROBLEM_KEY)

    adatas = {}

    for key, item in exp_meta_keys.items():
        for i, (mtx_path_id, metadata_orig_id) in enumerate(item):
            ad = pp_anndata(get_anndata(mtx_path_id, metadata_orig_id))
            ad = ad[ad.obs['final_cluster'].isin(imp_celltypes)]
            dataset_id = f"{key}_{i+1}"
            ad.obs['sample'] = dataset_id
            ad.obs['dataset'] = key
            adatas[f"{key}_{i+1}"] = ad

    ad = AD.concat(adatas, label='batch_key')
    ad.obs_names_make_unique()

    cluster_to_zone = {
        "Ankrd1": "BZ1",
        "Xirp2": "BZ2",
        "Myh6": "RZ",
    }

    ad.obs["zone"] = ad.obs["final_cluster"].map(cluster_to_zone)

    if n_top_genes is not None and n_top_genes < ad.shape[1]:
        sc.pp.highly_variable_genes(ad, n_top_genes=n_top_genes)
        ad.raw = ad
        ad = ad[:, ad.var.highly_variable]

    return ad

PROBLEM_KEY = 'ref_0_que_1-7'
seed = 0
scvi.settings.seed = seed

adata = get_ad(PROBLEM_KEY)

print(adata)

adata.obs["disease"] = adata.obs["dataset"]

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


import pandas as pd

# Select the columns you want to export
selected_columns = ['cell_attn']  # replace with your actual column names

# Save to CSV
adata.obs[selected_columns].to_csv("case-multimil.csv")

