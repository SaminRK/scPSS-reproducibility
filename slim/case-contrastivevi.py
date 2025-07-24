import scanpy as sc
import pandas as pd

ROOT_PATH = 'data/GSE214611_RAW'

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

    ad.var['mt'] = ad.var_names.str.startswith('mt-')  # annotate the group of mitochondrial genes as 'mt'
    sc.pp.calculate_qc_metrics(ad, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
    if plot:
        sc.pl.violin(ad, ['n_genes_by_counts', 'total_counts', 'pct_counts_mt'],
                jitter=0.4, multi_panel=True)
        sc.pl.scatter(ad, x='total_counts', y='pct_counts_mt')
        sc.pl.scatter(ad, x='total_counts', y='n_genes_by_counts')

    ad = ad[ad.obs.pct_counts_mt <= 5, :]
    ad.layers['counts'] = ad.X.copy()
    sc.pp.normalize_total(ad, target_sum=1e4)
    sc.pp.log1p(ad)

    if n_top_genes is not None:
        sc.pp.highly_variable_genes(ad, n_top_genes=n_top_genes)
        ad.raw = ad
        ad = ad[:, ad.var.highly_variable]

    if plot:
        sc.pl.highly_variable_genes(ad)

    return ad

"""## Prepare training and testing dataset"""

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


healthy_celltypes = ['Myh6']
disease_celltypes = ['Ankrd1', 'Xirp2']
imp_celltypes = disease_celltypes + healthy_celltypes

import time


PROBLEM_KEY = 'ref_0_que_1-7'
ad = get_ad(PROBLEM_KEY)

print(ad)

reference_samples = list(ad.obs[ad.obs['dataset'] == 'reference']['sample'].unique())
query_samples = list(ad.obs[ad.obs['dataset'] == 'query']['sample'].unique())
seed = 0

start_time = time.perf_counter()

## ContrastiveVI start

from contrastive_vi.model import ContrastiveVI
from pytorch_lightning.utilities.seed import seed_everything
import scanpy as sc
seed_everything(seed)

ContrastiveVI.setup_anndata(ad, layer="counts")

target_ad = ad[ad.obs["dataset"].isin(['query'])].copy()
background_ad = ad[ad.obs["dataset"].isin(['reference'])].copy()

print(target_ad)
print(background_ad)

model = ContrastiveVI(ad)

import numpy as np

background_indices = np.where(ad.obs["dataset"].isin(['reference']))[0]
target_indices = np.where(ad.obs["dataset"].isin(['query']))[0]

print(target_ad)
print(background_ad)

model.train(
    check_val_every_n_epoch=1,
    train_size=0.8,
    background_indices=background_indices,
    target_indices=target_indices,
    early_stopping=True,
    max_epochs=500,
)

from anndata import AnnData

salient_ad = AnnData(
    X=model.get_latent_representation(ad, representation_kind='salient'),
    obs=ad.obs
)

from sklearn.metrics import adjusted_rand_score
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=2)
kmeans.fit(salient_ad.X)
clustering_labels = kmeans.labels_
reference_clustering_labels = clustering_labels[background_indices]
if np.mean(reference_clustering_labels) > 0.5:
    reference_cluster = 1
else:
    reference_cluster = 0

distances = kmeans.transform(salient_ad.X)
y_scores_ = distances[:, reference_cluster] - distances[:, 1-reference_cluster]
predicted_labels_ = clustering_labels != reference_cluster

salient_ad.obs['variant_1_scores'] = y_scores_

y_scores = y_scores_[target_indices]
predicted_labels = predicted_labels_[target_indices]

## ContrastiveVI end
end_time = time.perf_counter()

true_labels = salient_ad[target_indices].obs['final_cluster'].isin(disease_celltypes).to_numpy().astype(int)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score

accuracy = accuracy_score(true_labels, predicted_labels)
precision = precision_score(true_labels, predicted_labels)
recall = recall_score(true_labels, predicted_labels)
f1 = f1_score(true_labels, predicted_labels)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)

# Calculate AUC
auc = roc_auc_score(true_labels, y_scores)
print("AUC:", auc)

# Calculate AUPR
aupr = average_precision_score(true_labels, y_scores)
print("AUPR:", aupr)

flipped_predicted_labels = 1 - predicted_labels

accuracyx = accuracy_score(true_labels, flipped_predicted_labels)
precisionx = precision_score(true_labels, flipped_predicted_labels)
recallx = recall_score(true_labels, flipped_predicted_labels)
f1x = f1_score(true_labels, flipped_predicted_labels)

print("Accuracy:", accuracyx)
print("Precision:", precisionx)
print("Recall:", recallx)
print("F1-score:", f1x)

flipped_y_scores = -y_scores

# Calculate AUC
aucx = roc_auc_score(true_labels, flipped_y_scores)
print("AUC:", aucx)

# Calculate AUPR
auprx = average_precision_score(true_labels, flipped_y_scores)
print("AUPR:", auprx)

sc.pp.pca(salient_ad)

X = salient_ad.X

val = np.sum(np.abs(X), axis=1)

val_ref = val[ad.obs["dataset"].isin(['reference'])]
val_que = val[ad.obs["dataset"].isin(['query'])]
ad_que = ad[ad.obs["dataset"].isin(['query'])]
val_que_healthy = val_que[ad_que.obs['final_cluster'].isin(healthy_celltypes)]
val_que_disease = val_que[ad_que.obs['final_cluster'].isin(disease_celltypes)]

salient_ad.obs['variant_2_scores'] = val

thres = np.percentile(val_ref, 95)
predicted_labels = (val_que > thres).astype(int)
true_labels = ad_que.obs['final_cluster'].isin(disease_celltypes).to_numpy().astype(int)

accuracy1 = accuracy_score(true_labels, predicted_labels)
precision1 = precision_score(true_labels, predicted_labels)
recall1 = recall_score(true_labels, predicted_labels)
f11 = f1_score(true_labels, predicted_labels)

y_scores = val_que

# Calculate AUC
auc1 = roc_auc_score(true_labels, y_scores)
print("AUC:", auc1)

# Calculate AUPR
aupr1 = average_precision_score(true_labels, y_scores)
print("AUPR:", aupr1)


time_taken = end_time - start_time

import pandas as pd

# Select the columns you want to export
selected_columns = ['variant_1_scores', 'variant_2_scores']  # replace with your actual column names

# Save to CSV
salient_ad.obs[selected_columns].to_csv("case-contrastivevi.csv")

