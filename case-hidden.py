import os
import sys
root_path = os.path.abspath('LabelCorrection')
sys.path.insert(0, root_path )

import itertools
import functools
from tqdm import tqdm

import pandas as pd
import numpy as np
import sklearn
import sklearn.linear_model
import sklearn.cluster
import sklearn.metrics
import matplotlib.pyplot as plt
import seaborn as sns

import hiddensc
from hiddensc import utils, files, vis

import scanpy as sc
import scvi
import anndata

utils.print_module_versions([sc, anndata, scvi, hiddensc])
vis.visual_settings()

from sklearn.exceptions import ConvergenceWarning
import warnings
warnings.simplefilter("ignore", category=ConvergenceWarning)

ROOT_PATH = 'data/GSE214611_RAW'

import scanpy as sc
import pandas as pd

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
utils.set_random_seed(seed)

adata = get_ad(PROBLEM_KEY)
adata.obs['binary_label'] = adata.obs['dataset']=='query'
hiddensc.datasets.preprocess_data(adata)
hiddensc.datasets.normalize_and_log(adata)

at_figure_dir = functools.partial(os.path.join, root_path, 'figures', 'tutorial')
os.makedirs(at_figure_dir(), exist_ok=True)
at_results_dir = functools.partial(os.path.join, root_path, files.RESULT_DIR, 'tutorial')
os.makedirs(at_results_dir(), exist_ok=True)
at_train_dir = functools.partial(os.path.join, root_path, files.RESULT_DIR, 'tutorial', 'training')
os.makedirs(at_train_dir(), exist_ok=True)

print(f'Generating results at {at_results_dir()}')

hiddensc.datasets.augment_for_analysis(adata)

num_pcs, ks, ks_pval = hiddensc.models.determine_pcs_heuristic_ks(adata=adata, orig_label="binary_label", max_pcs=60)

optimal_num_pcs_ks = num_pcs[np.argmax(ks)]

feats = {}
x_pca = hiddensc.models.get_pca(adata, n_comps=optimal_num_pcs_ks)
feats['PCA'] = x_pca

n_epochs = 250

model_classes = [scvi.model.LinearSCVI, scvi.model.SCVI]
ks = [10]#, 20, 30, 40, 50]
combos = list(itertools.product(model_classes, ks))

for model_cls, k in tqdm(combos):
    local_adata = adata.copy()
    name = f'{model_cls.__name__}_{k}'
    model_cls.setup_anndata(local_adata, layer="counts")
    model = model_cls(local_adata, n_latent=k)
    model.train(max_epochs=n_epochs, plan_kwargs={"lr": 5e-3}, check_val_every_n_epoch=5)
    train_elbo = model.history["elbo_train"][1:]
    test_elbo = model.history["elbo_validation"]
    ax = train_elbo.plot()
    test_elbo.plot(ax=ax)
    plt.yscale('log')
    #plt.savefig(at_train_dir(f'{name}.png'))
    plt.title(name)
    feats[name] = model.get_latent_representation()
    plt.show()
    del local_adata

fname = at_results_dir('features.npz')
np.savez_compressed(fname, **feats)

feats = files.load_npz(at_results_dir('features.npz'))
y = (adata.obs['dataset'].values == 'query').astype(np.int32)
ids = adata.obs.index.values
pred_fns = {'logistic': hiddensc.models.logistic_predictions,
                'svm': hiddensc.models.svm_predictions}

preds = [y]#, y_true]
info = [('dataset', '','query')]#, ('perturbed', '','Memory B')]
combos = list(itertools.product(feats.keys(), pred_fns.keys()))

for feat_name, strat_name  in tqdm(combos):
    rand_state=0
    x = feats[feat_name]
    p_hat, p_labels = pred_fns[strat_name](x, y, 1, rand_state)
    preds.append(p_hat)
    info.append((feat_name, strat_name, 'p_hat'))
    preds.append(p_labels)
    info.append((feat_name, strat_name, 'p_label'))

cols = pd.MultiIndex.from_tuples(info)
pred_df = pd.DataFrame(np.array(preds).T, index=adata.obs.index, columns=cols)
pred_df.to_csv(at_results_dir('predictions.csv'))

DIM_RED = 'PCA'
PRED_MODEL = 'logistic'

df = sc.get.obs_df(adata, ['dataset', 'binary_label'])
df.columns = ['dataset', 'binary_label']
df['p_hat'] = pred_df[f'{DIM_RED}'][f'{PRED_MODEL}']['p_hat'].values
df['new_label'] = pred_df[f'{DIM_RED}'][f'{PRED_MODEL}']['p_label'].values
df[' '] = 1

conditions = [
    (df['binary_label']==0),
    (df['binary_label']==1) & (df['new_label']==0),
    (df['binary_label']==1) & (df['new_label']==1)
]
values = ['Control_L0', 'Case_L0', 'Case_L1']
def label_row(row):
    if row['binary_label'] == 0:
        return 'Control_L0'
    elif row['binary_label'] == 1 and row['new_label'] == 0:
        return 'Case_L0'
    elif row['binary_label'] == 1 and row['new_label'] == 1:
        return 'Case_L1'
    else:
        return 'unknown'

df['three_labels'] = df.apply(label_row, axis=1)
adata.obs['new_label'] = pred_df[f'{DIM_RED}'][f'{PRED_MODEL}']['p_label'].values
adata.obs['three_labels'] = df['three_labels']
adata.obs['hidden_score'] = df['p_hat']

ad_que = adata[adata.obs['dataset'] == 'query']

true_labels = ad_que.obs['zone'].isin(['BZ1', 'BZ2']).to_numpy()
predicted_labels = ad_que.obs['three_labels'].isin(['Case_L1']).to_numpy()

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

y_true = true_labels
y_scores = ad_que.obs['hidden_score']

# Calculate AUC
auc = roc_auc_score(y_true, y_scores)
print("AUC:", auc)

# Calculate AUPR
aupr = average_precision_score(y_true, y_scores)
print("AUPR:", aupr)

import pandas as pd

# Assuming adata is your AnnData object
columns_to_export = ["hidden_score"]  # Replace with actual column names
df = adata.obs[columns_to_export]

# Save as CSV
df.to_csv("case-hidden-0.csv")

