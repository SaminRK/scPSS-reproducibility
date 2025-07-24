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

from preprocess_calcagno import get_preprocessed_anndata


with open('bench-hidden.csv', 'w') as f:

    print('PROBLEM_KEY', 'seed', 'accuracy', 'precision', 'recall', 'f1', 'auc', 'aupr', flush=True, sep=',', file=f)

    for PROBLEM_KEY in ('ref_0_que_1hr', 'ref_0_que_4hr', 'ref_0_que_1', 'ref_0_que_3', 'ref_0_que_7', 'ref_0_que_1-7'):
        for i in range(3):
            seed = i * 100
            utils.set_random_seed(seed)

            adata = get_preprocessed_anndata(PROBLEM_KEY, ROOT_PATH)
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
            from sklearn.metrics import roc_curve, precision_recall_curve

            y_true = true_labels
            y_scores = ad_que.obs['hidden_score']

            # Calculate AUC
            auc = roc_auc_score(y_true, y_scores)
            print("AUC:", auc)

            # Calculate AUPR
            aupr = average_precision_score(y_true, y_scores)
            print("AUPR:", aupr)
            print(PROBLEM_KEY, seed, accuracy, precision, recall, f1, auc, aupr, flush=True, sep=',', file=f)
