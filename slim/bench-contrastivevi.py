import scanpy as sc
import pandas as pd
from preprocess_calcagno import get_preprocessed_anndata


ROOT_PATH = 'data/GSE214611_RAW'


import time

healthy_celltypes = ['Myh6']
disease_celltypes = ['Ankrd1', 'Xirp2']
imp_celltypes = disease_celltypes + healthy_celltypes

with open('bench-contrastivevi-hvg-5000.csv', 'w') as f:
    print('PROBLEM_KEY', 
                    'accuracy', 'accuracyx', 'accuracy1', 'max(accuracy. accuracyx. accuracy1)',
                    'precision', 'precisionx', 'precision1', 'max(precision. precisionx. precision1)',
                    'recall', 'recallx', 'recall1', 'max(recall1. recallx. recall)',
                    'f1', 'f1x', 'f11', 'max(f1. f1x. f11)',
                    'auc', 'aucx', 'auc1', 'max(auc. aucx. auc1)',
                    'aupr', 'auprx', 'aupr1', 'max(aupr. auprx. aupr1)',
                    'time_taken',  sep=',', file=f, flush=True)

    for PROBLEM_KEY in ['ref_0_que_1hr', 'ref_0_que_4hr', 'ref_0_que_1', 'ref_0_que_3', 'ref_0_que_7',  'ref_0_que_1-7']:
        ad = get_preprocessed_anndata(PROBLEM_KEY, ROOT_PATH).copy()

        print(ad)

        reference_samples = list(ad.obs[ad.obs['dataset'] == 'reference']['sample'].unique())
        query_samples = list(ad.obs[ad.obs['dataset'] == 'query']['sample'].unique())


        for i in range(25):
            print('RUN: ', i)
            seed = i * 100

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
            target_distances = distances[target_indices]
            y_scores = target_distances[:, reference_cluster] - target_distances[:, 1-reference_cluster]
            predicted_labels = clustering_labels[target_indices] != reference_cluster

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

            salient_ad_all = AnnData(
                X=model.get_latent_representation(ad, representation_kind='salient'),
                obs=ad.obs
            )
            sc.pp.pca(salient_ad_all)

            X = salient_ad_all.X

            val = np.sum(np.abs(X), axis=1)

            val_ref = val[ad.obs["dataset"].isin(['reference'])]
            val_que = val[ad.obs["dataset"].isin(['query'])]
            ad_que = ad[ad.obs["dataset"].isin(['query'])]
            val_que_healthy = val_que[ad_que.obs['final_cluster'].isin(healthy_celltypes)]
            val_que_disease = val_que[ad_que.obs['final_cluster'].isin(disease_celltypes)]

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

            print(PROBLEM_KEY, 
                    accuracy, accuracyx, accuracy1, max(accuracy, accuracyx, accuracy1),
                    precision, precisionx, precision1, max(precision, precisionx, precision1),
                    recall, recallx, recall1, max(recall1, recallx, recall),
                    f1, f1x, f11, max(f1, f1x, f11),
                    auc, aucx, auc1, max(auc, aucx, auc1),
                    aupr, auprx, aupr1, max(aupr, auprx, aupr1),
                    time_taken,  sep=',', file=f, flush=True)

