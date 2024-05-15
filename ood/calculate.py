import numpy as np

def calc_distributions(cluster_type, score_metric, cluster, feats):
    # center(mean)                
    if cluster_type == 'GaussianMixture':
        class_means = cluster.means_
    elif cluster_type == 'HDBSCAN':
        class_means = cluster.centroids_
    elif cluster_type == 'HDBSCANm':
        class_means = cluster.medoids_
    else: # 'KMeans', 'MeanShift', 'AffinityPropagation', 'BisectingKMeans'
        class_means = cluster.cluster_centers_

    # covariance matrix
    if score_metric == 'mahalanobis':
        if cluster_type == 'GaussianMixture':
            covs = cluster.covariances_
        else:
            feats_per_cluster = [[] for centers in class_means]
            for i, feat in enumerate(feats):
                feats_per_cluster[cluster.labels_[i]].append(feat)

            covs = [np.cov(np.transpose(feats)) for feats in feats_per_cluster]
        distributions = [{'mean' : class_means[i], 'cov': cov, 'inv_covmat':np.linalg.inv(cov) } for i, cov in enumerate(covs)]

    else:
        distributions = [{'mean' : class_mean} for class_mean in class_means]

    return distributions
