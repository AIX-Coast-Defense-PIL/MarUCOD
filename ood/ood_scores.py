import numpy as np

def calc_distance_score(distributions, feat_log, method="euclidean"):
    def cosine_distance(A, B): # inverse of cosine similarity
        return 1-np.dot(A, B)/(np.linalg.norm(A)*np.linalg.norm(B))
    def euclidean_distance(A, B):
        return np.linalg.norm(A-B)
    def mahalanobis_distance(mean, inv_covmat, feat):
        x_mu = feat - mean
        left = np.dot(x_mu, inv_covmat)
        right = np.dot(left, x_mu.T)
        mahal = np.sqrt(right)
        return mahal
    if method == "euclidean": 
        scores = [min([euclidean_distance(dist['mean'], feat) for dist in distributions]) for feat in feat_log]
    elif method == "cosineSim": 
        scores = [min([cosine_distance(dist['mean'], feat) for dist in distributions]) for feat in feat_log]
    elif method == 'mahalanobis':
        scores = [min([mahalanobis_distance(distributions[i]['mean'], distributions[i]['inv_covmat'], feat) for i in range(len(distributions))]) for feat in feat_log]
    else:
        print('Wrong distance score method. Check the name setting.')
    
    return scores

if __name__=='__main__':
    pass