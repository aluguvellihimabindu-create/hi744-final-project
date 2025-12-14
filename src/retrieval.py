import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def top_k_similar(matrix, ids, k=5):
    sims = cosine_similarity(matrix)
    np.fill_diagonal(sims, -1.0)  # exclude self matches
    results = {}
    for i, pid in enumerate(ids):
        top_idx = np.argsort(sims[i])[::-1][:k]
        results[pid] = [ids[j] for j in top_idx]
    return results
