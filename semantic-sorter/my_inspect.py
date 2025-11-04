import numpy as np

def top_keywords_per_cluster(tfidf, labels, vocab, topn=10):
    for k in np.unique(labels):
        cluster_vec = tfidf[labels == k].mean(axis=0)
        top_idx = np.argsort(cluster_vec)[::-1][:topn]
        words = [vocab[i] for i in top_idx]
        print(f"\ncluster {k} keywords:", ", ".join(words))