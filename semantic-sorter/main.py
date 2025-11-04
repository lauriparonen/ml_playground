from text_utils import load_vault_texts, tokenize
from vectorizer import build_vocab, compute_tfidf
from my_clustering import kmeans
from my_inspect import top_keywords_per_cluster
from visualize import pca_2d
import matplotlib.pyplot as plt
import numpy as np

# add your Obsidian vault path here
VAULT = ""

if __name__ == "__main__":
    notes = load_vault_texts(VAULT)
    titles, texts = zip(*notes)
    tokenized = [tokenize(t) for t in texts]

    vocab, word2idx = build_vocab(tokenized)
    tfidf = compute_tfidf(tokenized, word2idx)

    k = int(np.sqrt(len(titles))) or 2
    labels, _ = kmeans(tfidf, k=k)

    for j in range(k):
        print(f"\ncluster {j}")
        for t in np.array(titles)[labels == j]:
            print("  -", t)
    top_keywords_per_cluster(tfidf, labels, vocab)
    
    points_2d = pca_2d(tfidf)
    plt.figure(figsize=(7, 6))
    for k in np.unique(labels):
        cluster_pts = points_2d[labels == k]
        plt.scatter(cluster_pts[:, 0], cluster_pts[:, 1], label=f"cluster {k}", alpha=0.6)
    plt.legend()
    plt.title("vault semantic clusters (tf-idf pca projection)")
    plt.show()