import numpy as np

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)

def kmeans(vectors, k=5, max_iter=100):
    n, d = vectors.shape
    centroids = vectors[np.random.choice(n, k, replace=False)]
    labels = np.zeros(n, dtype=int)

    for _ in range(max_iter):
        # assign
        for i, v in enumerate(vectors):
            sims = [cosine_similarity(v, c) for c in centroids]
            labels[i] = int(np.argmax(sims))
        # update
        new_centroids = np.zeros_like(centroids)
        for j in range(k):
            cluster_points = vectors[labels == j]
            if len(cluster_points) > 0:
                new_centroids[j] = np.mean(cluster_points, axis=0)
            else:
                new_centroids[j] = centroids[j]
        # convergence check
        if np.allclose(new_centroids, centroids):
            break
        centroids = new_centroids
    return labels, centroids