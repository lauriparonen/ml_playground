import matplotlib.pyplot as plt
import numpy as np

def load_word_freqs(path):
    words, freqs = [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            word, freq = line.strip().split()
            words.append(word)
            freqs.append(int(freq))
    return words, np.array(freqs)

words, freqs = load_word_freqs("./results/top_100.txt")

# normalize to get probabilities
total = freqs.sum()
probabilities = freqs / total

# sort by descending frequency
sorted_idx = np.argsort(-probabilities)
sorted_probabilities = probabilities[sorted_idx]
ranks = np.arange(1, len(sorted_probabilities)+1)

def zipf_distribution(V, a):
    ranks = np.arange(1, V + 1)
    weights = 1 / np.power(ranks, a)
    return weights / weights.sum()

plt.figure(figsize=(8,6))

# plot the distribution of the words alone
plt.plot(ranks, sorted_probabilities, label="empirical")

plt.xlabel("rank")
plt.ylabel("frequency (probability)")
plt.title("Empirical distribution (the 100 most frequent words in the corpus)")
plt.legend()
plt.savefig("results/word_distribution_linear.png", dpi=150)
plt.show()


plt.figure(figsize=(8,6))

plt.plot(ranks, sorted_probabilities, label="empirical")

# several a values
for a in [0.8, 1.0, 1.2]:
    plt.plot(ranks, zipf_distribution(len(sorted_probabilities), a), \
               label=f"zipf a={a}")

plt.xlabel("rank")
plt.ylabel("frequency (probability)")
plt.title("Zipf's law vs empirical distribution (linear scale)")
plt.legend()
plt.savefig("results/zipf_linear.png", dpi=150)
plt.show()


plt.figure(figsize=(8,6))

# log-log scale to see the adjacency better
plt.loglog(ranks, sorted_probabilities, label="empirical")

# several a values
for a in [0.8, 1.0, 1.2]:
    plt.loglog(ranks, zipf_distribution(len(sorted_probabilities), a), \
               label=f"zipf a={a}")

plt.xlabel("rank")
plt.ylabel("frequency (probability)")
plt.title("Zipf's law vs empirical distribution (log-log scale)")
plt.legend()
plt.savefig("results/zipf_loglog.png", dpi=150)
plt.show()