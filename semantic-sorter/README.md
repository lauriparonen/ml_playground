# semantic-sorter v0
Barebones NLP pipeline that semantically clusters Obsidian vaults

## pipeline
- tokenize markdown files
- build vocabulary + tf-idf vectors
- cosine similarity + k-means clustering
- PCA 2D projection for visualization

## usage
1. add your Obsidian vault path to `VAULT` in main.py
2. run `python main.py`

## notes
- This version (v0) uses hand-rolled tf-idf + k-means to show the geometry of text similarity; mostly for educational purposes
- A future version (v1) will use transformer embeddings and have an interactive UI for actually exploring semantic adjacency in Obsidian vaults; a kind of emergent cartography of mind for haphazard notetakers such as myself

## technical overview
This project builds a low-level semantic clustering pipeline using only Numpy and basic Python utilities. It parses all markdown files in an Obsidian vault, converts their text into numerical vectors, and groups semantically related notes based on geometric proximity in that vector space.

**TF–IDF vectorization**  
For each document (note):
- **term frequency (tf)** = how often a word appears in that document, usually normalized by document length  
- **document frequency (df)** = how many *different documents* contain that word at least once  
- **inverse document frequency (idf)** = `log((1 + N) / (1 + df)) + 1`, where *N* is the total number of documents  

The tf–idf value for a word is `tf * idf`.  
This assigns high weight to words that are frequent *within* a document but rare *across* the corpus, highlighting words that distinguish documents from one another.

Each document thus becomes a vector of tf–idf scores across the shared vocabulary, forming a high-dimensional “semantic” space.

#### Cosine similarity
Similarity between notes is measured using the cosine of the angle between their tf–idf vectors.

In high-dimensional text data, cosine similarity treats each document vector as a *direction* from the origin rather than a point with magnitude. Since tf–idf vectors can vary widely in length (depending on document size), comparing raw Euclidean distance mostly reflects document length, not content. Cosine similarity fixes this by normalizing every vector to the unit sphere; it measures *angle* between vectors instead of straight-line distance.

Geometrically, think of all documents lying on a hypersphere’s surface. Euclidean distance would draw a straight line between two points, "cutting through" the sphere and thereby producing a distorted measurement of distance (as it ignores the surface). Cosine similarity, on the other hand, measures how far you’d have to *travel along the surface* — the angular separation — which better preserves their relational geometry when the surface itself encodes meaning. In simpler terms: if the vectors point to the same direction (their shared angle is small), the documents are near each other in the vector space, which suggests they are semantically proximate.

Formally:  
$$
\frac{A \cdot B}{||A|| \ ||B||}
$$

This yields a score in $[-1, 1]$, where −1 means opposite orientation, 0 means orthogonal (unrelated), and 1 means identical orientation.

#### k-means
k-means iteratively assigns each document to one of *k* clusters based on proximity to cluster centroids, then updates each centroid to be the mean vector of its members.  
The process repeats until assignments stabilize or movement falls below a threshold.  
It’s used here to group notes with similar tf–idf profiles and reveal emergent topical structure.

#### PCA visualization
Principal Component Analysis (PCA) projects the high-dimensional tf–idf vectors into two principal components that capture the greatest variance.  
This 2D projection exposes clusters and approximate “semantic neighborhoods” within the vault.
