import numpy as np
from scipy.stats import entropy
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from itertools import product

def mutual_information(x, y, bins_x=10, bins_y=10):
    """
    Calculate mutual information between two variables using histogram-based binning.
    Args:
        x (np.ndarray): Input variable 1.
        y (np.ndarray): Input variable 2.
        bins_x (int): Number of bins for discretizing variable x.
        bins_y (int): Number of bins for discretizing variable y.

    Returns:
        float: Estimated mutual information.
    """
    # Discretize both variables into bins
    c_xy = np.histogram2d(x, y, bins=[bins_x, bins_y])[0]
    
    # Joint probability distribution
    p_xy = c_xy / c_xy.sum()
    
    # Marginal distributions
    p_x = p_xy.sum(axis=1)
    p_y = p_xy.sum(axis=0)
    
    # Mutual Information calculation
    mi = 0.0
    for i in range(bins_x):
        for j in range(bins_y):
            if p_xy[i, j] > 0:
                mi += p_xy[i, j] * np.log(p_xy[i, j] / (p_x[i] * p_y[j]))
    return mi

def mutual_information_embedding(x, y, bin_combinations=[(5, 5), (10, 10), (15, 15)]):
    """
    Generate an embedding of mutual information values for different binning schemes.
    Args:
        x (np.ndarray): Input variable 1.
        y (np.ndarray): Input variable 2.
        bin_combinations (list of tuples): List of (bins_x, bins_y) combinations.

    Returns:
        np.ndarray: Embedding of mutual information values for the given bin combinations.
    """
    embedding = []
    for bins_x, bins_y in bin_combinations:
        mi = mutual_information(x, y, bins_x=bins_x, bins_y=bins_y)
        embedding.append(mi)
    return np.array(embedding)

def visualize_embeddings(embeddings):
    """
    Visualize the mutual information embeddings using PCA.
    Args:
        embeddings (list of np.ndarray): List of mutual information embeddings for different datasets.
    """
    pca = PCA(n_components=2)
    reduced_embeddings = pca.fit_transform(embeddings)
    
    plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1])
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title("PCA of Mutual Information Embeddings")
    plt.show()

# Test the embedding and PCA visualization
def test_embedding_with_pca():
    # Generate multiple embeddings for demonstration
    embeddings = []
    for _ in range(5):  # Generate embeddings for 5 synthetic datasets
        x = np.random.normal(size=1000)
        y = 2 * x + np.random.normal(size=1000)
        embedding = mutual_information_embedding(x, y)
        embeddings.append(embedding)
    
    embeddings = np.array(embeddings)
    print("Mutual Information Embeddings:\n", embeddings)
    
    # Visualize embeddings using PCA
    visualize_embeddings(embeddings)

if __name__ == '__main__':
    test_embedding_with_pca()
