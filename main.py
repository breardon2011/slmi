import numpy as np
from scipy.stats import entropy
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from itertools import product

def mutual_information(x, y, bins=10):
    """
    Calculate mutual information between two variables using histogram-based binning.
    Args:
        x (np.ndarray): Input variable 1.
        y (np.ndarray): Input variable 2.
        bins (int): Number of bins for discretizing the variables.

    Returns:
        float: Estimated mutual information.
    """
    # Discretize both variables into bins
    c_xy = np.histogram2d(x, y, bins=bins)[0]
    
    # Joint probability distribution
    p_xy = c_xy / c_xy.sum()
    
    # Marginal distributions
    p_x = p_xy.sum(axis=1)
    p_y = p_xy.sum(axis=0)
    
    # Mutual Information calculation
    mi = 0.0
    for i in range(bins):
        for j in range(bins):
            if p_xy[i, j] > 0:
                mi += p_xy[i, j] * np.log(p_xy[i, j] / (p_x[i] * p_y[j]))
    return mi

# Test mutual information function with sample data
x = np.random.normal(size=1000)
y = 2 * x + np.random.normal(size=1000)
print("Mutual Information:", mutual_information(x, y))



def sliding_window_mutual_information(x, y, window_size=100, bins=10):
    """
    Calculate mutual information across sliding windows.
    Args:
        x (np.ndarray): Input variable 1.
        y (np.ndarray): Input variable 2.
        window_size (int): Size of the sliding window.
        bins (int): Number of bins for discretizing the variables.

    Returns:
        np.ndarray: Array of mutual information values for each window.
    """
    mi_values = []
    num_windows = len(x) - window_size + 1

    for start in range(num_windows):
        end = start + window_size
        x_window = x[start:end]
        y_window = y[start:end]
        mi = mutual_information(x_window, y_window, bins)
        mi_values.append(mi)
    
    return np.array(mi_values)


# Test the sliding window mutual information with sample data
def sliding_window_test():
    window_size = 50
    mi_values = sliding_window_mutual_information(x, y, window_size=window_size)

    # Plot the mutual information values across windows
    plt.plot(mi_values)
    plt.xlabel("Window Start Index")
    plt.ylabel("Mutual Information")
    plt.title(f"Sliding Window Mutual Information (Window Size = {window_size})")
    plt.show()




def mutual_information_gradient(x, y, delta=0.01, window_size=100, bins=10):
    """
    Calculate the gradient of mutual information with respect to `x` using finite differences.
    Args:
        x (np.ndarray): Input variable 1.
        y (np.ndarray): Input variable 2.
        delta (float): Small perturbation added to `x` for gradient approximation.
        window_size (int): Size of the sliding window.
        bins (int): Number of bins for discretizing the variables.

    Returns:
        np.ndarray: Array of mutual information gradient values for each window.
    """
    mi_gradients = []
    num_windows = len(x) - window_size + 1

    for start in range(num_windows):
        end = start + window_size
        x_window = x[start:end]
        y_window = y[start:end]
        
        # Calculate mutual information for original x
        mi_original = mutual_information(x_window, y_window, bins)
        
        # Perturb x slightly and calculate mutual information again
        x_window_perturbed = x_window + delta
        mi_perturbed = mutual_information(x_window_perturbed, y_window, bins)
        
        # Approximate gradient using finite differences
        gradient = (mi_perturbed - mi_original) / delta
        mi_gradients.append(gradient)
    
    return np.array(mi_gradients)

def test_mutual_information_gradient(window_size): 
    # Test the mutual information gradient function with sample data
    x = np.random.normal(size=1000)
    y = 2 * x + np.random.normal(size=1000)
    mi_gradients = mutual_information_gradient(x, y, window_size=window_size)

    # Plot the mutual information gradients across windows
    plt.plot(mi_gradients)
    plt.xlabel("Window Start Index")
    plt.ylabel("Mutual Information Gradient")
    plt.title(f"Mutual Information Gradient (Window Size = {window_size})")
    plt.show()






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
        mi = mutual_information(x, y, bins=(bins_x, bins_y))
        embedding.append(mi)
    return np.array(embedding)



def test_embedding():# Generate embeddings for sample data
    x = np.random.normal(size=1000)
    y = 2 * x + np.random.normal(size=1000)
    
    embedding = mutual_information_embedding(x, y)
    print("Mutual Information Embedding:", embedding)


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

def test_embeddings(): 
# Example embeddings for multiple synthetic datasets (for illustration)
    embeddings = [mutual_information_embedding(np.random.normal(size=1000), 2 * np.random.normal(size=1000)) for _ in range(5)]
    visualize_embeddings(embeddings)





if __name__ == '__main__':
    #mutual_information(x,y)
    #sliding_window_test()
    #test_mutual_information_gradient(100)
    test_embedding()



