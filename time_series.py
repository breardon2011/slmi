from embedding import mutual_information_embedding


def time_series_mutual_information_embeddings(data, window_size=100, bin_combinations=[(5, 5), (10, 10), (15, 15)]):
    """
    Generate mutual information embeddings for each pair of variables in a dataset over time using sliding windows.
    Args:
        data (np.ndarray): Time series data of shape (time_steps, num_variables).
        window_size (int): Size of the sliding window.
        bin_combinations (list of tuples): List of (bins_x, bins_y) combinations for MI embeddings.
        
    Returns:
        dict: A dictionary where keys are variable pairs and values are arrays of embeddings over time.
    """
    num_variables = data.shape[1]
    embeddings_over_time = {}

    for i in range(num_variables):
        for j in range(i + 1, num_variables):
            pair_key = (i, j)
            embeddings_over_time[pair_key] = []

            # Slide the window over time steps
            for start in range(data.shape[0] - window_size + 1):
                end = start + window_size
                x_window = data[start:end, i]
                y_window = data[start:end, j]
                
                # Calculate the embedding for this window
                embedding = mutual_information_embedding(x_window, y_window, bin_combinations)
                embeddings_over_time[pair_key].append(embedding)

    return embeddings_over_time




def time_series_mutual_information_gradients(data, window_size=100, delta=0.01, bin_combinations=[(5, 5), (10, 10), (15, 15)]):
    """
    Generate mutual information gradients for each pair of variables in a dataset over time using sliding windows.
    Args:
        data (np.ndarray): Time series data of shape (time_steps, num_variables).
        window_size (int): Size of the sliding window.
        delta (float): Small perturbation for gradient calculation.
        
    Returns:
        dict: A dictionary where keys are variable pairs and values are arrays of gradient values over time.
    """
    num_variables = data.shape[1]
    gradients_over_time = {}

    for i in range(num_variables):
        for j in range(i + 1, num_variables):
            pair_key = (i, j)
            gradients_over_time[pair_key] = []

            for start in range(data.shape[0] - window_size + 1):
                end = start + window_size
                x_window = data[start:end, i]
                y_window = data[start:end, j]

                # Calculate gradients for each bin combination
                gradients = []
                for bins_x, bins_y in bin_combinations:
                    mi_original = mutual_information(x_window, y_window, bins_x=bins_x, bins_y=bins_y)
                    mi_perturbed = mutual_information(x_window + delta, y_window, bins_x=bins_x, bins_y=bins_y)
                    gradient = (mi_perturbed - mi_original) / delta
                    gradients.append(gradient)

                gradients_over_time[pair_key].append(gradients)

    return gradients_over_time
