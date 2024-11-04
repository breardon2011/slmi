import numpy as np
import matplotlib.pyplot as plt

# Test with a non-linear relationship
x = np.linspace(-10, 10, 1000)
y = np.sin(x) + np.random.normal(0, 0.1, size=1000)

def mutual_information(x, y, bins=10):
    c_xy = np.histogram2d(x, y, bins=bins)[0]
    p_xy = c_xy / c_xy.sum()
    p_x = p_xy.sum(axis=1)
    p_y = p_xy.sum(axis=0)
    
    mi = 0.0
    for i in range(bins):
        for j in range(bins):
            if p_xy[i, j] > 0:
                mi += p_xy[i, j] * np.log(p_xy[i, j] / (p_x[i] * p_y[j]))
    return mi

def sliding_window_mutual_information(x, y, window_size=100, bins=10):
    mi_values = []
    num_windows = len(x) - window_size + 1

    for start in range(num_windows):
        end = start + window_size
        x_window = x[start:end]
        y_window = y[start:end]
        mi = mutual_information(x_window, y_window, bins)
        mi_values.append(mi)
    
    return np.array(mi_values)

def mutual_information_gradient(x, y, delta=0.1, window_size=100, bins=10):
    mi_gradients = []
    num_windows = len(x) - window_size + 1

    for start in range(num_windows):
        end = start + window_size
        x_window = x[start:end]
        y_window = y[start:end]
        
        mi_original = mutual_information(x_window, y_window, bins)
        x_window_perturbed = x_window + delta
        mi_perturbed = mutual_information(x_window_perturbed, y_window, bins)
        
        gradient = (mi_perturbed - mi_original) / delta
        mi_gradients.append(gradient)
    
    return np.array(mi_gradients)

def test_with_plots(window_size=100):
    mi_values = sliding_window_mutual_information(x, y, window_size=window_size)
    mi_gradients = mutual_information_gradient(x, y, delta=0.1, window_size=window_size)

    plt.figure(figsize=(12, 6))
    
    # Plot Mutual Information values
    plt.subplot(2, 1, 1)
    plt.plot(mi_values, label="Mutual Information")
    plt.xlabel("Window Start Index")
    plt.ylabel("Mutual Information")
    plt.title(f"Sliding Window Mutual Information (Window Size = {window_size})")
    plt.legend()

    # Plot Mutual Information Gradient
    plt.subplot(2, 1, 2)
    plt.plot(mi_gradients, label="Mutual Information Gradient", color='orange')
    plt.xlabel("Window Start Index")
    plt.ylabel("Mutual Information Gradient")
    plt.title(f"Mutual Information Gradient (Window Size = {window_size})")
    plt.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    test_with_plots(100)
