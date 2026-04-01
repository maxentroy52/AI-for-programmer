import numpy as np


def softmax(x):
    """
    Activation function for output layer (multi-class classification).
    Converts raw scores to probability distribution.
    Output values sum to 1, each between 0 and 1.
    Used with cross-entropy loss for training.
    """
    x_max = np.max(x)
    x_new = x - x_max
    x_new_exp = np.exp(x_new)
    x_new_exp_sum = np.sum(x_new_exp)
    return x_new_exp / x_new_exp_sum


def cross_entropy_error(y, t):
    """
    Cross-entropy loss function for multi-class classification.
    Measures the difference between predicted probabilities (y) and true labels (t).
    Lower values indicate better predictions.

    Formula: Loss = -Σ log(y_i[t_i]) / batch_size

    Parameters:
        y: Predicted probabilities from softmax output layer
        t: True labels (either one-hot encoded or index labels)

    Returns:
        Average cross-entropy loss across the batch
    """
    # Handle single sample case: reshape to batch format (1, n)
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]
    delta = 1e-7  # Small constant to prevent log(0)

    # Convert one-hot encoded labels to index labels if needed
    if t.ndim == 2:
        t = np.argmax(t, axis=1)

    # Extract predicted probabilities for true labels and compute average negative log
    return -np.sum(np.log(y[np.arange(batch_size), t] + delta)) / batch_size