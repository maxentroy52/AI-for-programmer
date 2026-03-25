import numpy as np

def sigmoid(x):
    """
    Activation function for hidden layer.
    Maps input to range (0,1) using: f(x) = 1 / (1 + e^(-x))
    Introduces non-linearity between layers.
    """
    return 1 / (1 + np.exp(-x))

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


def numerical_gradient_nd(f, x):
    """
    Calculate numerical gradient for multi-dimensional arrays using central difference method.
    Perturbs each element individually to approximate partial derivatives.

    Formula: f'(x) ≈ (f(x+h) - f(x-h)) / (2h)

    Parameters:
        f: Function that takes x as parameter and returns loss value
        x: Multi-dimensional array (weights/biases) to calculate gradient for

    Returns:
        Gradient array with same shape as x
    """
    h = 1e-4  # Small perturbation value (0.0001)
    grad = np.zeros_like(x)  # Initialize gradient array with zeros

    # Iterate through each element in the multi-dimensional array
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index  # Get current element's position (e.g., (row, col))
        tmp_val = x[idx]  # Store original value

        # Calculate f(x + h)
        x[idx] = tmp_val + h
        fxh1 = f(x)

        # Calculate f(x - h)
        x[idx] = tmp_val - h
        fxh2 = f(x)

        # Central difference approximation of derivative
        grad[idx] = (fxh1 - fxh2) / (2 * h)

        # Restore original value
        x[idx] = tmp_val
        it.iternext()

    return grad