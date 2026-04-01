import numpy as np


class Relu():
    """
    Rectified Linear Unit (ReLU) activation function.

    Use: Commonly used in hidden layers of deep neural networks.
    - Maps input to output: f(x) = max(0, x)
    - Output range: [0, ∞)

    Advantages:
    - Helps mitigate vanishing gradient problem
    - Computationally efficient (simple threshold operation)
    - Introduces non-linearity while preserving positive values
    """

    def __init__(self):
        # Mask to store which elements were <= 0 during forward pass
        # Used to zero out gradients for those positions during backpropagation
        self.mask = None

    def forward(self, x):
        """
        Forward pass of ReLU activation.

        Args:
            x: Input array of any shape

        Returns:
            Output array with negative values set to 0, positive values unchanged
        """
        # Create mask where input <= 0 (these will become 0 in output)
        self.mask = (x <= 0)
        out = x.copy()
        # Set all negative values to 0
        out[self.mask] = 0
        return out

    def backward(self, dout):
        """
        Backward pass of ReLU activation.

        Args:
            dout: Gradient flowing from the next layer

        Returns:
            Gradient with respect to input, zeroed out for positions where input <= 0
        """
        # Copy the incoming gradient
        dx = dout.copy()
        # Zero out gradients for positions where input was <= 0
        # (ReLU derivative is 0 for x <= 0, 1 for x > 0)
        dx[self.mask] = 0
        return dx


class Sigmoid():
    """
    Sigmoid activation function.

    Use: Often used in output layer for binary classification, or hidden layers
         in older neural networks.
    - Maps input to output: f(x) = 1 / (1 + e^(-x))
    - Output range: (0, 1)
    - Can be interpreted as probability (for binary classification)

    Properties:
    - Squashes input values to a probability-like range between 0 and 1
    - Smooth and differentiable everywhere
    - S-shaped (sigmoidal) curve

    Note: Can suffer from vanishing gradient problem for very large positive
          or negative inputs (gradients approach 0)
    """

    def __init__(self):
        # Store the output from forward pass for use in backward pass
        # This enables efficient gradient calculation without recomputing sigmoid
        self.out = None

    def forward(self, x):
        """
        Forward pass of Sigmoid activation.

        Args:
            x: Input array of any shape (logits or any real-valued numbers)

        Returns:
            Output array after applying sigmoid function, values between 0 and 1
        """
        # Compute sigmoid: 1 / (1 + exp(-x))
        # Maps any real number to a value between 0 and 1
        out = 1 / (1 + np.exp(-x))
        # Store output for backward pass
        self.out = out
        return out

    def backward(self, dout):
        """
        Backward pass of Sigmoid activation.

        Args:
            dout: Gradient flowing from the next layer

        Returns:
            Gradient with respect to input
            Formula: dout * sigmoid(x) * (1 - sigmoid(x))

        The derivative is at its maximum when sigmoid(x) = 0.5,
        and approaches 0 as sigmoid(x) approaches 0 or 1.
        """
        # Compute gradient: derivative of sigmoid is sigmoid(x) * (1 - sigmoid(x))
        # Chain rule: multiply by incoming gradient
        dx = dout * self.out * (1 - self.out)
        return dx