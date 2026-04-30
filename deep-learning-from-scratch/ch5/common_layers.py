import numpy as np

from common import softmax
from common import cross_entropy_error

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


class Affine:
    """
    Affine (Fully Connected) Layer.

    Performs linear transformation: out = x · W + b
    where:
    - x is input from previous layer (batch of samples)
    - W is weight matrix (learnable parameters)
    - b is bias vector (learnable parameters)

    This layer is the fundamental building block of neural networks,
    connecting all neurons between layers.
    """

    def __init__(self, W, b):
        """
        Initialize Affine layer with weights and bias.

        Args:
            W: Weight matrix of shape (input_dim, output_dim)
               Each column represents weights for one output neuron
            b: Bias vector of shape (output_dim,)
               Bias term added to each output neuron
        """
        # Learnable parameters
        self.W = W  # Weights connecting input to output
        self.b = b  # Bias terms

        # Cache variables for backward pass
        self.x = None  # Store input from forward pass (used to compute gradients)
        self.dW = None  # Gradient of loss with respect to weights
        self.db = None  # Gradient of loss with respect to bias

    def forward(self, x):
        """
        Forward pass: Compute output = x @ W + b

        Args:
            x: Input data of shape (batch_size, input_dim)
               Each row is one training sample

        Returns:
            out: Output of shape (batch_size, output_dim)
                 Each row is the transformed feature vector for one sample

        Mathematical formula:
            out_ij = Σ_k x_ik * W_kj + b_j
        """
        # Cache input for backward pass (needed to compute dW)
        self.x = x

        # Linear transformation: (batch_size, input_dim) @ (input_dim, output_dim)
        # Result shape: (batch_size, output_dim)
        # Add bias broadcasting: bias (output_dim,) is added to each row
        out = np.dot(x, self.W) + self.b
        return out

    def backward(self, dout):
        """
        Backward pass: Compute gradients and propagate error backwards.

        Args:
            dout: Gradient of loss with respect to output
                  Shape: (batch_size, output_dim)

        Returns:
            dx: Gradient of loss with respect to input
                Shape: (batch_size, input_dim)
                Propagated to previous layer

        Gradients computed:
            dW: Gradient for weight update (batch_size, output_dim)
            db: Gradient for bias update (output_dim,)

        Mathematical derivations:
            Let L be loss function.
            Given: dout = ∂L/∂out

            ∂L/∂x = ∂L/∂out · ∂out/∂x = dout · W^T
            ∂L/∂W = x^T · dout
            ∂L/∂b = Σ(dout) over batch dimension (axis=0)
        """
        # Compute gradient with respect to input (for backpropagation to previous layer)
        # dx = dout * W^T
        # Shape: (batch_size, output_dim) @ (output_dim, input_dim) = (batch_size, input_dim)
        dx = np.dot(dout, self.W.T)

        # Compute gradient with respect to weights
        # dW = x^T * dout
        # Shape: (input_dim, batch_size) @ (batch_size, output_dim) = (input_dim, output_dim)
        self.dW = np.dot(self.x.T, dout)

        # Compute gradient with respect to bias
        # db = Σ(dout) over batch samples (axis=0)
        # Shape: (output_dim,)
        # We sum because bias is shared across all samples in the batch
        self.db = np.sum(dout, axis=0)

        return dx


class SoftmaxWithLoss:
    """
    Softmax with Cross-Entropy Loss Layer.

    Combines softmax activation and cross-entropy loss into a single layer
    for numerical stability and computational efficiency.

    This layer is typically used as the final layer in multi-class classification
    networks. It converts logits to probabilities and computes the loss,
    while also providing the gradient for backpropagation.
    """

    def __init__(self):
        """
        Initialize SoftmaxWithLoss layer.

        This layer caches the predicted probabilities and target labels
        for efficient gradient computation during backpropagation.
        """
        # Cache variables for backward pass
        self.y = None  # Predicted probabilities after softmax (shape: batch_size, num_classes)
        self.t = None  # Target labels (one-hot encoded or class indices)
        self.loss = None  # Computed cross-entropy loss value

    def forward(self, x, t):
        """
        Forward pass: Compute softmax probabilities and cross-entropy loss.

        Args:
            x: Input logits from previous layer (raw scores)
               Shape: (batch_size, num_classes)
               These are unnormalized scores before softmax
            t: Target labels
               Shape: (batch_size, num_classes) for one-hot encoding
                     or (batch_size,) for class indices

        Returns:
            loss: Cross-entropy loss value (scalar)

        Mathematical process:
            1. y = softmax(x) → Convert logits to probabilities
               y_i = exp(x_i) / Σ_j exp(x_j)
            2. Loss = -Σ t_i * log(y_i) → Cross-entropy loss
               Lower loss = better predictions
        """
        # Store target labels for backward pass
        self.t = t

        # Apply softmax to convert logits to probabilities
        # Probabilities sum to 1 across classes for each sample
        self.y = softmax(x)

        # Compute cross-entropy loss between predictions and targets
        # This measures how well predictions match true labels
        self.loss = cross_entropy_error(self.y, self.t)

        return self.loss

    def backward(self, dout=1):
        """
        Backward pass: Compute gradient of loss with respect to input.

        Args:
            dout: Gradient from next layer (usually 1 for loss layer)
                  Shape: scalar or (batch_size, num_classes)

        Returns:
            dx: Gradient of loss with respect to input logits
                Shape: (batch_size, num_classes)

        Mathematical derivation:
            For softmax with cross-entropy loss, the gradient simplifies to:
            ∂L/∂x = y - t (where t is one-hot encoded)

            This elegant result combines:
            - Derivative of cross-entropy loss: ∂L/∂y = -t/y
            - Derivative of softmax: ∂y/∂x = y(1-y) for correct class, -y_i*y_j for others

            The combined gradient is simply: (y - t)

            We divide by batch_size to average gradient across samples.

        Why this works:
            The softmax and cross-entropy combination creates a gradient
            that is the difference between predictions and true labels,
            making it perfect for gradient-based optimization.
        """
        # Get batch size (number of samples)
        batch_size = self.t.shape[0]

        # Compute gradient
        # dx = (predicted_probabilities - one_hot_targets) / batch_size
        #
        # For correct class: gradient = y - 1 (negative, pulling probability up)
        # For incorrect classes: gradient = y - 0 (positive, pulling probability down)
        #
        # Division by batch_size averages gradients across all samples

        # There is the wrong implementation.
        # dx = (self.y - self.t) / batch_size # Average gradient

        if self.t.size == self.y.size:
            dx = (self.y - self.t) / batch_size
        else:
            dx = self.y.copy()
            dx[np.arange(batch_size), self.t] -= 1
            dx = dx / batch_size

        return dx
