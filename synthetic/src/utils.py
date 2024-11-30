import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, mean_absolute_error

from .logistic_regression import sigmoid


def generate_data(
    train_max_size: int = 8000,
    test_size: int = 10000,
    dim: int = 800,
    mu_x: float = 0.0,
    sigma2_x: float = 0.00025,
    mu_beta: float = 3.0,
    sigma2_beta: float = 16.0,
):
    """
    Generate synthetic data for logistic regression experiments.

    Args:
        train_max_size: Number of training samples.
        test_size: Number of test samples.
        dim: Number of features.
        mu_x: Mean of input features.
        sigma2_x: Variance of input features.
        mu_beta: Mean of coefficients.
        sigma2_beta: Variance of coefficients.

    Returns:
        X_train, y_train, X_test, y_test, beta (true coefficients).
    """

    n = train_max_size + test_size

    X = mu_x + np.random.randn(n, dim) * np.sqrt(sigma2_x)
    beta = mu_beta + np.random.randn(dim) * np.sqrt(sigma2_beta)

    logits = X @ beta
    probs = sigmoid(logits)
    y = np.random.binomial(1, probs)

    X_train, y_train = X[:train_max_size], y[:train_max_size]
    X_test, y_test = X[-test_size:], y[-test_size:]

    return X_train, y_train, X_test, y_test, beta


def compute_ece(predictions, labels, n_bins=10, strategy='uniform'):
    """
    Compute the Expected Calibration Error (ECE).

    Args:
        predictions: Predicted probabilities (n_samples,).
        labels: True labels (n_samples,).
        n_bins: Number of bins to use (default is 10).
        strategy: Binning strategy ('uniform' or 'quantile').

    Returns:
        The ECE value.
    """
    if strategy not in ['uniform', 'quantile']:
        raise ValueError("Invalid strategy: choose either 'uniform' or 'quantile'")
    
    if strategy == 'uniform':
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
    elif strategy == 'quantile':
        bin_boundaries = np.percentile(predictions, np.linspace(0, 100, n_bins + 1))
    
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = np.logical_and(predictions > bin_lower, predictions <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        
        if prop_in_bin > 0:
            fraction = np.mean(labels[in_bin])
            avg_prob = np.mean(predictions[in_bin])
            ece += np.abs(fraction - avg_prob) * prop_in_bin
    
    return ece


def calculate_metrics(
    y_true: np.ndarray,
    probs_true: np.ndarray,
    beta_true: np.ndarray,
    probs_pred: np.ndarray,
    beta_pred: np.ndarray
) -> dict:
    """
    Calculate performance metrics.

    Args:
        y_true: True binary labels (n_samples,).
        probs_true: True probabilities (n_samples,).
        beta_true: True coefficients (n_features,).
        probs_pred: Predicted probabilities (n_samples,).
        beta_pred: Predicted coefficients (n_features,).

    Returns:
        Dictionary of metrics.
    """
    mae_prob = mean_absolute_error(probs_true, probs_pred)
    mae_beta = mean_absolute_error(beta_true, beta_pred)
    ece = compute_ece(probs_pred, y_true)
    accuracy = accuracy_score(y_true, probs_pred >= 0.5)
    roc_auc = roc_auc_score(y_true, probs_true)

    return {
        'mae_prob': mae_prob,
        'mae_beta': mae_beta,
        'ece': ece,
        'accuracy': accuracy,
        'roc_auc': roc_auc,
    }
