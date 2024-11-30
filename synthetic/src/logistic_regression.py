import numpy as np
from scipy.optimize import minimize


def sigmoid(z):
    """
    Compute the sigmoid function.

    Args:
        z: Input array.

    Returns:
        Sigmoid of the input array.
    """
    return 1 / (1 + np.exp(-z))


def logloss(
    beta: np.ndarray, 
    X: np.ndarray, 
    y: np.ndarray, 
    reg_alpha: float = 0.0,
    epsilon: float = 1e-16
) -> float:
    """
    Compute logistic loss with L1 regularization.

    Args:
        beta: Coefficient vector (n_features,).
        X: Feature matrix (n_samples, n_features).
        y: Binary target vector (n_samples,).
        reg_alpha: Regularization strength.
        epsilon: For numerical stability

    Returns:
        The logistic loss with regularization.
    """
    p_hat = sigmoid(X @ beta)
    log_likelihood = -(y * np.log(p_hat + epsilon) + (1 - y) * np.log(1 - p_hat + epsilon))
    return log_likelihood.mean() + reg_alpha * np.abs(beta).mean()


def logloss_with_logits(
    beta: np.ndarray, 
    X: np.ndarray, 
    y: np.ndarray, 
    reg_alpha: float = 0.0
) -> float:
    """
    Compute logistic loss using logits with L1 regularization.

    Args:
        beta: Coefficient vector (n_features,).
        X: Feature matrix (n_samples, n_features).
        y: Binary target vector (n_samples,).
        reg_alpha: Regularization strength.

    Returns:
        The logistic loss with regularization.
    """
    z = X @ beta
    log_likelihood = z * (1 - y) + np.log1p(np.exp(-z))
    return log_likelihood.mean() + reg_alpha * np.abs(beta).mean()


def estimate_beta(
    X: np.ndarray, 
    y: np.ndarray, 
    beta_init: np.ndarray | None = None,
    reg_alpha: float = 0.0, 
    **kwargs
) -> np.ndarray:
    """
    Estimate coefficients for logistic regression using optimization.

    Args:
        X: Feature matrix (n_samples, n_features).
        y: Binary target vector (n_samples,).
        reg_alpha: Regularization strength.
        method: Optimization method to use (default is 'L-BFGS-B').

    Returns:
        Estimated coefficients.
    """
    if beta_init is None:
        beta_init = np.zeros(X.shape[1])

    res = minimize(
        logloss_with_logits,
        beta_init,
        args=(X, y, reg_alpha),
        **kwargs
    )

    if not res.success:
        raise ValueError(f"Optimization failed: {res.message}")

    return res.x
