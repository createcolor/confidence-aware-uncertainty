import numpy as np
import pandas as pd

from pathlib import Path

from logistic_regression import sigmoid, estimate_beta
from utils import generate_data, calculate_metrics


def size_experiment(
    generate_params: dict = {},
    estimate_params: dict = {},
    sample_size_values: list[int] = [125, 250, 500, 1000, 2000, 4000],
    n_runs: int = 10,
    noise_values: list[float] = [0.0, 0.5, 1.0, 1.5],
    seed: int = 42,
    save_path: str = './results',
    file_name: str = 'size_experiment.csv',
):
    results = []

    for i in range(n_runs):
        seed += i

        np.random.seed(seed)

        X_train, y_train, X_test, y_test, beta_true = generate_data(**generate_params)
        probs_test = sigmoid(X_test@beta_true)

        for sample_size in sample_size_values:
            X_train_sample, y_train_sample = X_train[:sample_size], y_train[:sample_size]

            beta_pred = estimate_beta(X_train_sample, y_train_sample, reg_alpha=0.01, **estimate_params)
            probs_pred = sigmoid(X_test @ beta_pred)

            metrics_dict = calculate_metrics(y_test, probs_test, beta_true, probs_pred, beta_pred)
            metrics_dict.update({'seed': seed, 'method': 'labels', 'sample_size': sample_size})

            results.append(metrics_dict)

            for noise in noise_values:
                prob_corrupted = sigmoid(X_train_sample@beta_true + noise*np.random.randn(sample_size)) 
                
                beta_pred = estimate_beta(X_train_sample, prob_corrupted, reg_alpha=0.0, **estimate_params)
                probs_pred = sigmoid(X_test @ beta_pred)

                metrics_dict = calculate_metrics(y_test, probs_test, beta_true, probs_pred, beta_pred)
                metrics_dict.update({'seed': seed, 'method': f'noised_probs_{noise}', 'sample_size': sample_size})

                results.append(metrics_dict)

    df = pd.DataFrame(results)

    save_path = Path(save_path)
    save_path.mkdir(exist_ok=True)
    df.to_csv(save_path / file_name, index=False)


def combination_experiment(
    sample_size_values: list[int] = [0, 125, 250, 500, 1000, 2000, 4000],
    n_runs: int = 10,
    noise: float = 0.5,

    generate_params, 
    estimate_params
):

    results = []

    for i in range(n_runs):
        seed += i

        X_train, y_train, X_test, y_test, beta_true = generate_data(seed=seed)
        probs_test = sigmoid(X_test@beta_true)

        for label_size in sample_size_values:
            for prob_size in sample_size_values:
                train_total = label_size + prob_size
                
                beta_init = np.zeros(X_train.shape[1])
                if train_total > 0:
                    X_train_sample, y_train_sample = X_train[:train_total], y_train[:train_total]

                    if label_size > 0:
                        reg_alpha = 0.1 if (prob_size > 0) and (label_size <= 2000) else 0.01

                        beta_pred = estimate_beta(X_train_sample, y_train_sample, beta_init, reg_alpha)


                    if prob_size > 0:
                        y_train_prob = sigmoid(X_train[label_size:] @ beta + .5*np.random.randn(prob_size))

                        res = minimize(logloss_opt, beta_hat, (X_train[label_size:], y_train_prob, 0.0), method='L-BFGS-B')#, options={'maxiter': 10000, 'maxfun': 10000000})
                        beta_hat = res.x

                    probs_hat = sigmoid(X_test @ beta_hat)

                    results.append({
                        'labels': label_size,
                        'probs':  prob_size,
                        'ECE': compute_ece(probs_hat, y_test),
                        'MAE': mean_absolute_error(sigmoid(X_test @ beta), probs_hat),
                        'seed': seed
                    })

                else:
                    results.append({
                        'labels': label_size,
                        'probs':  prob_size,
                        'ECE': np.nan,
                        'MAE': np.nan,
                    })