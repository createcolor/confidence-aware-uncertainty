import hydra
from omegaconf import DictConfig
import numpy as np

from src.logistic_regression import sigmoid, estimate_beta
from src.utils import generate_data, calculate_metrics
from src.experiment import size_experiment


@hydra.main(version_base=None, config_path="configs", config_name="default")
def main(cfg: DictConfig):
    size_experiment(cfg.generate_params, cfg.estimate_params, **cfg.size_experiment)


if __name__ == '__main__':
    main()
