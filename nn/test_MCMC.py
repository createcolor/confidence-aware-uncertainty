from pathlib import Path
import argparse
import json

from nn.test_classifier import test_nets, get_testloader

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-c', '--config_path', type=Path, \
        help='Path to MCMC classifier test config')
    parser.add_argument('-o', '--output_dir', type=Path, \
        help='Path to output dir')
    
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_args()

    if not args.output_dir.exists():
        args.output_dir.mkdir()

    with open(args.config_path, "r") as f:
        mcmc_config = json.load(f)

    test_config = mcmc_config.copy()
    for epoch in range(mcmc_config["epochs"][0], 
                       mcmc_config["epochs"][1], 
                       mcmc_config["epochs"][2]):
        test_config["epochs"] = epoch
        testloader = get_testloader(test_config)
        nets_outputs = test_nets(test_config, testloader)

        path2outputs = args.output_dir / \
            f"{test_config['nets_names']}_{test_config['epochs']}ep_outputs.json"
        
        with open(path2outputs, "w", encoding="utf-8") as f:
            json.dump(nets_outputs, f, indent=4)