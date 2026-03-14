import argparse
import copy
import yaml

from src.config import load_config, deep_update
from src.train import train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    ab_cfg = load_config(args.config)
    base = load_config(ab_cfg["base_config"])

    for variant in ab_cfg["variants"]:
        cfg = copy.deepcopy(base)
        cfg = deep_update(cfg, {"run": {"name": variant["name"]}})
        cfg = deep_update(cfg, variant)
        train(cfg)


if __name__ == "__main__":
    main()
