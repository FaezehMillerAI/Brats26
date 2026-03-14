import argparse
import copy
import yaml


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def deep_update(base, updates):
    result = copy.deepcopy(base)
    for k, v in (updates or {}).items():
        if isinstance(v, dict) and isinstance(result.get(k), dict):
            result[k] = deep_update(result[k], v)
        else:
            result[k] = v
    return result


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--split", default=None, help="Override eval split")
    return parser.parse_args()


def load_config(config_path):
    return load_yaml(config_path)
