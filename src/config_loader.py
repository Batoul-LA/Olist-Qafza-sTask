import yaml
import os

def load_config(config_path="config/config.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

if __name__ == "__main__":
    config = load_config()
    print(config)
    print("Model path:", config["paths"]["model_file"])