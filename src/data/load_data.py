"""Load the raw diabetes dataset from disk."""

from pathlib import Path

import pandas as pd
import yaml


def load_config(config_path: str = "config.yaml") -> dict:
    """Load project configuration from a YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_data(config: dict) -> pd.DataFrame:
    """Load the raw diabetes dataset as described in config['data']."""
    raw_dir = Path(config["data"]["raw"])
    filename = config["data"]["raw_filename"]
    filepath = raw_dir / filename

    if not filepath.exists():
        raise FileNotFoundError(
            f"Expected dataset at {filepath}. "
            "Place diabetes.csv in data/raw/ before running."
        )

    return pd.read_csv(filepath)
