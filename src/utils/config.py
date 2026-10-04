import yaml
from pathlib import Path
import logging
import logging.config

# Project root resolution using pathlib
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"

def load_config(config_name: str = "settings.yaml") -> dict:
    """Loads a YAML configuration file from the config directory."""
    config_path = CONFIG_DIR / config_name
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def setup_logging():
    """Configures the logging dictionary from config/logging.yaml."""
    log_config_path = CONFIG_DIR / "logging.yaml"
    if log_config_path.exists():
        with open(log_config_path, "r") as f:
            log_config = yaml.safe_load(f)
            logging.config.dictConfig(log_config)
    else:
        logging.basicConfig(level=logging.INFO)
