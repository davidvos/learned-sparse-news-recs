"""
Simple configuration management for news recommendation training.
Handles YAML loading and inheritance for experiment-based configs.
"""
import yaml
from pathlib import Path
from typing import Any, Dict


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load YAML configuration with inheritance support and proper type conversion.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Loaded configuration dictionary with proper types
    """
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    # Handle inheritance if _base_ is specified
    if '_base_' in config:
        base_path = config_file.parent / config['_base_']
        if not base_path.exists():
            raise FileNotFoundError(f"Base configuration file not found: {base_path}")
        
        # Load base configuration
        with open(base_path, 'r') as f:
            base_config = yaml.safe_load(f)
        
        # Merge configurations (current config overrides base)
        config = deep_merge(base_config, config)
        # Remove the _base_ key after merging
        config.pop('_base_', None)
    
    # Convert string scientific notation to proper types
    config = convert_scientific_notation(config)
    
    return config


def convert_scientific_notation(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert string scientific notation (e.g., '1e-4') to proper numeric types.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configuration with converted numeric values
    """
    def convert_value(value):
        if isinstance(value, str):
            # Try to convert scientific notation strings to float
            try:
                if 'e' in value.lower() or 'E' in value:
                    return float(value)
            except ValueError:
                pass
        elif isinstance(value, dict):
            return {k: convert_value(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [convert_value(v) for v in value]
        return value
    
    return convert_value(config)


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merge two dictionaries, with override taking precedence."""
    result = base.copy()
    
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    
    return result


def load_experiment_config(experiment_name: str, dataset: str = 'mind') -> Dict[str, Any]:
    """
    Load experiment configuration.
    
    Args:
        experiment_name: Name of the experiment (e.g., 'lsnr_baseline')
        dataset: Dataset to use (default: 'mind')
        
    Returns:
        Configuration dictionary
    """
    config_path = f"./experiments/{experiment_name}.yaml"
    config = load_config(config_path)
    
    # Add dataset info to config for easy access
    config['_dataset'] = dataset
    
    return config