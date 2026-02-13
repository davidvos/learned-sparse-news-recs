"""
Simple configuration management for news recommendation training.
Handles YAML loading, inheritance, and command-line argument overrides.
"""
import argparse
import os
import yaml
from pathlib import Path
from typing import Any, Dict, Tuple


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
        # Try checking if it's relative to project root or experiments dir
        repo_root = Path(__file__).resolve().parent.parent
        if (repo_root / config_path).exists():
            config_file = repo_root / config_path
        elif (repo_root / "experiments" / config_path).exists():
            config_file = repo_root / "experiments" / config_path
        else:
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    # Handle inheritance if _base_ is specified
    if '_base_' in config:
        base_path = config_file.parent / config['_base_']
        if not base_path.exists():
            # Try resolving relative to repo root if relative path fails
            repo_root = Path(__file__).resolve().parent.parent
            if (repo_root / config['_base_']).exists():
                base_path = repo_root / config['_base_']
            else:
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


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train News Recommendation Models")
    
    # Core arguments
    parser.add_argument("--model_type", type=str, default="nrms", 
                        help="Model type (nrms, naml, lstur, lsnr, star, npa)")
    parser.add_argument("--dataset", type=str, default="mind", 
                        help="Dataset type (mind)")
    parser.add_argument("--config_file", type=str, default="config.yaml", 
                        help="Path to configuration file. Can be a full path or filename in experiments/")
    
    # Overrides
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size override")
    parser.add_argument("--epochs", type=int, default=None, help="Num epochs override")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate override")
    parser.add_argument("--experiment_name", type=str, default=None, help="Experiment name override")
    parser.add_argument("--device", type=str, default="auto", help="Device preference (auto, cuda, mps, cpu)")
    parser.add_argument("--cold_threshold", type=int, default=None, help="Cold start threshold override")
    
    # Flags
    parser.add_argument("--debug", action="store_true", help="Run in debug mode (minimal data/epochs)")
    
    return parser.parse_args()


def get_config() -> Tuple[Dict[str, Any], argparse.Namespace]:
    """
    Orchestrates configuration loading:
    1. Parse args
    2. Load config file (default or specified)
    3. Apply overrides from args
    
    Returns:
        Tuple of (config_dict, params_namespace)
    """
    args = parse_args()
    
    # Determine config path
    config_path = args.config_file
    
    # If config file is just 'config.yaml' (default) but model_type is specified,
    # check if there's a specific experiment config for this model?
    # No, keep it simple: explicit config_file or use default.
    # However, if experiment_name is provided, we might want to look for that?
    # Let's stick to explicit config_file for now.
    
    try:
        config = load_config(config_path)
    except FileNotFoundError:
        # Fallback: if user provided a config name that exists in experiments/ folder
        if (Path("experiments") / config_path).exists():
             config = load_config(str(Path("experiments") / config_path))
        else:
            raise

    # --- Apply Overrides ---
    
    # Dataset
    config['_dataset'] = args.dataset
    
    # Global settings
    if args.device:
        config['global']['device'] = args.device
        
    # Training settings
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    if args.epochs:
        config['training']['num_epochs'] = args.epochs
    if args.lr:
        config['training']['learning_rate'] = args.lr
    if args.cold_threshold is not None:
        config['training']['cold_threshold'] = args.cold_threshold
        
    # Experiment metadata
    if 'experiment' not in config:
        config['experiment'] = {}
        
    if args.experiment_name:
        config['experiment']['name'] = args.experiment_name
    elif 'name' not in config['experiment']:
        # Auto-generate name if missing
        config['experiment']['name'] = f"{args.model_type}_{args.dataset}_{config['training']['learning_rate']}"
        
    if args.model_type:
        config['experiment']['model'] = args.model_type
        
    # Debug mode overrides
    if args.debug:
        config['training']['num_epochs'] = 1
        config['training']['batch_size'] = 2
        config['experiment']['name'] += "_debug"
        print("!! DEBUG MODE ENABLED !!")

    return config, args