"""Configuration loading and CLI argument parsing.

Provides a two-layer configuration system:

1. **YAML defaults** — loaded from ``config.yaml`` (or a path given via
   ``--config_file``).  Contains dataset paths, model hyperparameters,
   training schedule, and logging settings.
2. **CLI overrides** — any command-line flag (e.g. ``--lambda_n 0.01``)
   takes precedence over the YAML value for that key.

The main entry point is :func:`get_config`, which returns the merged
configuration dictionary together with the parsed argument namespace.
"""
import argparse
import os
import yaml
from pathlib import Path
from typing import Any, Dict, Tuple


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load YAML configuration with proper type conversion.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Loaded configuration dictionary with proper types
    """
    config_file = Path(config_path)
    if not config_file.exists():
        repo_root = Path(__file__).resolve().parent.parent
        if (repo_root / config_path).exists():
            config_file = repo_root / config_path
        else:
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
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
    parser.add_argument("--lambda_u", type=float, default=None, help="FLOPS lambda_u override for user vectors")
    parser.add_argument("--lambda_n", type=float, default=None, help="FLOPS lambda_n override for candidate news")
    parser.add_argument("--lambda_history", type=float, default=None, help="FLOPS lambda_history override for history news")
    parser.add_argument("--topk_sparsity_queries", type=int, default=None, help="Top-K user query sparsity limit override")
    parser.add_argument("--user_aggregation", type=str, default=None, help="User aggregation variant (attention, average, max)")
    parser.add_argument("--use_position_embeddings", action="store_true", help="Enable learned positional embeddings in user encoder")
    parser.add_argument("--user_sparsity_type", type=str, default=None, choices=["flops", "l1"], help="Regularization type for user vector (flops, l1)")
    parser.add_argument("--user_pathway", type=str, default=None, choices=["both", "sparse", "dense"], help="User encoder pathway ablation (both, sparse, dense)")
    
    # Evaluation specific
    parser.add_argument("--eval_mode", type=str, default="retrieval", choices=["retrieval", "dump_predictions"], help="Evaluation mode")
    parser.add_argument("--split", type=str, default="test", help="Data split to evaluate on (val, test)")
    parser.add_argument("--output_path", type=str, default="predictions.json", help="Path to save predictions")
    
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
    config = load_config(args.config_file)

    if 'global' not in config: config['global'] = {}
    config['global']['dataset'] = args.dataset
    config['global']['model_type'] = args.model_type
    
    if args.device:
        config['global']['device'] = args.device
        
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    if args.epochs:
        config['training']['num_epochs'] = args.epochs
    if args.lr:
        config['training']['learning_rate'] = args.lr
    if args.cold_threshold is not None:
        config['training']['cold_threshold'] = args.cold_threshold
    if args.lambda_u is not None:
        config['models'][args.model_type]['lambda_u'] = args.lambda_u
    if args.lambda_n is not None:
        config['models'][args.model_type]['lambda_n'] = args.lambda_n
    if args.lambda_history is not None:
        config['models'][args.model_type]['lambda_history'] = args.lambda_history
    if args.topk_sparsity_queries is not None:
        config['models'][args.model_type]['topk_sparsity_queries'] = args.topk_sparsity_queries
    if args.user_aggregation is not None:
        agg = args.user_aggregation.lower()
        if agg not in ['attention', 'average', 'weighted_sum', 'mean', 'max', 'max_pooling', 'decay', 'temporal_decay']:
            raise ValueError(f"Invalid user aggregation method: {args.user_aggregation}")
        config['models'][args.model_type]['user_aggregation'] = agg
    if args.use_position_embeddings:
        config['models'][args.model_type]['use_position_embeddings'] = True
    if args.user_sparsity_type is not None:
        config['models'][args.model_type]['user_sparsity_type'] = args.user_sparsity_type
    if args.user_pathway is not None:
        config['models'][args.model_type]['user_pathway'] = args.user_pathway
        
    if 'experiment' not in config:
        config['experiment'] = {}
        
    if args.experiment_name:
        config['experiment']['name'] = args.experiment_name
    elif 'name' not in config['experiment']:
        config['experiment']['name'] = f"{args.model_type}_{args.dataset}_{config['training']['learning_rate']}"
        
    if args.model_type:
        config['experiment']['model'] = args.model_type
        
    # Ensure description exists
    if 'description' not in config['experiment']:
        config['experiment']['description'] = f"Training {config['experiment'].get('model', 'unknown')} on {config['global']['dataset']}"
        
    # Debug mode overrides
    if args.debug:
        config['training']['num_epochs'] = 1
        config['training']['batch_size'] = 2
        config['global']['debug_mode'] = True
        config['experiment']['name'] += "_debug"
        print("!! DEBUG MODE ENABLED !!")

    return config, args