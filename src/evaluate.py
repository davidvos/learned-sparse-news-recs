import argparse
import os
import sys
import yaml
import torch
from pathlib import Path

# Add src to path if running from root
sys.path.append(str(Path(__file__).parent))

from evaluator import Evaluator
from utils import set_seeds, get_device, load_model_and_checkpoint

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate News Recommendation Models")
    
    parser.add_argument("--model_type", type=str, default="nrms", 
                        help="Model type (nrms, naml, lstur, lsnr, star, npa)")
    parser.add_argument("--dataset", type=str, default="mind", 
                        help="Dataset type (mind)")
    parser.add_argument("--config_file", type=str, default="config.yaml", 
                        help="Path to configuration file")
    
    parser.add_argument("--experiment_name", type=str, required=True, 
                        help="Experiment name to load checkpoint from")
    
    parser.add_argument("--eval_mode", type=str, default="retrieval", 
                        choices=["retrieval", "dump_predictions"],
                        help="Evaluation mode")
    
    parser.add_argument("--split", type=str, default="test", 
                        help="Data split to evaluate on (val, test)")
    
    parser.add_argument("--cold_threshold", type=int, default=None, 
                        help="Cold start threshold (override config)")
    
    # Overrides
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size override")
    parser.add_argument("--device", type=str, default="auto", help="Device preference")
    parser.add_argument("--output_path", type=str, default="predictions.json", 
                        help="Path to save predictions (for dump_predictions mode)")
    
    return parser.parse_args()

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def run_evaluation(args):
    # Load config
    config = load_config(args.config_file)
    
    # Apply overrides
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    if args.cold_threshold is not None:
        config['training']['cold_threshold'] = args.cold_threshold
    
    # Set seeds and device
    set_seeds(config['global']['seed'])
    device = get_device(args.device)
    print(f"Using device: {device}")
    
    # Determine checkpoint directory
    # Try to find folder matching experiment name in checkpoints
    # Or assume it is passed as experiment_name strictly if it's a path, 
    # but usually it's just the name. 
    # We'll search in ./checkpoints for directory starting with args.experiment_name
    
    base_ckpt_dir = Path("./checkpoints")
    target_ckpt_dir = None
    
    if (base_ckpt_dir / args.experiment_name).exists():
        target_ckpt_dir = str(base_ckpt_dir / args.experiment_name)
    else:
        # fuzzy search
        candidates = [d for d in base_ckpt_dir.iterdir() if d.is_dir() and d.name.startswith(args.experiment_name)]
        if candidates:
            target_ckpt_dir = str(candidates[0]) # distinct enough hopefully
            print(f"Found checkpoint directory: {target_ckpt_dir}")
        else:
             print(f"Warning: Could not find checkpoint directory for {args.experiment_name}. Loading fresh model.")
             target_ckpt_dir = None

    # Load Model
    model, checkpoint_data, step = load_model_and_checkpoint(
        args.model_type, 
        args.dataset, 
        config, 
        device,
        checkpoint_dir=target_ckpt_dir
    )
    
    # Initialize Evaluator
    evaluator = Evaluator(config, device, args.dataset, args.model_type)
    
    cold_threshold = config['training']['cold_threshold']
    
    if args.eval_mode == "retrieval":
        metrics = evaluator.evaluate(
            model, 
            cold_threshold=cold_threshold, 
            split=args.split,
            prefix=f"eval_{args.split}_"
        )
        print("Evaluation Metrics:")
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")
            
    elif args.eval_mode == "dump_predictions":
        # Build vectors first
        batch_size = config['training']['batch_size']
        num_workers = config['training']['num_workers']
        
        news2vector, news2category = evaluator.build_news_vectors(
            model, batch_size, num_workers, cold_threshold, args.split
        )
        
        evaluator.dump_predictions(
            model,
            news2vector,
            news2category,
            cold_threshold,
            args.split,
            args.output_path,
            batch_size,
            num_workers
        )
        print(f"Predictions saved to {args.output_path}")

if __name__ == "__main__":
    args = parse_args()
    run_evaluation(args)
