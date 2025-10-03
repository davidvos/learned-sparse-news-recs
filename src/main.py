#!/usr/bin/env python3
"""
Simplified evaluation interface for news recommendation models.
Just specify a model class and it will evaluate automatically.
"""

import argparse
import sys
import os
import torch

# Add src to path so we can import modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from evaluate import MODEL_REGISTRY
from config import load_experiment_config
from utils import latest_checkpoint, set_seeds, get_device


def load_model_and_checkpoint(model_type, experiment_name, dataset, learning_rate=None, checkpoint_dir=None):
    """Load model and checkpoint, return model and device."""
    # Load configuration
    config = load_experiment_config(experiment_name, dataset)
    
    # Set seeds and device
    set_seeds(config['global']['seed'])
    device = get_device(config['global']['device'])
    
    # Instantiate model
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Invalid model: {model_type}. Available models: {list(MODEL_REGISTRY.keys())}")
    model_class = MODEL_REGISTRY[model_type]
    model = model_class(config['models'][model_type], config['datasets'][dataset], device)
    
    # Determine checkpoint directory
    if checkpoint_dir is not None:
        resolved_ckpt_dir = checkpoint_dir
    else:
        # Use provided learning rate or fall back to config
        lr = learning_rate if learning_rate is not None else config['training']['learning_rate']
        cold_tr = config['training']['cold_threshold']
        checkpoint_name = f"{experiment_name}_{model_type}_{dataset}_cold{cold_tr}_lr{lr}"
        resolved_ckpt_dir = f"./checkpoints/{checkpoint_name}/"
    
    print(f"🔍 Looking for checkpoints in: {resolved_ckpt_dir}", flush=True)
    checkpoint_path = latest_checkpoint(resolved_ckpt_dir)
    if checkpoint_path is None:
        print(f"❌ No checkpoint file found in {resolved_ckpt_dir}!", flush=True)
        print("💡 Available checkpoint directories:", flush=True)
        checkpoints_base = "./checkpoints/"
        if os.path.exists(checkpoints_base):
            for item in os.listdir(checkpoints_base):
                if os.path.isdir(os.path.join(checkpoints_base, item)):
                    print(f"   📁 {item}", flush=True)
        raise FileNotFoundError(f"No checkpoint found for experiment {experiment_name} with dataset {dataset}")
    else:
        print(f"✅ Loading checkpoint: {checkpoint_path}", flush=True)
        # Handle PyTorch 2.6+ weights_only default
        try:
            checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        except Exception as e:
            if "weights_only" in str(e) or "WeightsUnpickler" in str(e):
                print("⚠️  Retrying checkpoint load with weights_only=False (trusted source)", flush=True)
                checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'), weights_only=False)
            else:
                raise e
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"📊 Checkpoint info - Step: {checkpoint.get('step', 'unknown')}, Epoch: {checkpoint.get('epoch', 'unknown')}", flush=True)
    
    model.to(device)
    model.eval()
    
    return model, device, config


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained news recommendation model - just specify the model class!",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py lsnr                    # Evaluate LSNR on MIND dataset
  python main.py nrms --dataset adressa  # Evaluate NRMS on Adressa dataset  
  python main.py naml --split val        # Evaluate NAML on validation set
  python main.py lstur --cold-thresholds 0 1 2  # Evaluate LSTUR on multiple cold thresholds
  python main.py lsnr --batch-size 32    # Evaluate LSNR with custom batch size
  python main.py --list-checkpoints      # List all available checkpoints
        """
    )
    
    parser.add_argument(
        'model', 
        nargs='?',
        choices=['lsnr', 'nrms', 'naml', 'lstur'],
        help='Model class to evaluate (lsnr, nrms, naml, lstur)'
    )
    
    parser.add_argument(
        '--dataset', 
        default='mind',
        choices=['mind', 'adressa', 'ebnerd'],
        help='Dataset to evaluate on (default: mind)'
    )
    
    parser.add_argument(
        '--split',
        default='test',
        choices=['val', 'test'],
        help='Split to evaluate on (default: test)'
    )
    
    parser.add_argument(
        '--cold-thresholds',
        nargs='*',
        type=int,
        default=None,
        help='Cold thresholds to evaluate (default: use experiment config)'
    )
    
    parser.add_argument(
        '--lr',
        type=str,
        default=None,
        help='Learning rate to select checkpoint (e.g., 0.0001 or 1e-05)'
    )
    
    parser.add_argument(
        '--checkpoint-dir',
        default=None,
        help='Explicit checkpoint directory path'
    )
    
    parser.add_argument(
        '--list-checkpoints',
        action='store_true',
        help='List all available checkpoints and exit'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help='Batch size for evaluation (default: use experiment config)'
    )

    parser.add_argument(
        '--regularization',
        type=str,
        default=None,
        help='Regularization to select checkpoint (e.g., baseline, low, high, very_high)'
    )
    
    args = parser.parse_args()
    # Default to evaluating cold thresholds 0..5 if none provided
    if args.cold_thresholds is None:
        args.cold_thresholds = [0, 1, 2, 3, 4, 5]
    
    # Handle list checkpoints option
    if args.list_checkpoints:
        print("📁 Available checkpoints:")
        checkpoints_base = "./checkpoints/"
        if os.path.exists(checkpoints_base):
            for item in sorted(os.listdir(checkpoints_base)):
                item_path = os.path.join(checkpoints_base, item)
                if os.path.isdir(item_path):
                    # Count checkpoint files in this directory
                    checkpoint_files = [f for f in os.listdir(item_path) if f.endswith('.pth')]
                    print(f"   📂 {item} ({len(checkpoint_files)} checkpoint{'s' if len(checkpoint_files) != 1 else ''})")
                    for ckpt in sorted(checkpoint_files):
                        print(f"      🔹 {ckpt}")
        else:
            print("   ❌ No checkpoints directory found")
        return
    
    # Auto-infer experiment name from model
    experiment_name = f"{args.model}_{args.regularization}"
    
    print(f"🚀 Evaluating {args.model.upper()} model on {args.dataset.upper()} dataset")
    print(f"📁 Using experiment: {experiment_name}")
    print(f"📊 Split: {args.split}")
    if args.cold_thresholds:
        print(f"❄️  Cold thresholds: {args.cold_thresholds}")
    if args.checkpoint_dir:
        print(f"💾 Checkpoint dir: {args.checkpoint_dir}")
    print()
    
    try:
        # Load model and checkpoint
        model, device, config = load_model_and_checkpoint(
            model_type=args.model,
            experiment_name=experiment_name,
            dataset=args.dataset,
            learning_rate=args.lr,
            checkpoint_dir=args.checkpoint_dir
        )
        
        # Run evaluation with the loaded model
        from evaluate import run_evaluation
        run_evaluation(
            model=model,
            device=device,
            config=config,
            dataset=args.dataset,
            eval_split=args.split,
            cold_thresholds=args.cold_thresholds,
            batch_size=args.batch_size
        )
        
        print("✅ Evaluation completed successfully!")
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
