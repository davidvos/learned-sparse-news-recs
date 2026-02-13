import sys
from pathlib import Path

# Add src to path if running from root
sys.path.append(str(Path(__file__).parent))

from config import get_config
from utils import set_seeds, get_device, load_model_and_checkpoint
from trainer import Trainer
from evaluator import Evaluator

def main():
    # 1. Load Configuration
    config, args = get_config()
    
    # 2. Setup
    set_seeds(config['global']['seed'])
    device = get_device(args.device)
    print(f"Using device: {device}")
    print(f"Experiment: {config['experiment']['name']}")
    
    # 3. Initialize Model
    # Explicitly no auto-resume for now to keep it simple and safe, unless checkpoint logic is enhanced
    model, checkpoint_data, start_step = load_model_and_checkpoint(
        args.model_type, 
        args.dataset, 
        config, 
        device,
        checkpoint_dir=None 
    )
    
    # 4. Initialize Trainer and Evaluator
    trainer = Trainer(model, config, device, args.model_type, args.dataset)
    evaluator = Evaluator(config, device, args.dataset, args.model_type)
    
    # 5. Start Training
    trainer.train(evaluator)
    
if __name__ == "__main__":
    main()
