import sys
import sys
import copy
from pathlib import Path

# Add project roots ensuring tinyllava imports cleanly
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(PROJECT_ROOT / "train"))

from tinyllava.train.train_uni import train as tinyllava_train_uni
from lora_plumber import apply_lora_lego_stack
import transformers

def main():
    """
    Wraps the monolithic train_uni.py to natively support Multilingual S3/S4 PEFT Stacking.
    """
    args = sys.argv.copy()
    
    base_adapter_path = None
    lora_rank = "8"
    
    # Intercept custom PEFT arguments to avoid crashing HuggingFace parser
    cleaned_args = []
    i = 0
    while i < len(args):
        if args[i] == "--base_task_adapter":
            base_adapter_path = args[i+1]
            i += 2
        elif args[i] == "--lora_rank":
            lora_rank = args[i+1]
            i += 2
        else:
            cleaned_args.append(args[i])
            i += 1
            
    sys.argv = cleaned_args
    
    # Monkey-patch LLaVATrainer to intercept the model right before training begins
    import tinyllava.train.tinyllava_trainer
    original_trainer_init = tinyllava.train.tinyllava_trainer.LLaVATrainer.__init__
    
    def patched_trainer_init(self, model=None, *args, **kwargs):
        # Intercept and structurally enforce mathematical Multi-PEFT stacking
        if base_adapter_path:
            model = apply_lora_lego_stack(model, base_adapter_path, lora_rank)
            
        original_trainer_init(self, model=model, *args, **kwargs)
        
    tinyllava.train.tinyllava_trainer.LLaVATrainer.__init__ = patched_trainer_init
    
    print("\n==========================================================")
    print("   [VoQA PyTorch Orchestrator] Initializing Training Loop")
    print(f"   Base Adapter Detected: {base_adapter_path is not None}")
    print("==========================================================\n")
    
    # Fire base deepspeed monolithic trainer with patched pipeline
    tinyllava_train_uni()

if __name__ == "__main__":
    main()
