import os
import sys

# Force HuggingFace to load custom code headless (bypassing SLURM prompt timeout)
os.environ['HF_TRUST_REMOTE_CODE'] = '1'
os.environ['TRUST_REMOTE_CODE'] = 'True'
import copy
from pathlib import Path

# Add project roots ensuring tinyllava imports cleanly
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(PROJECT_ROOT / "train"))

# Enforce PyTorch visibility overrides BEFORE HuggingFace is loaded
import os
os.environ["USE_TORCH"] = "1"

import importlib.metadata
_original_version = importlib.metadata.version

def _patched_version(name):
    if name == 'torch':
        return '2.5.0'
    return _original_version(name)

importlib.metadata.version = _patched_version

import torch
torch.__version__ = "2.5.0"

# Neutralize NVTX profiling bugs in DeepSpeed on CUDA accelerators
try:
    import types
    if 'nvtx' not in sys.modules:
        sys.modules['nvtx'] = types.ModuleType('nvtx')
    nvtx_mod = sys.modules['nvtx']
    if not hasattr(nvtx_mod, 'get_domain'):
        class _DummyDomain:
            def push_range(self, *args, **kwargs): pass
            def pop_range(self, *args, **kwargs): pass
        nvtx_mod.get_domain = lambda *args, **kwargs: _DummyDomain()
        nvtx_mod.range_push = lambda *args, **kwargs: None
        nvtx_mod.range_pop = lambda *args, **kwargs: None
except Exception:
    pass

try:
    import deepspeed.utils.nvtx as ds_nvtx
    ds_nvtx._range_push = lambda *args, **kwargs: None
    ds_nvtx._range_pop = lambda *args, **kwargs: None
except Exception:
    pass

# Read-write descriptor for all_tied_weights_keys supporting custom vision towers on transformers 4.45+
class _TiedWeightsDescriptor:
    def __get__(self, instance, owner):
        if instance is None:
            return self
        val = None
        if hasattr(instance, "__dict__"):
            val = instance.__dict__.get("all_tied_weights_keys")
        if val is None:
            val = getattr(instance, "_tied_weights_keys", {})
        if val is None:
            val = {}
        if isinstance(val, (list, tuple, set)):
            val = {k: k for k in val}
        elif not hasattr(val, "keys"):
            val = {}
        return val

    def __set__(self, instance, value):
        if value is None:
            value = {}
        instance.__dict__["all_tied_weights_keys"] = value

    def __delete__(self, instance):
        instance.__dict__.pop("all_tied_weights_keys", None)

import torch.nn as nn
nn.Module.all_tied_weights_keys = _TiedWeightsDescriptor()

try:
    import transformers.modeling_utils
    transformers.modeling_utils.PreTrainedModel.all_tied_weights_keys = _TiedWeightsDescriptor()
except Exception:
    pass

try:
    from tinyllava.train.train_uni import train as tinyllava_train_uni
    from lora_plumber import apply_lora_lego_stack
    import transformers
except Exception as e:
    import traceback
    print("\n" + "!"*80)
    print(" [FATAL ERROR] PyTorch Orchestrator Import Crash Detected!")
    print("!"*80)
    traceback.print_exc(file=sys.stdout)
    sys.exit(1)

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
    import inspect
    original_trainer_init = tinyllava.train.tinyllava_trainer.LLaVATrainer.__init__
    
    def patched_trainer_init(self, model=None, *args, **kwargs):
        # In modern transformers (4.47+ / 5.x), 'tokenizer' was replaced by 'processing_class'
        tokenizer = kwargs.pop("tokenizer", None)
        if tokenizer is not None:
            sig = inspect.signature(original_trainer_init)
            if "processing_class" in sig.parameters:
                kwargs["processing_class"] = tokenizer
            elif "tokenizer" in sig.parameters:
                kwargs["tokenizer"] = tokenizer

        # Intercept and structurally enforce mathematical Multi-PEFT stacking
        if base_adapter_path:
            model = apply_lora_lego_stack(model, base_adapter_path, lora_rank)
            
        original_trainer_init(self, model=model, *args, **kwargs)
        if tokenizer is not None and not hasattr(self, "tokenizer"):
            self.tokenizer = tokenizer
        
    tinyllava.train.tinyllava_trainer.LLaVATrainer.__init__ = patched_trainer_init

    # Patch _get_train_sampler to accept optional dataset parameter from modern transformers
    def patched_get_train_sampler(self, dataset=None, *args, **kwargs):
        ds = dataset if dataset is not None else self.train_dataset
        if ds is None:
            return None
        if self.args.group_by_modality_length:
            lengths = getattr(ds, "modality_lengths", getattr(self.train_dataset, "modality_lengths", None))
            if lengths is not None:
                return tinyllava.train.tinyllava_trainer.LengthGroupedSampler(
                    self.args.train_batch_size,
                    world_size=self.args.world_size,
                    lengths=lengths,
                    group_by_modality=True,
                )
        try:
            return super(tinyllava.train.tinyllava_trainer.LLaVATrainer, self)._get_train_sampler(ds, *args, **kwargs)
        except TypeError:
            return super(tinyllava.train.tinyllava_trainer.LLaVATrainer, self)._get_train_sampler()

    tinyllava.train.tinyllava_trainer.LLaVATrainer._get_train_sampler = patched_get_train_sampler
    
    print("\n==========================================================")
    print("   [VoQA PyTorch Orchestrator] Initializing Training Loop")
    print(f"   Base Adapter Detected: {base_adapter_path is not None}")
    print("==========================================================\n")
    
    # Fire base deepspeed monolithic trainer with patched pipeline
    tinyllava_train_uni()

if __name__ == "__main__":
    main()
