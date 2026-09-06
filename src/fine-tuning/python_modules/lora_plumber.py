import torch
from peft import PeftModel, LoraConfig, get_peft_model

def apply_lora_lego_stack(model, base_adapter_path, new_lora_rank):
    """
    Implements the "LoRA-LEGO" Multi-PEFT architecture (arXiv:2409.16167 logic proxy).
    Merges a frozen S1 Task Adapter (Chef) into the base weights, then attaches a 
    new trainable Language Translator Adapter on top.
    """
    print(f"\n[lora_plumber] Loading and Merging S1 Task Adapter from: {base_adapter_path}")
    
    # 1. Load the Stage 1 adapter and physically merge its math into the base weights
    # This prevents PyTorch from crashing due to nested PeftModel layers!
    model = PeftModel.from_pretrained(
        model, 
        base_adapter_path, 
        adapter_name="task_adapter"
    )
    model = model.merge_and_unload()
    print("[lora_plumber] Successfully merged Task Mathematics into native LLM.")

    # 2. Inject the semantic/linguistic Rank-4 Language Adapter on top
    print(f"[lora_plumber] Injecting new Trainable Language Adapter (Rank={new_lora_rank})")
    
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    
    new_config = LoraConfig(
        r=int(new_lora_rank),
        lora_alpha=int(new_lora_rank) * 2,
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    # Wrap the now-task-aligned model with the new Language LoRA
    model = get_peft_model(model, new_config)
    
    trainable_params, all_params = model.get_nb_trainable_parameters()
    print(f"[lora_plumber] PEFT Stacking complete.")
    print(f"[lora_plumber] Trainable Logic: {trainable_params:,} || Frozen Logic: {all_params - trainable_params:,}")
    
    return model
