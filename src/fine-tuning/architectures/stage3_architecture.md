# Stage 3: Pure Modularity (4 Translators + 8 Task Adapters)
### (The "Generalist" Assembly Line)

This architecture traces exactly what happens when you type `./execute_pipeline.sh --stage 3`. This introduces the **LoRA-LEGO Multi-PEFT Stacking**, proving how we isolate visual reasoning from language translation.

## 1. The Execution Pathway 

Below is the **Execution Trace** showing exactly how the orchestrator finally calls the Plumber [lora_plumber.py](file:///d:/Main/Rnd_Projects/VoQA-Multilingual/src/fine-tuning/python_modules/lora_plumber.py).

```mermaid
flowchart TD
    classDef script fill:#1e1e1e,stroke:#4a90e2,stroke-width:2px,color:#fff;
    classDef data fill:#2d3748,stroke:#50e3c2,stroke-width:2px,color:#fff;
    classDef core fill:#4a154b,stroke:#e91e63,stroke-width:2px,color:#fff;
    classDef pytorch fill:#b83216,stroke:#ffb300,stroke-width:2px,color:#fff;
    classDef magic fill:#059669,stroke:#34d399,stroke-width:3px,color:#fff;
    
    User([👨‍💻 You]) -->|Runs: --stage 3| CLI[scripts/execute_pipeline.sh]:::script
    CLI -->|Loads Definitions| Env[core/environment.sh]:::core
    CLI -->|Triggers| TaskModule[modules/language_adapter.sh]:::script
    
    TaskModule -->|Universal Mixed Pile| Subsets[(data/subsets/*/universal_100.jsonl)]:::data
    TaskModule -->|Passes Rank 4 + Base Adapter| Engine[core/engine.sh]:::core
    
    Engine -->|Deepspeed Injection| Orchestrator[python_modules/train_orchestrator.py]:::pytorch
    
    Orchestrator -->|Intercepts & Routes to Mechanic| Plumber[lora_plumber.py]:::magic
    Plumber -->|1. Merges Base Adapter| Memory[(RAM)]
    Plumber -->|2. Injects Rank-4 LoRA| Memory
    
    Plumber -->|Returns Upgraded Model| Orchestrator
    Orchestrator -->|Sends to PyTorch GPU Factory| Trainer((TinyLLaVA Trainer)):::pytorch
```

---

## 2. The Neural PyTorch Architecture (Rank-4)

Once [lora_plumber.py](file:///d:/Main/Rnd_Projects/VoQA-Multilingual/src/fine-tuning/python_modules/lora_plumber.py) prepares the model, the neural matrix looks like this. **Notice that the Base Brain and the Rank-8 Task Adapter are totally frozen.** The visual math is perfectly safe, and ONLY the Italian logic updates the tiny Rank-4 Translator block.

```mermaid
flowchart LR
    classDef static fill:#424242,stroke:#666,stroke-width:2px,color:#aaa;
    classDef task fill:#2b6cb0,stroke:#63b3ed,stroke-width:2px,color:#aaa;
    classDef active fill:#2c5282,stroke:#63b3ed,stroke-width:2px,color:#fff;
    classDef adapter fill:#97266d,stroke:#f687b3,stroke-width:3px,color:#fff;
    classDef result fill:#276749,stroke:#68d391,stroke-width:2px,color:#fff;
    
    subgraph GPU Memory Allocation
        Base[Base InternVL-1B / Qwen-2B]:::static
        Base --Frozen Weights--- Lock1((🔒)):::static
        
        Rank8[Rank-8 Target Task Adapter]:::task
        Base -.->|Permanently Merged| Rank8
        Rank8 --Frozen Visual Logic--- Lock2((🔒)):::static
        
        Rank4[Blank Rank-4 Language Translator]:::adapter
        Rank8 -.->|Cleanly Stacked| Rank4
        
        LangData[Foreign Language Flashcards]:::active -->|Gradients| Rank4
    end
    
    Rank4 -->|Saved at 100% completion| Vault[(output_tuning/S3_Pure_Language_Adapters/...)]:::result
```

## 3. What the Trace proves for the Defense:
- **No Catastrophic Forgetting:** Beacuse [lora_plumber.py](file:///d:/Main/Rnd_Projects/VoQA-Multilingual/src/fine-tuning/python_modules/lora_plumber.py) merges and freezes the Rank-8 Task Adapter, the Italian translation flashcards are physically blocked from altering the visual reasoning math.
- **Extreme Parameter Efficiency:** Translating syntax is vastly easier than mathematical reasoning. We correctly drop down to an ultra-compact **Rank 4** for the Translator module.
- **Universal Stacking:** This builds 4 generalist modules. It proves that one Italian Translator can successfully bridge across multiple different Datasets.

---

## 4. Text-Based Architecture Drawing (Non-Mermaid)

```text
========================================================================
            STAGE 3: THE UNIVERSAL "PEFT-LEGO" TRAINING LOOP
========================================================================

  [1. RAW TARGET DATA]
  
  [ 📸 cat_couch.jpg ] 
           |                        [📝 Italian Translation]
           v                 "Ci sono due animali sul divano?"
    +-------------+                                   |
    | Vision Tower|                                   |
    | (Eyeballs)  |                                   |
    +-------------+                                   |
           | (Math pieces)                            |
           +--------------------+---------------------+
                                |
                                V
  [2. THE 24-LAYER FORWARD PASS (The "Lego" Stack)]

 +--------------------------------------------------------------------+
 |                       THE 24 FROZEN LAYERS                         |
 |                                                                    |
 |   [Layer 0]                                                        |
 |    🔒 Frozen Q, K, V Attention Matrix                              |
 |    🔒 [Frozen Base Rank-8 Task Adapter] (Knows how to bake)        |
 |    └── 💥 SHOCK HITS HERE -> [Rank-4 Translator Block 0]           |
 |                                                                    |
 |                            | (Data flows down)                     |
 |                            v                                       |
 |   [...] (Layers 1 through 22 repeat exactly like this)             |
 |                            |                                       |
 |                            v                                       |
 |   [Layer 23] (The Final Layer)                                     |
 |    🔒 Frozen Q, K, V Attention Matrix                              |
 |    🔒 [Frozen Base Rank-8 Task Adapter]                            |
 |    └── 💥 SHOCK HITS HERE -> [Rank-4 Translator Block 23]          |
 |                                                                    |
 +--------------------------------------------------------------------+
                                |
                                V
  [3. THE BACKPROPAGATION (The Surgical Update)]
  
                  Target Answer: [ ✅ "Sì" (Yes) ]
                                     |
                                     v
                        [ 💥 PyTorch Calculates: LOSS! ]
                                     |
                                     | (Gradient Signal fires)
                                     v
                   (Shocks hit ONLY the tiny Rank-4 Blocks.)
                  (The Baking Manual is locked in a vault, safe!)

========================================================================

  [4. THE PEFT SYSTEM OUTPUT]
  
  The system ejects exactly 4 Universal Language Modulators (Translators)
  that can be plugged into the 8 Baseline English models at inference!
  
  💾 -> output_tuning/.../S3_Pure_Language_Adapters/ita/adapter_model.safetensors

========================================================================
```
