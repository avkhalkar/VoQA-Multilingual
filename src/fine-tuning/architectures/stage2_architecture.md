# Stage 2: Monolithic Multilingual Baseline
### (The Lazy Assembly Line)

This architecture traces exactly what happens when you type `./execute_pipeline.sh --stage 2`. This stage is intentionally designed as the "baseline" to prove why our modular Lego stacking is vastly superior.

## 1. The Execution Pathway 

Below is the **Execution Trace** showing exactly how the modules communicate and trigger PyTorch natively.

```mermaid
flowchart TD
    classDef script fill:#1e1e1e,stroke:#4a90e2,stroke-width:2px,color:#fff;
    classDef data fill:#2d3748,stroke:#50e3c2,stroke-width:2px,color:#fff;
    classDef core fill:#4a154b,stroke:#e91e63,stroke-width:2px,color:#fff;
    classDef pytorch fill:#b83216,stroke:#ffb300,stroke-width:2px,color:#fff;
    
    User([👨‍💻 You]) -->|Runs: --stage 2| CLI[scripts/execute_pipeline.sh]:::script
    CLI -->|Loads Definitions| Env[core/environment.sh]:::core
    CLI -->|Triggers| TaskModule[modules/monolithic.sh]:::script
    
    TaskModule -->|Loops 4 target languages| Subsets[(data/subsets/*_ft_100.jsonl)]:::data
    TaskModule -->|Passes Rank 8, NO Base Adapter| Engine[core/engine.sh]:::core
    
    Engine -->|Deepspeed Injection| Orchestrator[python_modules/train_orchestrator.py]:::pytorch
    Orch_Sub[lora_plumber.py] -.->|Bypassed in Stage 2| Orchestrator
    
    Orchestrator -->|Sends to PyTorch GPU Factory| Trainer((TinyLLaVA Trainer)):::pytorch
```

---

## 2. The Neural PyTorch Architecture (Rank-8)

Once the execution flow hits PyTorch, the neural matrix undergoes the following operation. **Notice that we are forcing BOTH Visual Data and Foreign Language Data into the exact same Rank-8 block.** This causes the visual logic to be overwritten by the language logic!

```mermaid
flowchart LR
    classDef static fill:#424242,stroke:#666,stroke-width:2px,color:#aaa;
    classDef error fill:#9b2c2c,stroke:#fc8181,stroke-width:2px,color:#fff;
    classDef adapter fill:#b83216,stroke:#fbd38d,stroke-width:3px,color:#fff;
    classDef result fill:#744210,stroke:#d69e2e,stroke-width:2px,color:#fff;
    
    subgraph GPU Memory Allocation
        Base[Base InternVL-1B / Qwen-2B]:::static
        Base --Frozen Weights--- Lock((🔒)):::static
        
        Rank8[Rank-8 OVERCROWDED LoRA]:::adapter
        Base -.->|Attached| Rank8
        
        EngData[English Visual Flashcards]:::error -->|Conflicts With| TargetData[Target Language Flashcards]:::error
        TargetData -->|Erodes Gradients| Rank8
        EngData -->|Erodes Gradients| Rank8
    end
    
    Rank8 -->|Saved at 100% completion| Vault[(output_tuning/S2_Monolithic_Baseline/...)]:::result
```

## 3. What the Trace proves for the Defense:
- **Multilingual Erosion:** Because the visual reasoning math and the Italian translation math are violently trying to update the exact same `Rank-8 LoRA` weights simultaneously, the model experiences Catastrophic Forgetting.
- **No Plumber Intervention:** The [lora_plumber.py](file:///d:/Main/Rnd_Projects/VoQA-Multilingual/src/fine-tuning/python_modules/lora_plumber.py) is bypassed because we are not mathematically stacking Lego blocks here. We are trying (and failing) to do everything in one single block.

---

## 4. Text-Based Architecture Drawing (Non-Mermaid)

```text
========================================================================
       STAGE 2: THE MONOLITHIC "LAZY" TRAINING LOOP (Rank 8 Only)
========================================================================

  [1. RAW OVERCRAMMED DATA]
  
  [ 📸 cat_couch.jpg ] 
           |                        [📝 Italian Translation]
           v                 "Ci sono due animali sul divano?" (Are there 2 animals...)
    +-------------+                                   |
    | Vision Tower|                                   |
    | (Eyeballs)  |                                   |
    +-------------+                                   |
           | (Math pieces)                            |
           +--------------------+---------------------+
                                |
                                V
  [2. THE 24-LAYER FORWARD PASS (The Bottleneck)]

 +--------------------------------------------------------------------+
 |                       THE 24 FROZEN LAYERS                         |
 |                                                                    |
 |   [Layer 0]                                                        |
 |    🔒 Frozen Q, K, V Attention Matrix                              |
 |    └── 💥 SHOCK HITS HERE -> [Rank-8 LoRA Block 0]                 |
 |          (Catastrophic Forgetting: Visual and Language Math Clash) |
 |                                                                    |
 |                            | (Data flows down)                     |
 |                            v                                       |
 |   [...] (Layers 1 through 22 repeat exactly like this)             |
 |                            |                                       |
 |                            v                                       |
 |   [Layer 23] (The Final Layer)                                     |
 |    🔒 Frozen Q, K, V Attention Matrix                              |
 |    └── 💥 SHOCK HITS HERE -> [Rank-8 LoRA Block 23]                |
 |                                                                    |
 +--------------------------------------------------------------------+
                                |
                                V
  [3. THE BACKPROPAGATION (The Erosion)]
  
                  Target Answer: [ ✅ "Sì" (Yes) ]
                                     |
                                     v
                        [ 💥 PyTorch Calculates: LOSS! ]
                                     |
                                     | (Gradient Signal fires)
                                     v
                  (Shocks overwrite the visual math with language math!)

========================================================================

  [4. THE FLAWED SYSTEM OUTPUT]
  
  The final adapter is saved, but its visual reasoning capacity has 
  been severely eroded to make room for Italian grammar rules:
  
  💾 -> output_tuning/.../S2_Monolithic_Baseline/adapter_model.safetensors

========================================================================
```
