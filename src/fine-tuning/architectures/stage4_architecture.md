# Stage 4: Exhaustive Modularity (32 Translators + 8 Task Adapters)
### (The "Specialist" Assembly Line)

This architecture traces exactly what happens when you type `./execute_pipeline.sh --stage 4`. This is the final and absolute most powerful execution stage of your thesis. It proves maximum mathematical isolation natively.

## 1. The Execution Pathway 

Below is the **Execution Trace** showing the deep nested loops executed by the Assembly Line Manager.

```mermaid
flowchart TD
    classDef script fill:#1e1e1e,stroke:#4a90e2,stroke-width:2px,color:#fff;
    classDef data fill:#2d3748,stroke:#50e3c2,stroke-width:2px,color:#fff;
    classDef core fill:#4a154b,stroke:#e91e63,stroke-width:2px,color:#fff;
    classDef pytorch fill:#b83216,stroke:#ffb300,stroke-width:2px,color:#fff;
    classDef magic fill:#059669,stroke:#34d399,stroke-width:3px,color:#fff;
    classDef loop fill:#9c4221,stroke:#ed8936,stroke-width:2px,color:#fff;
    
    User([👨‍💻 You]) -->|Runs: --stage 4| CLI[scripts/execute_pipeline.sh]:::script
    CLI -->|Loads Definitions| Env[core/environment.sh]:::core
    CLI -->|Triggers| TaskModule[modules/language_adapter.sh]:::script
    
    TaskModule -->|Nested Loop 1| Langs((4 Languages)):::loop
    Langs -->|Nested Loop 2| DS((5 Datasets)):::loop
    DS -->|Nested Loop 3| Strat((8 Strategies)):::loop
    
    Strat -->|Highly Specific Pile| Subsets[(data/subsets/*_rerendered_100.jsonl)]:::data
    Strat -->|Passes Rank 4 + MATCHED Base Adapter| Engine[core/engine.sh]:::core
    
    Engine -->|Deepspeed Injection| Orchestrator[python_modules/train_orchestrator.py]:::pytorch
    
    Orchestrator -->|Intercepts & Routes| Plumber[lora_plumber.py]:::magic
    Plumber -->|1. Merges Specific S1 Adapter| Memory[(RAM)]
    Plumber -->|2. Injects Rank-4 LoRA| Memory
    
    Plumber -->|Returns Upgraded Model| Orchestrator
    Orchestrator -->|Sends to PyTorch GPU Factory| Trainer((TinyLLaVA Trainer)):::pytorch
```

---

## 2. The Neural PyTorch Architecture (Rank-4)

The memory architecture mathematically looks identical to Stage 3. However, instead of one generalist adapter, there are **32 completely unique, perfectly matched pairs**.

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
        
        Rank8[Rank-8 **STRATEGY-SPECIFIC** Task Adapter]:::task
        Base -.->|Permanently Merged| Rank8
        Rank8 --Frozen Expert Visual Logic--- Lock2((🔒)):::static
        
        Rank4[Blank Rank-4 Language Translator]:::adapter
        Rank8 -.->|Cleanly Stacked| Rank4
        
        LangData[Strategy-Specific Foreign Flashcards]:::active -->|Gradients| Rank4
    end
    
    Rank4 -->|Saved at 100% completion| Vault[(output_tuning/S4_Exhaustive_Language_Adapters/...)]:::result
```

## 3. What the Trace proves for the Defense:
- **Maximum Modularity:** This demonstrates the absolute peak of the Lego-PEFT paradigm. You dynamically match 8 different visual baking manuals perfectly down to 4 different language dictionaries. 
- **Highest Accuracy Yield:** Because the Italian dictionary is only trained on GQA visual reasoning rules, it doesn't get confused by ScienceQA logic, producing the highest native accuracy score mathematically possible.

---

## 4. Text-Based Architecture Drawing (Non-Mermaid)

```text
========================================================================
       STAGE 4: THE SPECIALIZED "PEFT-LEGO" TRAINING LOOP
========================================================================

  [1. THE HIGHLY SPECIFIC INPUT DATA]
  
  [ 📸 GQA Spatial Image ] 
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
 |    🔒 [Frozen EXPERT GQA-SFT Rank-8 Task Adapter] (Deep Expert)    |
 |    └── 💥 SHOCK HITS HERE -> [Rank-4 Translator Block 0]           |
 |                                                                    |
 |                            | (Data flows down)                     |
 |                            v                                       |
 |   [...] (Layers 1 through 22 repeat exactly like this)             |
 |                            |                                       |
 |                            v                                       |
 |   [Layer 23] (The Final Layer)                                     |
 |    🔒 Frozen Q, K, V Attention Matrix                              |
 |    🔒 [Frozen EXPERT GQA-SFT Rank-8 Task Adapter]                  |
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
            (The Expert Baking Manual is totally locked down & safe!)

========================================================================

  [4. THE PERFECT SYSTEM OUTPUT]
  
  Because of the loops, the system mathematically cycles this process 
  hundreds of times, officially popping out 32 Mastermind Adapters:
  
  💾 -> output_tuning/.../S4_Exhaustive/ita/gqa/qa_sft/adapter_model.safetensors

========================================================================
```
