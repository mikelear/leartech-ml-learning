# Session 12: LoRA Concepts — Preview

## LoRA is like MCP for model weights

If you understand MCP servers, you already understand the shape of LoRA.

### The pattern

```
MCP:   Claude (unchanged) + your tools (runtime)     = extended capabilities
LoRA:  Qwen (frozen)      + your adapter (trained)    = extended knowledge
```

| | MCP Server | LoRA Adapter |
|---|---|---|
| **Base system** | Claude — closed, unchanged | Qwen 14B — frozen weights |
| **Your addition** | Tools + context you define | Small weight matrices you train |
| **How it connects** | Protocol layer — LLM calls your tools | Matrix multiply — inserted alongside frozen layers |
| **What it changes** | Extends capabilities (new actions) | Extends knowledge (new patterns) |
| **Base aware of it?** | Yes — sees tool descriptions at runtime | Yes — adapter output added to base output |
| **Remove it** | LLM works fine without it | Model works fine without it |
| **You own it** | Your server, your code | Your adapter, your weights |
| **Base owner** | Anthropic (Claude) | Alibaba (Qwen) |

### Where the analogy breaks down

**MCP is runtime.** Claude discovers your tools each session. Nothing persists in the model. Remove the MCP server and Claude is exactly the same as before.

**LoRA is baked in.** After training, the adapter weights are merged into the base model and exported as a single GGUF file. Ollama serves one model — no concept of "base + adapter" at inference time. It's permanent.

### The progression in our pipeline

Each level is more permanent, more expensive, and more capable:

```
Level 1: Prompt engineering          ≈  System prompt / instructions
         Cheapest. No training. Change the prompt, change the output.
         This is Phase 1 — what we run now.

Level 2: Runtime context injection   ≈  MCP tools / Layer 2 conventions
         Still no training. Inject hub/conventions.md into every prompt.
         Claude/DeepSeek/Qwen all get leartech context at call time.
         Removable. Updated by editing conventions.md.

Level 3: LoRA fine-tuning            ≈  Permanent learned knowledge
         Actual training. Adapter weights baked into model.
         Qwen KNOWS leartech patterns without being told each time.
         Not removable (new GGUF file). Updated by retraining.

Level 4: Full fine-tune / DPO        ≈  Deep capability change
         Train more of the model. Learns review style, not just facts.
         Most expensive. We may never need this.
```

### When to use each level

| Signal | Level 1 (prompt) | Level 2 (MCP/inject) | Level 3 (LoRA) |
|---|---|---|---|
| "Don't flag test files for missing coverage" | ✅ Add to prompt | ✅ Add to conventions.md | Overkill |
| "leartech uses subchart patterns, not chart forks" | Prompt gets long | ✅ conventions.md | Worth it if prompt is full |
| "This Go error-handling pattern is our standard" | Hard to describe | Partial — examples help | ✅ Model learns from examples |
| "Review Angular like a senior leartech dev" | Impossible in prompt | Partial — guidelines help | ✅ This is what LoRA excels at |

**Rule of thumb:** If you can write the rule clearly in one sentence, use Level 1/2. If the knowledge is "I know it when I see it" — patterns, style, judgment — that's where LoRA (Level 3) earns its cost.

### How it maps to our models

```
┌─────────────────────────────────────────────────────────────┐
│ Claude Sonnet (Anthropic)                                    │
│   Level 1: system prompt + standards ✅ (running now)        │
│   Level 2: conventions.md injection ← PLAN-context-injection │
│   Level 3: impossible (closed weights)                       │
├─────────────────────────────────────────────────────────────┤
│ DeepSeek V3 (DeepSeek)                                       │
│   Level 1: system prompt + standards ✅ (running now)        │
│   Level 2: conventions.md injection ← PLAN-context-injection │
│   Level 3: possible (open-weight) but we use their API       │
├─────────────────────────────────────────────────────────────┤
│ Ollama/Qwen (self-hosted)                                    │
│   Level 1: system prompt + standards ✅ (running now)        │
│   Level 2: conventions.md injection ← PLAN-context-injection │
│   Level 3: LoRA fine-tune ← Session 12b                      │
│   This is the only model we can actually train.              │
├─────────────────────────────────────────────────────────────┤
│ Our Classifier (leartech-ai-classifier)                      │
│   Not an LLM — no prompts, no LoRA.                          │
│   Trained from scratch on features (Sessions 10–10.8).       │
│   Entirely separate from the above three.                    │
└─────────────────────────────────────────────────────────────┘
```

### What Session 12 will build

A toy LoRA fine-tune on a small Qwen (1.5B or 7B) using ~50 synthetic review pairs. The goal is to understand the mechanics — rank, target modules, QLoRA quantisation, adapter merging, GGUF export — before doing it for real in Session 12b.

**This does NOT change what Ollama serves in production.** The toy model stays local.

Session 12b is when we train on real `[leartech]`-tagged feedback and upload the merged GGUF to Ollama as `qwen2.5-coder-14b-leartech`. That's the production change.
