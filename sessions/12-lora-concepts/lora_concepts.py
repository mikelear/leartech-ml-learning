"""
Session 12: LoRA Concepts — Adapter Training on a Pre-trained Model

Everything in Sessions 1–11.5 was OUR model — trained from scratch,
every weight ours. LoRA is fundamentally different:

  We take SOMEONE ELSE'S model (GPT-2, 124M params)
  and train a SMALL ADAPTER (~0.5% of params) on our data.

  The base weights are FROZEN. We only train the adapter matrices.
  The result: their foundation + our specialisation.

  Think of it like an MCP server:
    MCP:   Claude (unchanged) + your tools (runtime)
    LoRA:  GPT-2 (frozen)     + your adapter (trained)

Model affected: None in production. This is a local learning exercise.
Type: Learning only — toy model, not deployed.

We use GPT-2 (124M) instead of Qwen (14B) because:
  - Fits on CPU (no GPU needed for learning)
  - Same LoRA mechanics at 100× smaller scale
  - Fast iteration in PyCharm debugger

Set breakpoints at every 🔴 BREAKPOINT line. Debug and inspect.
"""

import os

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType


# ============================================================
# PART 1: Load a pre-trained model (someone else's weights)
# ============================================================

print("=" * 60)
print("PART 1: Load GPT-2 — someone else's model")
print("=" * 60)

print("""
  GPT-2 was trained by OpenAI on internet text.
  We're downloading their weights from Hugging Face.

  We did NOT train this model. We don't own these weights.
  We're going to ADD our own small adapter on top.
""")

MODEL_NAME = "gpt2"  # 124M params, ~500MB, runs on CPU

print(f"  Downloading {MODEL_NAME} from Hugging Face...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

# GPT-2 needs a pad token
tokenizer.pad_token = tokenizer.eos_token
base_model.config.pad_token_id = tokenizer.pad_token_id

total_params = sum(p.numel() for p in base_model.parameters())
print(f"\n  Model: {MODEL_NAME}")
print(f"  Total parameters: {total_params:,}")
print(f"  Size: ~{total_params * 4 / 1024 / 1024:.0f} MB (float32)")
print(f"  Layers: {base_model.config.n_layer}")
print(f"  Attention heads: {base_model.config.n_head}")
print(f"  Hidden dim: {base_model.config.n_embd}")

# Show what the base model does WITHOUT our training
prompt = "Code review verdict for this diff: PASS because"
inputs = tokenizer(prompt, return_tensors="pt")
with torch.no_grad():
    output = base_model.generate(**inputs, max_new_tokens=30, do_sample=False)
base_response = tokenizer.decode(output[0], skip_special_tokens=True)
print(f"\n  Base model response (no fine-tuning):")
print(f"    Input:  '{prompt}'")
print(f"    Output: '{base_response}'")
print(f"\n  Note: GPT-2 has no idea about code review — it just generates text.")

x = 1  # 🔴 BREAKPOINT — Line 65: inspect base_model, total_params
# Explore the model structure:
#   base_model.transformer.h[0]  — first transformer layer
#   base_model.transformer.h[0].attn  — attention module
#   base_model.transformer.h[0].attn.c_attn.weight.shape  — the weight matrix
#
# These weights are OpenAI's. We're about to add our own matrices alongside them.


# ============================================================
# PART 2: What LoRA does — add small matrices alongside frozen weights
# ============================================================

print(f"\n{'='*60}")
print("PART 2: What LoRA actually does")
print(f"{'='*60}")

print("""
  Normal fine-tuning: change ALL weights (124M params)
    - Expensive: needs lots of GPU memory
    - Destructive: overwrites the base model's knowledge
    - Slow: training 124M params takes hours

  LoRA: add SMALL matrices alongside frozen weights
    - Cheap: only train ~600K params (0.5% of total)
    - Non-destructive: base weights never change
    - Fast: training 600K params takes minutes

  How it works:
    Original:  output = W × input        (W is frozen, huge)
    With LoRA: output = W × input + B × A × input
                        ↑ frozen    ↑ trainable (small)

    W: original weight matrix [768 × 768] = 589,824 params (FROZEN)
    A: down-projection [768 × 8] = 6,144 params (TRAINED)
    B: up-projection [8 × 768] = 6,144 params (TRAINED)

    Rank 8 means: compress 768 dims → 8 dims → back to 768
    The "8" is the bottleneck — it forces the adapter to learn
    only the MOST IMPORTANT adjustments, not memorise everything.
""")

# Show the actual weight matrix we'll modify
attn_weight = base_model.transformer.h[0].attn.c_attn.weight
print(f"  First attention layer weight: {attn_weight.shape}")
print(f"  Parameters in this ONE matrix: {attn_weight.numel():,}")
print(f"\n  LoRA adds two small matrices alongside this:")
print(f"    A (down): [768, 8] = {768 * 8:,} params")
print(f"    B (up):   [8, 768] = {8 * 768:,} params")
print(f"    Total adapter for this layer: {768 * 8 + 8 * 768:,} params")
print(f"    vs original matrix: {attn_weight.numel():,} params")
print(f"    Adapter is {(768 * 8 + 8 * 768) / attn_weight.numel():.1%} of original size")

x = 2  # 🔴 BREAKPOINT — Line 114: inspect attn_weight
# The original weight matrix is [768, 2304] (attention projections Q, K, V combined)
# LoRA adds A [768, rank] and B [rank, 768] alongside it
# The rank controls how much the adapter can learn:
#   rank=1: minimal change (very few params)
#   rank=8: moderate change (our default)
#   rank=64: large change (approaches full fine-tuning)


# ============================================================
# PART 3: Apply LoRA — watch the model change
# ============================================================

print(f"\n{'='*60}")
print("PART 3: Apply LoRA config to the model")
print(f"{'='*60}")

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,                      # Rank — the bottleneck dimension
    lora_alpha=16,            # Scaling factor (alpha/r = effective learning rate)
    lora_dropout=0.1,         # Dropout on adapter (prevents overfitting)
    target_modules=["c_attn", "c_proj"],  # Which layers get adapters
    # c_attn = attention QKV projection
    # c_proj = attention output projection
)

print(f"  LoRA config:")
print(f"    Rank (r): {lora_config.r}")
print(f"    Alpha: {lora_config.lora_alpha}")
print(f"    Scaling: alpha/r = {lora_config.lora_alpha / lora_config.r}")
print(f"    Dropout: {lora_config.lora_dropout}")
print(f"    Target modules: {lora_config.target_modules}")

# Apply LoRA — this WRAPS the model, adding adapter matrices
lora_model = get_peft_model(base_model, lora_config)

# Count params
trainable = sum(p.numel() for p in lora_model.parameters() if p.requires_grad)
frozen = sum(p.numel() for p in lora_model.parameters() if not p.requires_grad)
total = trainable + frozen

print(f"\n  After applying LoRA:")
print(f"    Total parameters:     {total:>12,}")
print(f"    Frozen (base model):  {frozen:>12,}  ({frozen/total:.1%})")
print(f"    Trainable (adapter):  {trainable:>12,}  ({trainable/total:.1%})")
print(f"\n  We're only training {trainable/total:.1%} of the model!")

# Show the adapter layers
print(f"\n  Adapter structure (first transformer layer):")
lora_model.print_trainable_parameters()

x = 3  # 🔴 BREAKPOINT — Line 161: inspect lora_model, trainable vs frozen
# Key things to inspect:
#   lora_model.base_model.model.transformer.h[0].attn.c_attn
#   — this now has .lora_A and .lora_B alongside the original .weight
#
#   lora_model.base_model.model.transformer.h[0].attn.c_attn.lora_A['default'].weight.shape
#   — the down-projection matrix [rank, in_features]
#
#   lora_model.base_model.model.transformer.h[0].attn.c_attn.lora_B['default'].weight.shape
#   — the up-projection matrix [out_features, rank]
#
# Try: list(lora_model.base_model.model.transformer.h[0].attn.c_attn.named_parameters())
# You'll see: weight (frozen), lora_A.default.weight (trainable), lora_B.default.weight (trainable)


# ============================================================
# PART 4: Train the adapter on code review examples
# ============================================================

print(f"\n{'='*60}")
print("PART 4: Train the adapter — teach it code review")
print(f"{'='*60}")

print("""
  We'll train on synthetic code review examples.
  The goal: teach the model to output structured review verdicts.

  In production (Session 12b), these would be REAL feedback records
  from leartech-llm-training-data with [leartech]-tagged examples.
""")

# Synthetic training data — code review prompt/response pairs
training_examples = [
    "Review: + eval(userInput)\nVerdict: FAIL. eval() executes arbitrary code from user input.",
    "Review: + const API_KEY = 'sk-secret'\nVerdict: FAIL. Hardcoded API key in source code.",
    "Review: + document.innerHTML = data\nVerdict: FAIL. innerHTML allows XSS injection.",
    "Review: + import pickle; pickle.load(open(f))\nVerdict: FAIL. pickle.load on untrusted data allows code execution.",
    "Review: + subprocess.call(cmd, shell=True)\nVerdict: FAIL. shell=True allows command injection.",
    "Review: + from pydantic import BaseModel\n+ class Item(BaseModel):\n+   name: str\nVerdict: PASS. Proper typed model with validation.",
    "Review: + func Get(ctx context.Context) (*Item, error) {\n+   if err != nil { return nil, fmt.Errorf('get: %w', err) }\nVerdict: PASS. Proper Go error handling with wrapping.",
    "Review: + @Injectable({ providedIn: 'root' })\n+ constructor(private http: HttpClient) {}\nVerdict: PASS. Proper Angular dependency injection.",
    "Review: + async def fetch(id: int) -> dict:\n+   result = await db.find(id)\n+   return result\nVerdict: PASS. Typed async Python with proper return type.",
    "Review: + def test_login():\n+   assert response.status == 200\nVerdict: PASS. Test file with proper assertions.",
]

print(f"  Training examples: {len(training_examples)}")
print(f"  Example:")
print(f"    '{training_examples[0]}'")

# Tokenize
print(f"\n  Tokenizing...")
encodings = tokenizer(
    training_examples,
    return_tensors="pt",
    padding=True,
    truncation=True,
    max_length=128,
)

input_ids = encodings["input_ids"]
attention_mask = encodings["attention_mask"]
labels = input_ids.clone()  # For causal LM, labels = input_ids

print(f"  Input shape: {input_ids.shape}  (batch × sequence_length)")
print(f"  Vocabulary size: {tokenizer.vocab_size:,}")

x = 4  # 🔴 BREAKPOINT — Line 216: inspect input_ids, training_examples
# input_ids[0] is the tokenized version of the first training example
# tokenizer.decode(input_ids[0]) should give back the original text
# Each number maps to a word/subword in the vocabulary


# ============================================================
# PART 5: Training loop — only adapter weights change
# ============================================================

print(f"\n{'='*60}")
print("PART 5: Training — watch only adapter weights change")
print(f"{'='*60}")

# Snapshot base weights BEFORE training
base_weight_before = lora_model.base_model.model.transformer.h[0].attn.c_attn.weight.clone()

# Snapshot adapter weights BEFORE training
adapter_a_before = None
for name, param in lora_model.named_parameters():
    if "lora_A" in name and "h.0." in name:
        adapter_a_before = param.clone()
        break

optimizer = torch.optim.AdamW(
    [p for p in lora_model.parameters() if p.requires_grad],
    lr=5e-4,
)

print(f"  Training for 50 steps...")
lora_model.train()
losses = []

for step in range(50):
    optimizer.zero_grad()
    outputs = lora_model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
    )
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    losses.append(loss.item())

    if step % 10 == 0:
        print(f"    Step {step:3d}: loss={loss.item():.4f}")

print(f"\n  Loss: {losses[0]:.4f} → {losses[-1]:.4f}")

# Check: did base weights change? (they shouldn't!)
base_weight_after = lora_model.base_model.model.transformer.h[0].attn.c_attn.weight
base_changed = not torch.equal(base_weight_before, base_weight_after)

# Check: did adapter weights change? (they should!)
adapter_a_after = None
for name, param in lora_model.named_parameters():
    if "lora_A" in name and "h.0." in name:
        adapter_a_after = param.clone()
        break

adapter_changed = not torch.equal(adapter_a_before, adapter_a_after) if adapter_a_before is not None else False

print(f"\n  Base weights changed:    {'YES ✗' if base_changed else 'NO ✓ (frozen, as expected)'}")
print(f"  Adapter weights changed: {'YES ✓ (trained, as expected)' if adapter_changed else 'NO ✗'}")

if adapter_a_before is not None and adapter_a_after is not None:
    weight_delta = (adapter_a_after - adapter_a_before).abs().mean().item()
    print(f"  Adapter weight delta:    {weight_delta:.6f} (mean absolute change)")

x = 5  # 🔴 BREAKPOINT — Line 271: inspect base_changed, adapter_changed, losses
# THIS is the key insight:
#   base_changed = False  — OpenAI's weights are UNTOUCHED
#   adapter_changed = True — OUR adapter learned something
#
# The base model is exactly the same as when we downloaded it.
# Only our adapter matrices changed. That's LoRA.
#
# Try: torch.equal(base_weight_before, base_weight_after)  → True
# Try: (adapter_a_after - adapter_a_before).abs().max()  → how much the biggest weight moved


# ============================================================
# PART 6: Test the fine-tuned model
# ============================================================

print(f"\n{'='*60}")
print("PART 6: Does the adapter change the output?")
print(f"{'='*60}")

lora_model.eval()

test_prompts = [
    "Review: + eval(userInput)\nVerdict:",
    "Review: + from pydantic import BaseModel\nVerdict:",
    "Review: + const SECRET = 'password123'\nVerdict:",
]

print(f"\n  Comparing base model vs LoRA model on test prompts:\n")

for prompt in test_prompts:
    inputs = tokenizer(prompt, return_tensors="pt")

    # LoRA model output
    with torch.no_grad():
        lora_output = lora_model.generate(**inputs, max_new_tokens=20, do_sample=False)
    lora_text = tokenizer.decode(lora_output[0], skip_special_tokens=True)

    print(f"  Prompt: '{prompt}'")
    print(f"  LoRA:   '{lora_text[len(prompt):]}'")
    print()

x = 6  # 🔴 BREAKPOINT — Line 304: inspect test outputs
# The LoRA model should produce more review-relevant output than base GPT-2.
# With only 10 examples and 50 steps, the results will be imperfect —
# the point is seeing that the adapter CHANGES the output.
#
# In production (Session 12b):
#   - 200+ real training examples (not 10 synthetic)
#   - Qwen 14B (not GPT-2 124M) — already understands code
#   - 1000+ training steps on GPU
#   - The model learns leartech-SPECIFIC patterns


# ============================================================
# PART 7: Save and merge the adapter
# ============================================================

print(f"\n{'='*60}")
print("PART 7: Save adapter + merge into base model")
print(f"{'='*60}")

output_dir = os.path.join(os.path.dirname(__file__), "lora_adapter")

# Save JUST the adapter (not the full model)
lora_model.save_pretrained(output_dir)

adapter_size = sum(
    os.path.getsize(os.path.join(dp, f))
    for dp, _, fns in os.walk(output_dir)
    for f in fns
)
model_size = total_params * 4  # float32

print(f"\n  Adapter saved to: {output_dir}")
print(f"  Adapter size: {adapter_size / 1024:.1f} KB")
print(f"  Base model size: {model_size / 1024 / 1024:.0f} MB")
print(f"  Adapter is {adapter_size / model_size:.2%} of base model size")

print(f"""
  What's in the adapter directory:
    adapter_config.json  — LoRA config (rank, target modules, etc.)
    adapter_model.safetensors — ONLY the adapter weights (A and B matrices)

  The base model weights are NOT in there — they're on Hugging Face.
  To use this adapter, you need: base model + adapter.
""")

# Merge adapter into base model (for deployment)
print(f"  Merging adapter into base model...")
merged_model = lora_model.merge_and_unload()

merged_params = sum(p.numel() for p in merged_model.parameters())
print(f"  Merged model: {merged_params:,} params (same as original)")
print(f"  The adapter weights are now PART OF the base weights.")
print(f"  There is no separate adapter — it's one model.")

print(f"""
  This is what gets exported to GGUF for Ollama:
    1. Train LoRA adapter on our data
    2. Merge adapter into base model
    3. Export merged model as GGUF
    4. ollama create qwen-leartech -f Modelfile
    5. Ollama serves the merged model — no concept of "adapter"

  The adapter is permanent. To "remove" it, you'd re-download
  the original base model. Like merging a PR — the diff is gone,
  it's just the code now.
""")

x = 7  # 🔴 BREAKPOINT — Line 353: inspect merged_model, adapter dir
# After merging:
#   merged_model has the SAME architecture as base_model
#   But the weights are DIFFERENT (adapter baked in)
#   merged_params == total_params (no extra matrices)
#
# Compare to MCP:
#   MCP: remove the server → Claude is unchanged
#   LoRA: merge the adapter → model is permanently changed
#   That's the key difference from the analogy.


# ============================================================
# PART 8: Rank experiment — how much does rank matter?
# ============================================================

print(f"\n{'='*60}")
print("PART 8: Rank experiment — the compression knob")
print(f"{'='*60}")

print("""
  Rank controls how much the adapter can learn:
    rank=1:  minimal — 1 dimension of adjustment
    rank=8:  moderate — 8 dimensions (our default)
    rank=64: large — 64 dimensions (approaching full fine-tune)

  Higher rank = more trainable params = more capacity to learn
  But also more risk of overfitting on small datasets.
""")

for rank in [1, 4, 8, 16, 64]:
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=rank,
        lora_alpha=rank * 2,
        lora_dropout=0.1,
        target_modules=["c_attn", "c_proj"],
    )

    # Re-load base model (merging modified it)
    temp_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    temp_lora = get_peft_model(temp_model, config)

    trainable = sum(p.numel() for p in temp_lora.parameters() if p.requires_grad)
    total = sum(p.numel() for p in temp_lora.parameters())
    pct = trainable / total * 100

    print(f"  rank={rank:2d}:  {trainable:>8,} trainable params  ({pct:.2f}%)")

    del temp_model, temp_lora

x = 8  # 🔴 BREAKPOINT — Line 394: inspect rank comparison
# For leartech Session 12b:
#   - 200 training examples → rank 8-16 is appropriate
#   - rank 64 would overfit on 200 examples
#   - rank 4 might not capture enough conventions
#
# Rule of thumb: start with rank=8, increase if model isn't learning enough


# ============================================================
# Summary
# ============================================================

print(f"\n{'='*60}")
print("Session 12 Complete!")
print(f"{'='*60}")
print(f"""
What you learned:

  1. LoRA adds small matrices (A × B) alongside frozen base weights
     - Base model: 124M params (frozen, OpenAI's)
     - Adapter: ~600K params (trainable, ours)
     - We changed {trainable/total:.1%} of the model

  2. Only adapter weights change during training
     - Base weights: IDENTICAL to download ✓
     - Adapter weights: learned our patterns ✓

  3. Rank controls the adapter's capacity
     - rank=8: {768*8*2:,} params per layer (moderate)
     - Higher rank = more capacity but more overfitting risk

  4. After training: merge adapter into base → one model
     - Adapter is permanent (like merging a PR)
     - Export as GGUF → Ollama serves it as a single model

  5. vs MCP analogy:
     - MCP: runtime, removable, no permanent change
     - LoRA: baked in, permanent, changes the weights

How this maps to production (Session 12b):

  Instead of:   GPT-2 (124M)           + 10 synthetic examples
  We'll use:    Qwen 14B (14,000M)     + 200+ real feedback records

  Instead of:   CPU, 50 steps, minutes
  We'll use:    L4 GPU, 1000+ steps, ~30 min

  Instead of:   saved locally
  We'll deploy: merged GGUF → Ollama → qwen2.5-coder-14b-leartech

  Same mechanics. Different scale.
  That's why we learned on GPT-2 first.
""")
