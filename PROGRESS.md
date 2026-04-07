# ML Learning Journey — Progress

## Session 1: What is a Tensor? ✅

**Concept:** Tensors are containers for numbers with a shape. Everything in ML is a tensor.

**What you learned:**
- Scalar (0D), vector (1D), matrix (2D), 3D tensor — all just tensors with different shapes
- **Shape is everything** — `[2, 3]` means 2 rows, 3 columns. 90% of ML bugs are shape mismatches
- NumPy ↔ PyTorch share memory — modifying one changes the other
- `@` (matrix multiply) is THE core operation in neural networks: `[3, 2] @ [2] = [3]`
- `requires_grad=True` tells PyTorch to track operations for training (backpropagation)

**Key debugger moments:**
- Line 45 (matrix): shape `[2, 3]` — rows × columns
- Line 69 (shared memory): changing NumPy changed PyTorch — same memory
- Line 89 (matrix multiply): `W @ x_input` — this is literally what a neural network layer does

**Files:** `sessions/01-tensors/tensors.py`

---

## Session 2: Feature Extraction ✅

**Concept:** Neural networks only understand numbers. Text must be converted to tensors.

**Four approaches, simplest → most powerful:**
1. **Hand-crafted features** `[12]` — you manually count things (eval calls, secrets). Limited by your imagination.
2. **Bag of words** `[~50]` — count every word. Model sees everything but loses word order.
3. **TF-IDF** `[~50]` — like bag of words but rare words weighted higher. Automatically finds distinctive words.
4. **Embeddings** `[8]` — each word becomes a learned vector. The model discovers what dimensions mean.

**Key insight:** Features ARE the input tensor. Feature extraction is the bridge between text and the model.

**The journey:** `text → tokens → integer IDs → embedding vectors → single vector → model input`

**Key debugger moments:**
- Line 81: hand-crafted features — bad diff has `eval=1, secrets=2`, good has `0, 0`
- Line 110: vocabulary — every unique word gets a number
- Line 191: embedding table — a matrix of random numbers, each row IS a word
- Line 203: embedded diff `[30, 8]` — 30 tokens, each represented by 8 numbers

**Files:** `sessions/02-features/features.py`

---

## Session 3: A Single Neuron ✅

**Concept:** A neuron computes `dot(weights, features) + bias`, then squishes through sigmoid to get a 0-1 probability.

**The training loop:**
1. **Forward pass:** multiply weights × features, add bias, apply sigmoid → prediction
2. **Loss:** compare prediction to truth — how wrong is it?
3. **Backward pass:** compute gradients — which weights need to change, by how much?
4. **Update:** nudge weights in the right direction (weight -= learning_rate × gradient)
5. **Repeat** until predictions are correct

**Key insight:** After training, the weights ENCODE knowledge. Positive weight = feature pushes toward FAIL. Negative = pushes toward PASS. The neuron discovered `eval=bad, HttpClient=good` — same as our code review standards.

**Key debugger moments:**
- Line 56: random weights — the neuron knows nothing
- Line 85: sigmoid squishes any number to 0-1 probability
- Line 122: **THE key moment** — gradients show exactly how to adjust each weight
- Line 142: after one update — weights shifted, prediction improved
- Line 195: after 100 updates — bad→1.0, good→0.0, the neuron learned
- Line 219: final weights — positive weights match security red flags

**Connection to real models:**
- A single neuron IS a model (the simplest one)
- A neural network is many neurons stacked (Session 4)
- Training is the same loop at any scale — even GPT uses forward → loss → backward → update

**Files:** `sessions/03-single-neuron/neuron.py`

---

## Session 4: A Neural Network — Stacking Layers ✅

**Concept:** A neural network is multiple neurons organised into layers. Each layer is just a matrix multiply: `W[16, 8] @ input[8] + bias[16] = output[16]`.

**What you learned:**
- `nn.Linear(8, 16)` creates 16 neurons, each reading 8 inputs — a `[16, 8]` weight matrix
- Data flows: `[8] → layer1 → [16] → relu → [16] → layer2 → [8] → relu → [8] → layer3 → [1] → sigmoid`
- **ReLU** zeros out negatives — without it, stacking layers is pointless (just one big matrix multiply)
- `nn.Module` is PyTorch's standard model class — handles parameter tracking, saving, GPU transfer
- `optimizer.zero_grad() → loss.backward() → optimizer.step()` = one training step
- Our model: 289 parameters. Qwen 14B: 14,000,000,000. Same concept, different scale.

**Key debugger moments:**
- Line 76: expand `model` — see weight matrices for each layer. These ARE the model.
- Line 112: compare before/after ReLU — negatives become zero, positives pass through
- Line 153: count parameters — 128 + 16 + 128 + 8 + 8 + 1 = 289
- Line 197: loss decreasing from ~4.0 to ~0.01 — the network learned

**Why layers matter:** A single neuron learns straight-line rules ("eval > 0 = FAIL"). Multiple layers learn combinations ("eval + innerHTML together = DEFINITELY FAIL, eval in test file = probably OK").

**Files:** `sessions/04-neural-network/network.py`

---

## Session 5: Forward Pass Deep Dive ✅

**Concept:** Zooming into what happens at each layer — tracing specific numbers through the network.

**What you learned:**
- Inside one neuron: weight × feature for each input, sum, add bias, ReLU
- Pattern of which neurons fire IS the representation — different inputs activate different neurons
- After training, neurons specialise: some become "bad code detectors", others "good code detectors"
- **Representation learning** — the network discovers its own internal features without being told
- Killing one neuron changes the prediction — some neurons matter more than others
- `model(features)` = manual trace through layers (same result, PyTorch just does the bookkeeping)

**Key debugger moments:**
- Line 87: inside one neuron — individual weight × feature contributions
- Line 109: 16 neurons — which fire, which are dead after ReLU
- Line 207: **bad vs good activation patterns** — different neurons fire for different code. The network invented its own features.
- Line 235: killed most active neuron — prediction shifted

**Files:** `sessions/05-forward-pass/forward_pass.py`

---

## Session 6: Loss Function + C++ Companion ✅

**Concept:** Loss is a single number measuring how wrong the prediction is. Training exists to make it smaller.

**What you learned:**
- Loss = `-log(prediction)` when truth=1. Penalises confident wrong predictions exponentially.
- `pred=0.9 → loss=0.105` (good), `pred=0.1 → loss=2.302` (very bad), `pred=0.01 → loss=4.605` (terrible)
- Gradient of loss = direction to move prediction to reduce error (negative = increase prediction)
- BCE (classification) has HUGE gradients when very wrong — forces fast learning. MSE is gentler.
- Total loss across examples is the error signal that drives training
- Loss decreasing over epochs = the model is learning

**C++ companion — what `loss.backward()` actually does:**
- The chain rule applied step by step: `d(loss)/d(weight) = d(loss)/d(pred) × d(pred)/d(z) × d(z)/d(weight)`
- Three local derivatives multiplied together per weight
- `dloss_dpred = -1/prediction` (how loss depends on prediction)
- `dpred_dz = prediction * (1 - prediction)` (sigmoid derivative)
- `dz_dweight = feature_value` (how weighted sum depends on weight)
- PyTorch does this for ALL weights simultaneously via the computation graph
- The C++ code shows the "glass box" version of Python's black box `loss.backward()`

**CLion setup note:** If CLion defaults to Python mode (due to Python plugin), disable the Python plugin in CLion. CLion = C++, PyCharm = Python. Delete `.idea` folder and reopen if needed.

**libtorch (PyTorch's C++ library):** Pre-built ~200MB download, doable but heavy. Our manual C++ shows the same maths without the dependency. Not needed for the learning journey.

**Key debugger moments:**
- Python line 44: loss for different predictions — see the exponential penalty curve
- Python line 89: gradient direction — negative means "increase prediction"
- Python line 115: MSE vs BCE comparison — BCE is harsher on confident wrong answers
- C++ line 82: **THE KEY** — three chain rule derivatives as separate floats, multiplied together
- C++ line 97: final gradients — exactly what Python's `loss.backward()` computes

**Files:** `sessions/06-loss-function/loss.py`, `loss.cpp`, `CMakeLists.txt`

---

## Session 6b: libtorch Debug — The Real PyTorch C++ Library ✅

**Concept:** Same code as Session 6's manual C++, but using the actual PyTorch C++ library (libtorch). You can step INTO `torch::sigmoid()`, `loss.backward()`, and see the real autograd engine.

**Setup:**
- Pre-built libtorch downloaded (~60MB ARM64 macOS)
- `make setup` downloads it, `make build` compiles, `make run` executes
- Open folder in CLion → detects CMakeLists.txt → Debug

**Makefile targets:**
```
make setup    # Downloads libtorch (one time, ~60MB)
make build    # Compiles the C++ binary with debug symbols
make run      # Runs the binary
make debug    # Runs with lldb terminal debugger
make clean    # Removes build artifacts
```

**What you learned:**
- `torch::Tensor` in C++ is the SAME object as Python's `torch.tensor` — same memory layout, same autograd
- The computation graph (`grad_fn` chain) is visible in the debugger: `NegBackward0 → LogBackward0 → SigmoidBackward0 → ...`
- Stepping into `loss.backward()` enters `autograd::Engine::execute()` — the real chain rule engine
- Python is the UI, C++ is the engine: `torch.sigmoid(x)` in Python calls `at::sigmoid()` in C++

**Relationship to other sessions:**
- Session 6 manual C++ = "here's what the chain rule does" (hand-written)
- Session 6b libtorch = "here's PyTorch actually doing it" (real library)
- Both teach the same concept, different depth

**CLion setup note:** Disable Python plugin in CLion if it defaults to Python mode. Open each C++ session folder separately. The Makefile handles downloading libtorch and building — CLion just needs to load CMakeLists.txt.

**Files:** `sessions/06b-libtorch-debug/main.cpp`, `CMakeLists.txt`, `Makefile`

---

## Session 7: Backpropagation Through Multiple Layers ✅

**Concept:** The chain rule applied layer by layer in reverse. Gradients flow backwards from loss through each layer.

**What you learned:**
- Backprop = receive gradient from layer ahead, multiply by local derivative, pass backwards
- **Gradient magnitude shrinks** as it flows backwards — layer 3 gets strongest signal, layer 1 gets weakest
- **Vanishing gradient problem:** sigmoid derivative max = 0.25, so 5 layers = 0.25^5 ≈ 0.001 shrinkage
- **ReLU fixes it:** derivative is 0 or 1, never shrinks the gradient
- Manual backprop through 3 layers matched PyTorch's `backward()` exactly — proof we understand autograd
- As training converges (loss → 0), all gradients → 0 (nothing left to learn)

**The complete training cycle understood:**
```
Session 2:  text → features (input)
Session 5:  features → prediction (forward pass)
Session 6:  prediction vs truth → loss (how wrong)
Session 7:  loss → gradients for every weight (backward pass)
Session 8:  gradients → weight updates (optimizer) ← next
```

**Key debugger moments:**
- Line 80: gradient sizes across layers — layer3 > layer2 > layer1
- Line 161: 5-layer sigmoid vs ReLU — ReLU gradient orders of magnitude larger
- Line 196: manual backprop step by step, all 3 layers
- Line 226: manual == PyTorch — verified

**Files:** `sessions/07-backpropagation/backprop.py`

---

## Session 8: The Training Loop ✅

**Concept:** All pieces assembled — forward → loss → zero_grad → backward → step → repeat. Plus practical decisions: optimizers, learning rate, scheduling, train/val split.

**What you learned:**
- The 5-step training loop is universal — same for our 289-param model and GPT-4's trillions
- **Adam > SGD** for most cases — adapts learning rate per-weight, converges faster
- **Learning rate** is the most important hyperparameter: too small = slow, too large = explodes (NaN), just right = fast + stable
- **LR scheduling:** start big, shrink over time (StepLR, gamma=0.5 every 20 epochs)
- **Train/val split:** train_loss shows fitting, val_loss shows generalisation. If val diverges → overfitting
- `model.train()` vs `model.eval()` — affects dropout/batch norm behaviour
- `torch.no_grad()` during validation — saves memory, no gradient computation needed

**Key debugger moments:**
- Line 95: one complete training step annotated — watch prediction improve after step
- Line 121: SGD vs Adam — Adam converges faster
- Line 155: learning rate sweep — 0.001 (slow), 0.1 (good), 5.0 (explodes)
- Line 223: train vs val loss — two curves diverging = overfitting preview

**Files:** `sessions/08-training-loop/training.py`

---

## Session 9: Overfitting ✅

**Concept:** A model that memorises training data instead of learning patterns is useless on new data. Detected by train_loss ↓ while val_loss ↑.

**What you learned:**
- **Overfit signature:** 135,425 parameters for 20 examples → train_loss near 0, val_loss rising
- **Dropout** (0.5): randomly kills 50% of neurons each forward pass → forces network to learn patterns, not rely on specific neurons
- **Early stopping:** monitor val_loss, stop when it hasn't improved for N epochs. Restore best weights.
- **Weight decay** (L2 regularisation): penalises large weights → keeps model simpler internally
- **Right-sizing** the model is the best fix: 145 params for 20 examples beats 135,425 params
- Rule of thumb: parameters should be < 10× training examples

**Key debugger moments:**
- Line 95: 6,771 parameters per training example — way too many
- Line 120: THE overfit signature — train_loss → 0, val_loss → ↑
- Line 155: dropout comparison — more stable val_loss
- Line 202: early stopping caught the overfit at epoch ~30-60
- Line 278: all techniques compared — simple model usually wins

**For our real classifier:** ~500 examples → < 5,000 params → 2-layer network with 16-32 hidden neurons

**Files:** `sessions/09-overfitting/overfitting.py`

---

## Session 10: Our Classifier — Real Data ✅

**Concept:** Everything from Sessions 1-9 assembled into a real classifier trained on 102 actual feedback records from the AI review pipeline.

**What you learned:**
- Loaded real feedback data from `leartech-llm-training-data/feedback/` — diffs, verdicts, scores
- 16 hand-crafted features: security (eval, innerHTML, secrets), quality (imports, constructor, async), metrics (lines, functions, tests, debt)
- Train/val/test split (70/15/15) — model never sees test data during training
- Right-sized model: ~1,100 parameters with dropout + early stopping
- **Confusion matrix:** TP, TN, FP, FN — the full story beyond accuracy
- **Precision vs Recall:** for security, recall matters most (missing bad code is worse than false alarms)
- Model saved as `code_classifier.pt` (~5KB) — ready for deployment

**Key debugger moments:**
- Line 72: real feedback data — inspect `raw_data[0]` for actual diff + verdict
- Line 107: feature matrix `[102, 16]` — 102 real diffs, 16 features
- Line 222: confusion matrix + precision/recall/F1
- Line 258: individual predictions — which diffs get right/wrong
- Line 282: saved model — 5KB vs Qwen's 8GB, same concept

**Output:** `code_classifier.pt` — deployed in Session 11

**Files:** `sessions/10-our-classifier/classifier.py`

---

## Session Summary

| Session | Topic | Track | Status |
|---------|-------|-------|--------|
| 1 | Tensors | Python | ✅ |
| 2 | Feature Extraction | Python | ✅ |
| 3 | Single Neuron | Python | ✅ |
| 4 | Neural Network | Python | ✅ |
| 5 | Forward Pass | Python | ✅ |
| 6 | Loss Function | Python + C++ (manual) | ✅ |
| 6b | libtorch Debug | C++ (real library) | ✅ |
| 7 | Backpropagation | Python | ✅ |
| 8 | Training Loop | Python | ✅ |
| 9 | Overfitting | Python | ✅ |
| 10 | Our Classifier (real data) | Python | ✅ |
| 11 | Deploy to K8s | Python + Cluster | Pending |
| 12 | LoRA Concepts | Python | Pending |
