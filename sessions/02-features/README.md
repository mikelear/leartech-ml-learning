# Session 2: Feature Extraction — Turning Code Diffs into Numbers

## The Problem

A neural network only understands numbers. But our input is text — a code diff:

```diff
+ const API_KEY = 'sk-secret-12345';
+ const parsed = eval('(' + input + ')');
```

How do we turn this into a tensor that a model can process?

## Three Approaches (simplest → most sophisticated)

### 1. Hand-crafted Features
Count specific things: "how many `eval()` calls?", "any hardcoded secrets?", "how many lines changed?"
Simple, interpretable, but limited — you have to think of every feature manually.

### 2. Bag of Words / TF-IDF
Turn text into a vector where each position represents a word, and the value is how often it appears.
Better — the model can learn which words matter. But loses word order.

### 3. Embeddings
Turn each word (token) into a dense vector of numbers that captures meaning.
"eval" and "Function()" end up close together because they're used similarly.
This is what LLMs use internally.

## What You'll Build

In this session you'll implement all three, starting from the simplest. You'll see how each one turns the same diff into a different tensor shape, and understand why embeddings are the standard approach.

## Debugger Focus

| Breakpoint | What to inspect |
|-----------|----------------|
| Hand-crafted features | A simple vector: `[3, 1, 0, 2, ...]` — each number is a count |
| Vocabulary | The word → index mapping: `{"eval": 0, "const": 1, ...}` |
| Bag of words | A sparse vector: mostly zeros, a few non-zero counts |
| TF-IDF | Same shape as bag of words, but values are weighted by rarity |
| Token IDs | Text → integer IDs: `[42, 156, 7, ...]` |
| Embeddings | Each token ID becomes a dense vector: `shape [num_tokens, embedding_dim]` |

## Key Takeaway

The model is only as good as the features you give it. Hand-crafted features limit what it can learn. Embeddings let it discover patterns you didn't think of.
