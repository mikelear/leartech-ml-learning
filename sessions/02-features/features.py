"""
Session 2: Feature Extraction — Turning Code Diffs into Numbers

The fundamental challenge: neural networks only understand numbers,
but our input is text (code diffs). This session shows three ways
to convert text → tensors, from simplest to most powerful.

Set breakpoints at every 🔴 BREAKPOINT line. Debug and inspect.
"""

import torch
import numpy as np
import re
from collections import Counter

# ============================================================
# Our sample data — real code diffs like our AI review pipeline sees
# ============================================================

# A BAD diff (should FAIL) — hardcoded secrets, eval, innerHTML
bad_diff = """
+ import { Component } from '@angular/core';
+ const API_KEY = 'sk-proj-abc123def456';
+ const DB_URL = 'postgres://admin:password@db.example.com:5432/prod';
+
+ export class UserSettings {
+   loadData(): void {
+     const input = localStorage.getItem('data');
+     const parsed = eval('(' + input + ')');
+     document.getElementById('output').innerHTML = parsed.html;
+     fetch('/api/settings', {
+       headers: { 'Authorization': 'Bearer ' + API_KEY }
+     });
+   }
+ }
"""

# A GOOD diff (should PASS) — proper Angular patterns, no secrets
good_diff = """
+ import { Component } from '@angular/core';
+ import { HttpClient } from '@angular/common/http';
+ import { DomSanitizer } from '@angular/platform-browser';
+
+ export class UserSettings {
+   constructor(
+     private http: HttpClient,
+     private sanitizer: DomSanitizer
+   ) {}
+
+   loadData(): void {
+     this.http.get('/api/settings').subscribe(data => {
+       this.settings = data;
+     });
+   }
+ }
"""


# ============================================================
# APPROACH 1: Hand-Crafted Features
# ============================================================
# YOU decide what to count. Simple but limited.

def extract_handcrafted_features(diff: str) -> torch.Tensor:
    """
    Turn a diff into a fixed-size feature vector by counting
    specific patterns we know are important.
    """
    features = [
        # Security red flags
        len(re.findall(r'eval\s*\(', diff)),               # eval() calls
        len(re.findall(r'innerHTML', diff)),                 # innerHTML usage
        len(re.findall(r'(API_KEY|SECRET|PASSWORD)', diff, re.IGNORECASE)),  # hardcoded secrets
        len(re.findall(r'(sk-|ghp_|password)', diff, re.IGNORECASE)),       # secret patterns

        # Code quality signals
        len(re.findall(r'import\s+', diff)),                # number of imports
        len(re.findall(r'constructor', diff)),               # has constructor
        len(re.findall(r'(subscribe|Observable)', diff)),    # uses reactive patterns
        len(re.findall(r'(HttpClient|DomSanitizer)', diff)), # uses Angular services

        # Size metrics
        len(diff.split('\n')),                               # total lines
        len(re.findall(r'^\+', diff, re.MULTILINE)),        # added lines
        len(re.findall(r'^-', diff, re.MULTILINE)),         # removed lines
        len(re.findall(r'function|=>', diff)),               # function count
    ]
    return torch.tensor(features, dtype=torch.float32)


print("=" * 60)
print("APPROACH 1: Hand-Crafted Features")
print("=" * 60)

bad_features = extract_handcrafted_features(bad_diff)
good_features = extract_handcrafted_features(good_diff)

feature_names = [
    "eval_calls", "innerHTML", "secret_names", "secret_patterns",
    "imports", "constructor", "reactive", "angular_services",
    "total_lines", "added_lines", "removed_lines", "functions"
]

print("\nBad diff features:")
for name, val in zip(feature_names, bad_features):
    print(f"  {name:20s}: {val.item():.0f}")

print("\nGood diff features:")
for name, val in zip(feature_names, good_features):
    print(f"  {name:20s}: {val.item():.0f}")

x = 1  # 🔴 BREAKPOINT — inspect bad_features and good_features
# Notice: bad_features has eval=1, innerHTML=1, secrets=2
#         good_features has eval=0, innerHTML=0, secrets=0, but angular_services=2
# The model would learn: "high eval + secrets = FAIL, high angular_services = PASS"
#
# Problem: we had to MANUALLY decide what to count.
# What if there's a pattern we didn't think of?


# ============================================================
# APPROACH 2: Bag of Words
# ============================================================
# Let the DATA decide what matters. Count every word.

def tokenize(text: str) -> list[str]:
    """Split text into words (tokens). Simple but effective."""
    # Remove diff markers and split on non-alphanumeric chars
    text = re.sub(r'^[+\-]', '', text, flags=re.MULTILINE)
    tokens = re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*', text.lower())
    return tokens


def build_vocabulary(texts: list[str]) -> dict[str, int]:
    """Build a mapping from word → index number."""
    all_tokens = []
    for text in texts:
        all_tokens.extend(tokenize(text))
    # Unique words, sorted, each gets a number
    unique = sorted(set(all_tokens))
    return {word: idx for idx, word in enumerate(unique)}


def bag_of_words(text: str, vocab: dict[str, int]) -> torch.Tensor:
    """Turn text into a vector of word counts."""
    tokens = tokenize(text)
    vector = torch.zeros(len(vocab))
    for token in tokens:
        if token in vocab:
            vector[vocab[token]] += 1
    return vector


print("\n" + "=" * 60)
print("APPROACH 2: Bag of Words")
print("=" * 60)

# Build vocabulary from both diffs
vocab = build_vocabulary([bad_diff, good_diff])
print(f"\nVocabulary size: {len(vocab)} unique words")
print(f"First 10 words: {dict(list(vocab.items())[:10])}")

x = 2  # 🔴 BREAKPOINT — inspect vocab
# vocab is a dictionary: {"angular": 0, "api": 1, "authorization": 2, ...}
# Each unique word gets a number. This is the foundation of NLP.
# Try in Evaluate: vocab["eval"], vocab["import"], len(vocab)

bad_bow = bag_of_words(bad_diff, vocab)
good_bow = bag_of_words(good_diff, vocab)

print(f"\nBag of words shape: {bad_bow.shape}")  # [vocab_size]
print(f"Non-zero elements (bad):  {(bad_bow > 0).sum().item():.0f}")
print(f"Non-zero elements (good): {(good_bow > 0).sum().item():.0f}")

# Show which words appear in each diff
print("\nBad diff top words:")
for idx in bad_bow.topk(5).indices:
    word = list(vocab.keys())[idx]
    count = bad_bow[idx].item()
    print(f"  {word}: {count:.0f}")

x = 3  # 🔴 BREAKPOINT — inspect bad_bow, good_bow
# bad_bow is a SPARSE vector — mostly zeros, a few word counts.
# Shape is [vocab_size] — one element per word in the vocabulary.
# Try in Evaluate: bad_bow.sum() (total word count), (bad_bow > 0).sum() (unique words)
#
# Improvement over hand-crafted: the model sees EVERY word, not just
# the ones we thought to count. It can discover patterns itself.
#
# Problem: "eval is dangerous" and "dangerous eval" produce the same vector.
# Word ORDER is lost.


# ============================================================
# APPROACH 3: TF-IDF (Term Frequency - Inverse Document Frequency)
# ============================================================
# Like bag of words, but rare words get higher weight.
# "eval" appearing in 1 out of 100 diffs is more informative
# than "import" appearing in 99 out of 100.

def compute_tfidf(texts: list[str], vocab: dict[str, int]) -> list[torch.Tensor]:
    """Compute TF-IDF vectors for a list of texts."""
    n_docs = len(texts)
    bow_vectors = [bag_of_words(t, vocab) for t in texts]

    # IDF: log(total_docs / docs_containing_word)
    doc_freq = torch.zeros(len(vocab))
    for bow in bow_vectors:
        doc_freq += (bow > 0).float()

    # Add 1 to avoid division by zero
    idf = torch.log((n_docs + 1) / (doc_freq + 1))

    # TF-IDF = word_count * idf_weight
    tfidf_vectors = []
    for bow in bow_vectors:
        tf = bow / (bow.sum() + 1e-8)  # normalize counts
        tfidf_vectors.append(tf * idf)

    return tfidf_vectors


print("\n" + "=" * 60)
print("APPROACH 3: TF-IDF")
print("=" * 60)

tfidf_vectors = compute_tfidf([bad_diff, good_diff], vocab)
bad_tfidf = tfidf_vectors[0]
good_tfidf = tfidf_vectors[1]

print(f"\nTF-IDF shape: {bad_tfidf.shape}")

# Show highest weighted words — these are the DISTINCTIVE words
print("\nBad diff — most distinctive words (highest TF-IDF):")
for idx in bad_tfidf.topk(5).indices:
    word = list(vocab.keys())[idx]
    weight = bad_tfidf[idx].item()
    print(f"  {word}: {weight:.4f}")

print("\nGood diff — most distinctive words:")
for idx in good_tfidf.topk(5).indices:
    word = list(vocab.keys())[idx]
    weight = good_tfidf[idx].item()
    print(f"  {word}: {weight:.4f}")

x = 4  # 🔴 BREAKPOINT — inspect bad_tfidf, good_tfidf
# Compare to bag of words: the VALUES are different.
# Common words like "import" have LOW weight.
# Rare/distinctive words like "eval", "sanitizer" have HIGH weight.
# TF-IDF automatically discovers what makes each diff DIFFERENT.


# ============================================================
# APPROACH 4: Token IDs → Embeddings (what LLMs actually use)
# ============================================================
# Each word becomes a LEARNED vector of numbers.
# Words with similar meaning end up with similar vectors.

print("\n" + "=" * 60)
print("APPROACH 4: Embeddings")
print("=" * 60)

# Step 1: Convert words to integer IDs
def text_to_ids(text: str, vocab: dict[str, int], max_length: int = 50) -> torch.Tensor:
    """Convert text to a fixed-length sequence of token IDs."""
    tokens = tokenize(text)[:max_length]
    ids = [vocab.get(t, 0) for t in tokens]  # 0 for unknown words
    # Pad to fixed length (neural networks need fixed-size inputs)
    while len(ids) < max_length:
        ids.append(0)  # 0 = padding
    return torch.tensor(ids, dtype=torch.long)


bad_ids = text_to_ids(bad_diff, vocab, max_length=30)
good_ids = text_to_ids(good_diff, vocab, max_length=30)

print(f"\nToken IDs shape: {bad_ids.shape}")  # [30] — 30 integer IDs
print(f"First 10 IDs (bad):  {bad_ids[:10].tolist()}")
print(f"First 10 IDs (good): {good_ids[:10].tolist()}")

x = 5  # 🔴 BREAKPOINT — inspect bad_ids, good_ids
# These are just integers — each number is an index into the vocabulary.
# The model doesn't know what these numbers MEAN yet.
# That's what embeddings are for.

# Step 2: Embedding layer — turns integer IDs into dense vectors
VOCAB_SIZE = len(vocab)
EMBEDDING_DIM = 8  # Each word becomes an 8-dimensional vector
                    # (real models use 768-4096)

# This creates a lookup table: vocab_size rows × embedding_dim columns
# Each row is a word's vector representation
embedding = torch.nn.Embedding(VOCAB_SIZE, EMBEDDING_DIM)

print(f"\nEmbedding table shape: {embedding.weight.shape}")  # [vocab_size, 8]
print(f"  {VOCAB_SIZE} words × {EMBEDDING_DIM} dimensions each")

x = 6  # 🔴 BREAKPOINT — inspect embedding.weight
# This is a MATRIX of random numbers. Each row is a word's vector.
# Try in Evaluate: embedding.weight[vocab["eval"]] — the vector for "eval"
#                  embedding.weight[vocab["import"]] — the vector for "import"
# They're random now. After TRAINING, similar words would have similar vectors.

# Step 3: Look up embeddings for our token IDs
bad_embedded = embedding(bad_ids)
good_embedded = embedding(good_ids)

print(f"\nEmbedded bad diff shape: {bad_embedded.shape}")   # [30, 8]
print(f"Embedded good diff shape: {good_embedded.shape}")    # [30, 8]
# Each of the 30 tokens is now an 8-dimensional vector.
# The diff went from text → 30 integers → a [30, 8] tensor.

x = 7  # 🔴 BREAKPOINT — inspect bad_embedded
# Shape [30, 8]: 30 tokens, each represented by 8 numbers.
# Try in Evaluate: bad_embedded[0] — the embedding vector for the first token
#                  bad_embedded.shape
#
# This is what goes INTO the neural network.
# The network will learn which patterns in these 8-number vectors
# predict PASS vs FAIL.

# Step 4: Collapse the sequence into a single vector
# A simple approach: average all token vectors
bad_pooled = bad_embedded.mean(dim=0)    # Average across tokens → shape [8]
good_pooled = good_embedded.mean(dim=0)  # Same

print(f"\nPooled bad shape: {bad_pooled.shape}")   # [8]
print(f"Pooled good shape: {good_pooled.shape}")    # [8]

x = 8  # 🔴 BREAKPOINT — inspect bad_pooled, good_pooled
# Now each entire diff is a SINGLE vector of 8 numbers.
# This is the INPUT to the classifier network.
#
# The journey:
#   Text → tokens → integer IDs → embedding vectors → pooled to single vector
#   "code diff" → ["eval", "const", ...] → [42, 7, ...] → [[0.1, -0.3, ...], ...] → [0.05, -0.12, ...]


# ============================================================
# COMPARISON: What does the model actually see?
# ============================================================

print("\n" + "=" * 60)
print("COMPARISON: Tensor shapes the model sees")
print("=" * 60)

print(f"""
Approach              | Shape           | What the model sees
──────────────────────|─────────────────|─────────────────────
Hand-crafted features | {str(bad_features.shape):15s} | {bad_features.shape[0]} numbers you chose to count
Bag of words          | {str(bad_bow.shape):15s} | {bad_bow.shape[0]} word counts (one per vocab word)
TF-IDF                | {str(bad_tfidf.shape):15s} | {bad_tfidf.shape[0]} weighted word importances
Embeddings (pooled)   | {str(bad_pooled.shape):15s} | {bad_pooled.shape[0]} learned dimensions

Hand-crafted: you decide what matters (limited by your imagination)
Bag of words: every word matters equally (loses word order)
TF-IDF:       rare words matter more (still loses word order)
Embeddings:   the model LEARNS what matters (keeps some context)
""")

x = 9  # 🔴 BREAKPOINT — final comparison
# Look at the shapes side by side.
# Hand-crafted: [12] — tiny, you chose 12 features
# Bag of words: [~50] — one per vocab word
# Embeddings: [8] — compressed, learned representation
#
# In our real classifier (Session 10), we'll use embeddings
# because they let the model discover patterns we didn't think of.


print("=" * 60)
print("Session 2 Complete!")
print("=" * 60)
print("""
Key concepts:
1. Neural networks need NUMBERS — text must be converted
2. Hand-crafted features: simple, limited, you decide what to count
3. Bag of words: counts every word, but loses order
4. TF-IDF: weights rare words higher — finds distinctive words
5. Embeddings: learned dense vectors — the standard in modern ML
6. The embedding table is ITSELF a tensor of weights that gets trained

Next session: A Single Neuron — one weight, one bias, one prediction
""")
