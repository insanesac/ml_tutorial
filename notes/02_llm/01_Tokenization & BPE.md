# Tokenization & BPE

## What Is Tokenization?

Neural networks cannot process raw text. Transformers operate on numbers using operations such as `Q @ K.T`, `W @ x`, and `Softmax`. Text must first be converted into integer token IDs.

```
"Hello world" -> [15496, 995]
```

These IDs are then mapped to embeddings. The model never sees characters or words directly — only integer IDs.

---

## Why Not Character-Level?

- Very long sequences for any real text
- O(N²) attention cost explodes

**Cost example:**

```
1000 word tokens    → 1,000²  = 1 million attention operations
7000 characters     → 7,000²  = 49 million attention operations
```

Sequence length grows dramatically, making training and inference much more expensive.

- Weak semantic structure: tokenizer sees `p l a y i n g` and must learn from scratch that `play`, `playing`, `played`, `player` are related.

## Why Not Word-Level?

- Huge vocabulary (millions of words)
- Can't handle new words, typos, or rare words
- No subword sharing ("run", "running", "runner" are completely separate)
- OOV problem: `playfulness` → `<UNK>` — loses all useful information

---

## BPE (Byte Pair Encoding)

### Core Idea

Start with characters. Repeatedly merge the most frequent adjacent pair until vocabulary size is reached.

### Algorithm

```
1. Initialize vocabulary as individual characters
2. Count all adjacent pair frequencies
3. Merge the most frequent pair into a new token
4. Repeat until vocab size V is reached
```

### Worked Example

```
Corpus: "low low low lower lower newest newest"

Start:  l o w _ l o w _ l o w _ l o w e r _ l o w e r _ n e w e s t _ n e w e s t

Iteration 1: most frequent pair = ('l','o') -> merge to 'lo'
Iteration 2: most frequent pair = ('lo','w') -> merge to 'low'
Iteration 3: most frequent pair = ('low','_') -> merge to 'low_'
...
```

### Implementation Sketch

```python
import re
from collections import Counter

def get_pairs(vocab):
    pairs = Counter()
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            pairs[(symbols[i], symbols[i+1])] += freq
    return pairs

def merge_pair(pair, vocab):
    new_vocab = {}
    bigram = ' '.join(pair)
    replacement = ''.join(pair)
    for word, freq in vocab.items():
        new_word = word.replace(bigram, replacement)
        new_vocab[new_word] = freq
    return new_vocab

def bpe(vocab, num_merges):
    merges = []
    for _ in range(num_merges):
        pairs = get_pairs(vocab)
        if not pairs:
            break
        best = max(pairs, key=pairs.get)
        vocab = merge_pair(best, vocab)
        merges.append(best)
    return vocab, merges
```

---

## Vocabulary Size

| Model | Vocab Size |
|---|---|
| GPT-2 | 50,257 |
| GPT-4 | ~100,000 |
| LLaMA 3 | 128,000 |

Larger vocab = fewer tokens per sentence = shorter sequences = cheaper attention.

---

## Tokenization Quirks (Interview Traps)

- `"ChatGPT"` might tokenize as `["Chat", "G", "PT"]` — subword splits are not intuitive.
- Numbers are often split digit by digit: `"1234"` -> `["1", "2", "3", "4"]`.
- Whitespace is part of some tokens: `" hello"` ≠ `"hello"`.
- Same word can tokenize differently depending on position/context.

---

## Special Tokens

| Token | Purpose |
|---|---|
| `<BOS>` | Beginning of sequence |
| `<EOS>` | End of sequence |
| `<PAD>` | Padding to fixed length |
| `<UNK>` | Unknown token (rare in BPE) |

---

## WordPiece (Used in BERT)

### How It Differs from BPE

BPE merges the most **frequent** pair. WordPiece merges the pair that **maximizes the likelihood** of the training corpus.

**What does "maximize likelihood" mean?**

Likelihood measures how much probability the model assigns to the data that actually occurred. High likelihood means the tokenizer can represent the observed training corpus naturally and efficiently.

```
Corpus: playing, played, player, playful

Vocabulary A: playing, played, player, playful  (4 large tokens)
Vocabulary B: play, ing, ed, er, ful             (5 reusable subwords)

Vocabulary B represents the corpus much more naturally.
Therefore it gives the corpus a higher likelihood.
```

```
Score(pair) = freq(pair) / (freq(first) * freq(second))
```

This favors pairs that are relatively more common compared to their individual parts.

### Key Differences

| | BPE | WordPiece |
|---|---|---|
| Merge criterion | Most frequent pair | Highest likelihood gain |
| Direction | Bottom-up (merge) | Bottom-up (merge) |
| Used in | GPT-2, GPT-4 | BERT, DistilBERT |
| Unknown tokens | Rare (subword covers most) | Uses `[UNK]` for truly unknown |

---

## SentencePiece

### What It Is

A tokenizer-agnostic framework that treats text as a **raw byte stream** — no word boundaries assumed.

Many languages do not separate words using spaces:

```
Chinese:   我喜欢机器学习
Japanese:  私は機械学習が好きです
```

Traditional tokenizers assume words already exist. SentencePiece removes that assumption.

- Works directly on raw text (no pre-tokenization by whitespace)
- Spaces themselves become ordinary symbols (encoded as `▁`)
- Supports both BPE and Unigram algorithms
- Language-agnostic (handles CJK, Thai, etc. where there are no spaces)
- Used in LLaMA, T5, XLNet

```
▁play  →  the leading ▁ represents whitespace

If you've inspected LLaMA tokens, you've likely seen tokens such as:
▁The  ▁cat  ▁ing
The underscore-like symbol is simply an encoded space.
```

### Why It Matters

```
English:  "Hello world"  ->  whitespace split is natural
Chinese:  "你好世界"      ->  no spaces, need byte-level or char-level
```

SentencePiece sidesteps the whitespace assumption entirely.

---

## Unigram Language Model (Used in T5, ALBERT)

### Core Idea

Start with a **large** vocabulary of candidate subwords. Iteratively **remove** the subwords that least reduce the corpus likelihood.

```
1. Initialize with a large set of candidate subwords
2. Compute likelihood of corpus under current vocabulary
3. Remove the subword whose removal least decreases likelihood
4. Repeat until target vocab size is reached
```

### BPE vs Unigram

| | BPE | Unigram |
|---|---|---|
| Direction | Build up (merge) | Prune down (remove) |
| Determinism | Deterministic (greedy merge) | Probabilistic (multiple valid segmentations) |
| Used in | GPT family | T5, ALBERT |

Unigram can produce multiple valid tokenizations for the same text, which can be useful for data augmentation.

---

## Byte-Level BPE (Used in GPT-2, GPT-4)

### The Problem with Character-Level BPE

Character-level BPE fails on non-ASCII characters, emojis, and multilingual text because the "characters" are Unicode codepoints.

### The Solution

Operate on **bytes** (256 possible values) instead of characters. Every text can be represented as bytes, so there are no unknown characters.

```
Text -> UTF-8 bytes -> BPE merges on byte sequences
```

### Why GPT-2 Uses It

- No `[UNK]` token needed — every byte sequence can be tokenized
- Works for any language, any emoji, any encoding
- The vocabulary is built from 256 bytes + merges

---

## Encoding (Tokenizing New Text)

Once BPE merge rules are learned, encoding new text:

```python
def bpe_encode(text, merges):
    # Start with characters
    tokens = list(text)
    while True:
        pairs = [(tokens[i], tokens[i+1]) for i in range(len(tokens)-1)]
        # Find the pair with the lowest merge rank (earliest learned = highest priority)
        mergeable = [(p, merges.index(p)) for p in pairs if p in merges]
        if not mergeable:
            break
        best_pair = min(mergeable, key=lambda x: x[1])[0]
        # Merge all occurrences
        tokens = merge_in_text(tokens, best_pair)
    return tokens
```

### Key Insight

During encoding, you don't pick the most frequent pair — you pick the pair with the **lowest merge rank** (earliest learned merge). This ensures deterministic, consistent tokenization.

---

## Decoding (Tokens to Text)

```python
def bpe_decode(token_ids, id_to_token, merges):
    tokens = [id_to_token[i] for i in token_ids]
    return ''.join(tokens)  # For byte-level, decode bytes to UTF-8
```

For byte-level BPE (GPT-2/4), you concatenate the byte sequences and decode as UTF-8.

---

## Vocabulary Size

| Model | Vocab Size | Tokenizer |
|---|---|---|
| GPT-2 | 50,257 | Byte-level BPE |
| GPT-4 | ~100,000 | Byte-level BPE (tiktoken) |
| LLaMA 3 | 128,000 | BPE (SentencePiece) |
| BERT | 30,522 | WordPiece |
| T5 | 32,128 | Unigram (SentencePiece) |

Larger vocab = fewer tokens per sentence = shorter sequences = cheaper attention.

But: larger vocab = bigger embedding matrix = more parameters = more memory.

### Embedding Memory Math

```
Embedding params = vocab_size * D_model

GPT-2:  50,257 * 768   = ~38.6M params
LLaMA 3: 128,000 * 4096 = ~524M params (just embeddings!)
```

This is why vocab size is a significant design decision — it directly impacts model size.

---

## Tokenization Quirks (Interview Traps)

- `"ChatGPT"` might tokenize as `["Chat", "G", "PT"]` — subword splits are not intuitive.
- Numbers are often split digit by digit: `"1234"` -> `["1", "2", "3", "4"]`.
- Whitespace is part of some tokens: `" hello"` ≠ `"hello"`.
- Same word can tokenize differently depending on position/context.
- Leading spaces matter: `"The cat"` vs `"cat"` produce different tokens.
- Code tokenization is notoriously bad: indentation, brackets, operators split unpredictably.
- Multi-byte characters (emojis, CJK) may take multiple tokens — costs more and loses semantic unit.
- Tokenization is not reversible in general (information loss from pre-tokenization).

---

## Special Tokens

| Token | Purpose |
|---|---|
| `<BOS>` | Beginning of sequence |
| `<EOS>` | End of sequence |
| `<PAD>` | Padding to fixed length |
| `<UNK>` | Unknown token (rare in BPE/byte-level) |
| `<|endoftext|>` | GPT-2/4 end of text |
| `<|im_start|>` / `<|im_end|>` | ChatML message delimiters (GPT-4) |

### Why Special Tokens Matter for L5

- **Padding**: affects batching efficiency. Left vs right padding matters for generation.
- **EOS detection**: serving systems need to detect EOS to stop generation and free resources.
- **Chat formatting**: `<|im_start|>system\n...<|im_end|>` — token-level conversation structure.
- **Fine-tuning**: custom special tokens for tool use, function calling, structured output.

---

## Complexity

| Operation | Complexity |
|---|---|
| BPE training | O(V * N) where V = merges, N = corpus size |
| Tokenization (inference) | O(N * M) where M = max merge rank depth |
| Byte-level encoding | O(N) bytes -> BPE on byte sequence |

---

## L5 Interview Q&A

### Q: "Walk me through how GPT-4 tokenizes 'Hello, world!'"

1. Text is encoded to UTF-8 bytes.
2. Bytes are mapped to a printable Unicode range (GPT-2's byte-to-unicode trick to avoid control characters).
3. BPE merge rules are applied in rank order — earliest merges first.
4. Each resulting token is looked up in the vocabulary to get its integer ID.
5. The model receives a list of integer IDs.

### Q: "Why does tokenization matter for model quality?"

- **Multilingual fairness**: if a language needs 3x more tokens than English, it costs 3x more to process and the model has less effective context for it.
- **Code generation**: poor tokenization of code (splitting `==`, `!=`, indentation) hurts code completion quality.
- **Math**: digit-by-digit tokenization means the model can't do arithmetic on "multi-token numbers" as easily.
- **Context window**: more tokens per word = less effective context = worse long-context performance.

### Q: "How would you choose a vocabulary size for a new model?"

Tradeoffs:
- **Larger vocab**: shorter sequences, cheaper attention, better for multilingual, but bigger embedding matrix and output projection.
- **Smaller vocab**: longer sequences, more attention cost, but fewer parameters in embeddings.
- **Rule of thumb**: vocab should be large enough that common words are 1-2 tokens, rare words are 3-5 tokens.
- **Multilingual**: need larger vocab to cover multiple scripts efficiently.

### Q: "What's the difference between tiktoken and SentencePiece?"

| | tiktoken | SentencePiece |
|---|---|---|
| Used by | GPT-4, GPT-3.5 | LLaMA, T5 |
| Algorithm | Byte-level BPE | BPE or Unigram |
| Whitespace | Part of tokens (Ġ prefix) | Raw byte stream, no pre-split |
| Speed | Very fast (Rust) | Slower (C++) |
| Language support | All (byte-level) | All (byte-level mode) |

### Q: "Why is tokenization blamed for LLM weaknesses?"

- **Spelling**: if a word is split into subwords, the model doesn't "see" the full word as a unit.
- **Counting**: token-level position ≠ character-level position.
- **Reversing strings**: if "hello" is one token, the model can't reverse it character by character without learning to do so internally.
- **Math**: multi-digit numbers are split, making arithmetic harder.
- **Prompt injection**: tokenizer quirks can create adversarial tokens that bypass filters.

---

## Interview Sound Bites

- BPE balances vocabulary size with sequence length — starts with characters (or bytes), merges most frequent pairs iteratively.
- Byte-level BPE (GPT-2/4) operates on raw bytes, eliminating `[UNK]` tokens and handling any language.
- SentencePiece treats text as a raw byte stream — no whitespace pre-tokenization, making it language-agnostic.
- WordPiece (BERT) merges by likelihood gain, not just frequency.
- Unigram (T5) prunes from a large vocabulary down — probabilistic, multiple valid segmentations.
- Larger vocabulary = shorter sequences = cheaper attention, but bigger embedding matrix.
- Tokenization is the root cause of many LLM quirks: spelling, counting, reversing, math, multilingual fairness.
- Encoding uses merge rank order (earliest merges first), not frequency — this ensures deterministic tokenization.

---

## Summary Comparison

| Feature | Character | Word | Subword |
|---|---|---|---|
| Small vocabulary | ✅ | ❌ | ✅ |
| Short sequences | ❌ | ✅ | ✅ |
| Handles unknown words | ✅ | ❌ | ✅ |

| Tokenizer | Main Idea | Typical Models |
|---|---|---|
| BPE | Merge most frequent adjacent pair | GPT-2 (byte-level BPE), RoBERTa |
| WordPiece | Merge pair that maximizes corpus likelihood | BERT |
| SentencePiece | Learn subwords directly from raw text without assuming words | LLaMA, T5, ALBERT |
