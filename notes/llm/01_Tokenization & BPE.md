# Tokenization & BPE

## What Is Tokenization?

Converting raw text into token IDs that the model can process.

```
"Hello world" -> [15496, 995]
```

The model never sees characters or words directly — only integer IDs.

---

## Why Not Character-Level?

- Very long sequences for any real text
- O(N²) attention cost explodes

## Why Not Word-Level?

- Huge vocabulary (millions of words)
- Can't handle new words, typos, or rare words
- No subword sharing ("run", "running", "runner" are completely separate)

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

## Complexity

| Operation | Complexity |
|---|---|
| BPE training | O(V * N) where V = merges, N = corpus size |
| Tokenization (inference) | O(N) with precomputed merge rules |

---

## Interview Sound Bites

- BPE balances vocabulary size with sequence length.
- Starts with characters, merges most frequent pairs iteratively.
- Handles rare words and morphology through subword sharing.
- Larger vocabulary = shorter sequences = cheaper attention at inference.
- Tokenization artifacts (split numbers, whitespace sensitivity) are a known weakness.
