# LoRA Forward Pass

## Recap

LoRA keeps base weight `W` frozen and learns low-rank adapters `A` and `B`.

`ΔW = A @ B`, where rank `r << D`.

## Forward Equation

For input `x`:

`y = x @ (W + α/r * A @ B)`

Equivalent split form:

`y = xW + (xA)B * α/r`

This split is computationally convenient.

## Shapes

- `x`: `(B, D_in)`
- `W`: `(D_in, D_out)`
- `A`: `(D_in, r)`
- `B`: `(r, D_out)`
- output `y`: `(B, D_out)`

## NumPy Implementation

```python
def lora_forward(x, W, A, B, alpha=1.0):
    r = A.shape[1]
    base = x @ W
    lora = (x @ A) @ B
    return base + (alpha / r) * lora
```

## Training Behavior

- Freeze `W`
- Train `A` and `B`
- Usually initialize `B` to zeros so initial behavior matches base model

## Inference Merge

For deployment, you can merge once:

`W_merged = W + α/r * A @ B`

Then use normal linear layer with no runtime adapter overhead.

## Where Used

Commonly applied in attention projections:

- `Wq`, `Wk`, `Wv`, `Wo`

Modern best practice: apply to **all linear layers** — attention + FFN (gate, up, down).

---

## Initialization Deep Dive

### Why A is Gaussian and B is Zero

```
A ~ N(0, σ²)    # Random Gaussian initialization
B = 0           # Zero initialization
```

**Critical**: This ensures that **at the start of training**, `BA = 0`, so the LoRA output is zero and the model behaves exactly as the base model. Training then gradually learns the delta.

If both A and B were random, the initial `BA` would be non-zero, immediately perturbing the base model's behavior and causing unstable training.

### Why Not Both Zero?

If both A and B were zero, the gradient of B would be:

```
∂L/∂B = A^T · (∂L/∂y) · x^T
```

With `A = 0`, the gradient of B is zero — B never learns. A must be non-zero to break symmetry and allow gradients to flow.

### Summary

| | A | B |
|---|---|---|
| Initialization | Gaussian (random) | Zero |
| Role | Down-projection (feature extraction) | Up-projection (feature reconstruction) |
| Why | Breaks symmetry, allows gradient flow | Ensures initial delta = 0, stable start |

---

## Gradient Flow in LoRA

### Forward

```
y = Wx + (α/r) * BAx
```

### Backward

```
∂L/∂B = (α/r) * (∂L/∂y) * (Ax)^T     # gradient for B
∂L/∂A = (α/r) * B^T * (∂L/∂y) * x^T  # gradient for A
∂L/∂W = 0                              # frozen, no gradient
```

### Key Observations

1. **W is frozen**: no gradient flows to W — it's not in the computational graph for backprop.
2. **A and B gradients depend on each other**: `∂L/∂B` depends on A, `∂L/∂A` depends on B.
3. **Scaling factor (α/r)**: controls the effective learning rate of the adapter.
4. **Gradient magnitude**: since `r << D`, the gradients are low-dimensional — this is why LoRA trains fast.

### Why LoRA Trains Fast

- Only `2 * r * D` parameters per layer (vs `D²` for full fine-tuning).
- Optimizer state (Adam: momentum + variance) is also `2 * r * D` — much smaller.
- Fewer parameters → faster convergence, less memory, smaller optimizer state.

---

## Multi-Adapter Serving

### The Idea

Serve a single base model with **multiple LoRA adapters** — swap adapters per request without reloading the model.

```
Request 1 (coding task):    base model + coding_adapter
Request 2 (math task):      base model + math_adapter
Request 3 (general chat):   base model + general_adapter
```

### Implementation

```python
class MultiAdapterServing:
    def __init__(self, base_model, adapters):
        self.base_model = base_model  # loaded once
        self.adapters = adapters      # {name: (A, B)} per layer

    def forward(self, x, adapter_name):
        for layer in self.base_model.layers:
            W = layer.weight  # frozen
            A, B = self.adapters[adapter_name][layer.idx]
            x = W @ x + (self.alpha / self.r) * B @ A @ x
        return x
```

### Benefits

- **Memory**: one base model + many small adapters (each ~10-100 MB).
- **Flexibility**: switch behavior per request without reloading.
- **Cost**: much cheaper than serving multiple full models.

### Production Considerations

- **Batching with different adapters**: tricky — each request in a batch may use a different adapter. Requires per-request adapter application within the batch.
- **vLLM LoRA support**: vLLM supports multi-LoRA serving with batched requests.
- **Adapter hot-swapping**: load/unload adapters at runtime without restarting the server.

---

## LoRA Applied to Different Layer Types

### Attention Layers

| Layer | Shape | LoRA Params (r=16) |
|---|---|---|
| W_q (query projection) | (D, D) | 2 * 16 * D |
| W_k (key projection) | (D, D) | 2 * 16 * D |
| W_v (value projection) | (D, D) | 2 * 16 * D |
| W_o (output projection) | (D, D) | 2 * 16 * D |

### FFN Layers (SwiGLU)

| Layer | Shape | LoRA Params (r=16) |
|---|---|---|
| W_gate | (D, D_ff) | 2 * 16 * (D + D_ff) |
| W_up | (D, D_ff) | 2 * 16 * (D + D_ff) |
| W_down | (D_ff, D) | 2 * 16 * (D_ff + D) |

### Parameter Count Example (LLaMA-7B, r=16, all linear)

```
D = 4096, D_ff = 11008, n_layers = 32

Per layer attention: 4 * 2 * 16 * 4096 = 524,288
Per layer FFN: 3 * 2 * 16 * (4096 + 11008) = 1,835,008
Per layer total: 2,359,296

Total: 32 * 2,359,296 = 75,497,472 ≈ 75M params (0.1% of 7B)
```

---

## L5 Interview Q&A

### Q: "Why does LoRA use two matrices (A and B) instead of one trainable matrix?"

A single matrix `ΔW` of shape `(D, D)` would have `D²` parameters — that's full fine-tuning. The factorization `B @ A` with `A: (r, D)` and `B: (D, r)` gives only `2rD` parameters while still producing a `(D, D)` update. This is the **low-rank hypothesis**: the weight update during fine-tuning has low intrinsic rank.

### Q: "What is the low-rank hypothesis and is it validated?"

The hypothesis: pretrained models already have high-rank weights, but the **adaptation delta** (what you need to learn for a new task) is low-rank. This was empirically validated by the LoRA paper — effective rank of fine-tuning updates is very low (often < 16).

### Q: "How does the scaling factor α/r affect training?"

- `α/r` acts as the effective learning rate for the adapter.
- Higher `α/r` → larger updates → faster adaptation but risk of instability.
- Lower `α/r` → smaller updates → more stable but slower.
- Common choice: `α = 2r` → `α/r = 2` — a good default.

### Q: "Can you stack multiple LoRA adapters?"

Yes — you can add multiple LoRA updates:

```
y = Wx + α₁/r₁ * B₁A₁x + α₂/r₂ * B₂A₂x
```

This is useful for:
- **Multi-task**: different adapters for different capabilities.
- **Progressive**: add new adapters on top of existing ones.
- **Composition**: combine adapters for new tasks (e.g., coding + math).

### Q: "How does LoRA interact with gradient checkpointing?"

Gradient checkpointing recomputes forward activations during backward to save memory. With LoRA:
- Base model activations are checkpointed (frozen weights, deterministic).
- LoRA activations are small (low-rank) — no need to checkpoint.
- **Result**: LoRA + gradient checkpointing = very memory-efficient training.

### Q: "What's the difference between LoRA and adapter layers?"

- **LoRA**: modifies existing weight matrices via low-rank addition. No extra layers, no architecture change.
- **Adapters**: adds new small bottleneck layers between transformer blocks. Changes architecture.
- **LoRA is preferred** because it can be merged into weights (zero inference overhead) and doesn't change the model architecture.

---

## Interview Sound Bites

- LoRA changes trainable params from `O(D²)` to `O(D*r)` — typically 0.1% of model.
- Forward pass is base linear output plus low-rank residual: `y = Wx + (α/r) * BAx`.
- `A` is Gaussian init (breaks symmetry), `B` is zero init (ensures stable start).
- Adapter merge gives zero inference overhead: `W_merged = W + (α/r) * BA`.
- **Multi-adapter serving**: one base model + many small adapters, swap per request.
- Low-rank hypothesis: fine-tuning updates have low intrinsic rank — validated empirically.
- LoRA > adapter layers because LoRA can be merged (no architecture change, zero inference overhead).
- LoRA + gradient checkpointing = extremely memory-efficient fine-tuning.
- Apply to all linear layers (not just attention) for best quality.
