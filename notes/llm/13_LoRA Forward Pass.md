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

## Interview Sound Bites

- LoRA changes trainable params from `O(D^2)` to `O(D*r)`.
- Forward pass is base linear output plus low-rank residual.
- Adapter merge gives cheap inference.
