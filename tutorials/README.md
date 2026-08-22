# ViT ONNX Tutorial

## Architecture

The model is a single Transformer encoder block with three attention heads:

```text
Input tokens
  -> LayerNorm
  -> Q/K/V projections
  -> split into 3 attention heads
  -> QK^T
  -> scale by 1/sqrt(64) = 1/8
  -> Softmax
  -> attention times V
  -> merge heads
  -> attention output projection
  -> residual Add
  -> LayerNorm
  -> MLP: 192 -> 768 -> 192
  -> residual Add
  -> final LayerNorm
  -> classification head: 192 -> 16
```

The current constants are:

| Symbol | Value |
| --- | ---: |
| Batch | 1 |
| Physical sequence length | 256 |
| Embedding dimension | 192 |
| Number of heads | 3 |
| Head dimension | 64 |
| MLP hidden dimension | 768 |
| Real classes | 10 |
| Padded output classes | 16 |

## Tensor Flow

The model input:

```text
x_i8: (1, 256, 192), int8
  -> DequantizeLinear
x_dq: (1, 256, 192), float
```

The first 197 tokens represent the intended ViT sequence: token 0 is the class token and tokens 1 through 196 are image patch tokens. Tokens 197 through 255 are physical padding.

Note: The ONNX graph does not create patch embeddings, class tokens, or positional embeddings.

### Q/K/V projections

Each projection is represented as:

```text
MatMul(input, weight)
  -> Add(bias)
  -> QuantizeLinear / DequantizeLinear
```

Shapes before head splitting:

```text
Q: (1, 256, 192)
K: (1, 256, 192)
V: (1, 256, 192)
```

Each projection is then converted into head-major layout:

```text
(1, 256, 192)
  -> Reshape  (1, 256, 3, 64)
  -> Transpose (1, 3, 256, 64)
```

The layout is:

```text
(batch, heads, tokens, head_features)
```

### Attention scores

K is transposed over its final two dimensions:

```text
K:  (1, 3, 256, 64)
KT: (1, 3, 64, 256)
```

The score MatMul is:

```text
Q  @ KT
(1, 3, 256, 64) @ (1, 3, 64, 256)
= (1, 3, 256, 256)
```

The graph applies the scale before score quantization:

```text
raw QK^T
  -> Mul(1/8)
  -> score QuantizeLinear / DequantizeLinear
  -> Softmax over the final key-token axis
```

This ordering matches the Keras expression:

```text
QMatMul(scale=1/sqrt(64)) -> QActivation(SCQ)
```

### Attention output and head merge

The attention probabilities multiply V:

```text
(1, 3, 256, 256) @ (1, 3, 256, 64)
= (1, 3, 256, 64)
```

The context is merged back to the model dimension:

```text
(1, 3, 256, 64)
  -> Transpose (1, 256, 3, 64)
  -> Reshape  (1, 256, 192)
```

The output projection preserves the shape:

```text
(1, 256, 192) -> (1, 256, 192)
```

### Residual and MLP path

The attention output is added to the original input:

```text
(1, 256, 192) + (1, 256, 192)
= (1, 256, 192)
```

The MLP operates independently on each token:

```text
(1, 256, 192)
  -> MatMul + bias: (1, 256, 768)
  -> ReLU:           (1, 256, 768)
  -> MatMul + bias: (1, 256, 192)
```

The second residual Add again produces:

```text
(1, 256, 192)
```

### Classification output

The final LayerNorm preserves the shape. The classification head produces 16 values per token for PLIO alignment:

```text
(1, 256, 192) -> (1, 256, 16)
```

Only 10 values are real classes:

```python
logits = output[:, 0, :10]
probs = softmax(logits)
predicted_class = argmax(probs)
```

The graph output is therefore a sequence of padded logits (not a class result).

## Quantization Mapping

QKeras specifies total bit width and integer bit count. For signed `quantized_bits`, fractional bits are:

```text
fractional_bits = total_bits - integer_bits - sign_bit
```

The intended mapping is:

| Tensor | Keras quantizer | Fractional bits |
| --- | --- | ---: |
| Kernels | `quantized_bits(8, 0)` | 7 |
| Biases | `quantized_bits(8, 2)` | 5 |
| Generic activations | `quantized_bits(8, 2)` | 5 |
| Attention probabilities | `quantized_relu(8, 0)` | 8 |
| Attention scores | `quantized_bits(8, 4)` | 3 |
| Final logits | `quantized_bits(8, 4)` | 3 |

For example:

```text
quantized_bits(8, 4)
  -> 8 total bits
  -> 4 integer bits
  -> 1 sign bit
  -> 3 fractional bits
  -> scale = 2^-3
  -> range approximately [-16, 16)
```

Note: Quantization must be applied to the float token input using the same input scale before supplying `x_i8`.

## Parameters and Correctness

The ONNX script creates deterministic synthetic parameters with NumPy. It does not contain trained Keras weights:

```python
np.random.default_rng(0)
```

Consequently, the ONNX graph can be tested against its own ONNX Runtime reference, but it cannot reproduce the output of a separate Keras model unless all of the following are identical:

- input tokens,
- Q/K/V weights and biases,
- output projection weights and bias,
- MLP weights and biases,
- classification weights and bias,
- LayerNorm parameters,
- quantization scales and rounding,
- padding and masking behavior.

No Keras library is needed for the current synthetic ONNX test. Keras is only needed if a real Keras model is later provided and its parameters must be exported.

## Padding and Masking

The physical sequence length is 256, while the intended real ViT sequence is 197 tokens. 
Note: The current ONNX graph has no attention mask. Therefore padded keys participate in Softmax unless masking is added explicitly.

To implement the documented padded-key behavior, the graph would need an additional step before Softmax:

```text
scaled attention scores
  -> add a large negative value to padded key columns
  -> Softmax
```

The same mask must be used by both the reference path and the AIE path for numerical comparison. The visible Keras example also does not pass an actual mask, despite describing one in comments.

## AIE Workarounds

The Keras example uses custom `QMatMul` layers and an AIE extension that decomposes batched multi-head operations into AIE-compatible per-head rank-2 MatMuls. The ONNX graph expresses the same architecture using standard ONNX operators, which is useful for ONNX Runtime but exposes additional frontend requirements.

| Workaround | Reason |
| --- | --- |
| Sequence padded from 197 to 256 | Makes cascade and tile splits divide cleanly and keeps ports within producer buffers |
| Output padded from 10 to 16 classes | Makes the int8 output width align to a 128-bit PLIO word |
| Input is pre-embedded | Avoids adding patch projection, class-token creation, and positional embedding to this test graph |
| HCCS Softmax directives | The AIE Softmax implementation requires explicit HCCS configuration |
| `1/sqrt(64) = 1/8` | Power-of-two scale can be implemented using shift-based AIE scaling |
| Reduced encoder depth | Keeps the test graph smaller; the current ONNX graph contains one encoder block |
| Synthetic deterministic parameters | Provides reproducible ONNX Runtime and AIE test inputs without a Keras checkpoint |

## AIE Frontend Constraints

The graph can be valid ONNX while still being unsupported by the local AIE frontend. There are two meanings of correctness:

- **Correct ONNX answer:** achievable after fixing quantization settings and applying the correct output slicing.
- **Correct AIE answer:** additionally requires frontend support or decomposition for `Reshape`, rank-4 `MatMul`, and the required transpose layouts.

The current implementation status is:

| Priority | Item | Status | Why the rest can fail without it | What is needed |
| ---: | --- | --- | --- | --- |
| 1 | Correct QKeras-to-ONNX fractional-bit mapping | Fixed | Different scales round activations, attention scores, probabilities, or logits differently, so the ONNX reference is numerically different. | Keep the current mapping: kernels 7, biases 5, activations 5, scores 3, attention probabilities 8, and final head 3 fractional bits. |
| 2 | Use identical deterministic parameters for reference and AIE | Needed for Keras comparison | Different weights or biases produce different answers even when every operator and shape is correct. | Use the same ONNX initializers and input data for ONNX Runtime and AIE. A Keras checkpoint is not required for the synthetic test. |
| 3 | Decide whether padded keys need masking | Open design choice | With 256 physical tokens and 197 real tokens, unmasked padding participates in Softmax and changes attention output. | Either accept the current unmasked behavior in both references, or add the same pre-Softmax padded-key mask to both paths. |
| 4 | Add ONNX `Reshape` frontend support | Not implemented | Without Reshape, Q/K/V remain rank-3 and cannot become `(batch, heads, tokens, head_dim)`; the AIE importer stops at `reshape_q`. | Add a constant-shape Reshape handler and fold it as a view, or lower the graph into explicit per-head rank-2 operations. |
| 5 | Decompose rank-4 attention into AIE-supported rank-2 operations | Not yet verified | ONNX Runtime supports broadcasted rank-4 MatMul, but the AIE execution model may only accept 2-D tensor contracts. | Split the three heads, run separate rank-2 QK and attention-value MatMuls, then concatenate or reshape the results. |
| 6 | Handle non-final-axis Transpose layouts | Not yet verified | The graph uses permutations such as `[0, 2, 1, 3]` and `[0, 1, 3, 2]`; the backend's tensor-view rules are more restrictive than ONNX Runtime. | Extend tensor-view/layout lowering for these permutations, or remove them by emitting AIE-compatible per-head rank-2 graphs. |
| 7 | Slice token 0 and the first 10 channels after inference | Required for class prediction | The graph returns 256 token rows and 16 aligned channels; treating all 16 channels or all rows as classes gives the wrong prediction shape. | Apply `output[:, 0, :10]`, then calculate probabilities and `argmax`. |
| 8 | Make LayerNorm width AIE-compatible | Current blocker | The AIE LayerNorm implementation requires a power-of-two normalized width; `DIM = 192` stops compilation before attention lowering. | Use a power-of-two model width, such as 256, or add a padded LayerNorm implementation that preserves the intended 192-feature semantics. |
| 9 | Account for HCCS Softmax approximation | AIE numerical limitation | HCCS is not ordinary exponential Softmax, so an AIE result will not generally equal the Keras/ONNX floating-point Softmax bit-for-bit. | Compare using an agreed tolerance or use an AIE-compatible reference Softmax. |

The first AIE blocker currently encountered is LayerNorm with 192 columns. Once that is resolved, the importer must still reach and validate the Reshape, rank-4 MatMul, and transpose paths.

The practical AIE implementation choices are either:

```text
Option A: add Reshape/rank-4 layout support to the ONNX frontend
```

or:

```text
Option B: lower each attention head to explicit rank-2 ONNX operations,
          similar to the custom Keras AIE extension
```

Option B is more compatible with the current frontend constraints, but it produces a larger graph.

## Where Reshape Support Goes

The ONNX `Reshape` operator changes logical shape without changing element order or values. To support it in the AIE path, modify `src/aie4ml/frontends/onnx/handlers/tensor.py` to register `@onnx_handler('Reshape')`, validate its constant shape input and element count, and emit a logical `reshape` node. Modify `src/aie4ml/passes/fold_views.py` to fold that node into the producer/view relationship so no standalone AIE Reshape kernel is generated. `handlers/__init__.py` already imports `tensor`, and `pipeline.py` already runs `FoldViewOps` before `Resolve`, so those files normally need verification rather than code changes.

The lowering flow is:

```text
ONNX Reshape(data, shape_initializer)
  -> handler dispatch in handlers/tensor.py
  -> validate constant target shape and element count
  -> ctx.emit('reshape', ...)
  -> FoldViewOps folds the logical view
  -> Resolve plans the following AIE operation using the new shape
  -> no standalone Reshape kernel
```

For this ViT graph, Reshape is used for:

```text
(1, 256, 192) -> (1, 256, 3, 64)   # Q/K/V head split
(1, 256, 3, 64) -> (1, 256, 192)     # merged context
```

A Reshape handler alone does not guarantee AIE compilation. Rank-4 MatMul and transpose layout contracts must also be supported or decomposed into per-head rank-2 operations.

## Running All Model Generators

Run from the repository root with the project environment:

```bash
./.venv/bin/python tutorials/onnx_test_models/dense.py
./.venv/bin/python tutorials/onnx_test_models/dense_softmax.py
./.venv/bin/python tutorials/onnx_test_models/dense_ln.py
./.venv/bin/python tutorials/onnx_test_models/dense_softmax_ln.py
./.venv/bin/python tutorials/vit_onnx.py
```

Or run all five in one command:

```bash
for script in \
    tutorials/onnx_test_models/dense.py \
    tutorials/onnx_test_models/dense_softmax.py \
    tutorials/onnx_test_models/dense_ln.py \
    tutorials/onnx_test_models/dense_softmax_ln.py \
    tutorials/vit_onnx.py; do
    ./.venv/bin/python "$script" || exit 1
done
```

## Validation

The current script is validated as an ONNX model generator:

```bash
./.venv/bin/python tutorials/vit_onnx.py
```

This runs `onnx.checker.check_model(model)` and writes:

```text
tutorials/onnx_models/vit_onnx.onnx
tutorials/onnx_models/vit_onnx.json
```

ONNX Runtime execution has also been verified with an input of shape `(1, 256, 192)` and dtype `int8`, producing output shape `(1, 256, 16)` and dtype `float32`.

The separate AIE bit-exact checker does not currently complete for this graph. Its first known blocking condition is the AIE LayerNorm requirement that the normalized feature count be a power of two; the current feature count is 192.
