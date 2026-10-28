# Assignment 1 — Retrospective Notes

This is a recovery-oriented set of notes: the questions and errors that were
hard the first time, organized in the order I encountered the corresponding
functions.  It is intentionally about invariants and mental models, rather
than a copy of an implementation.

## 1. `rope`

### What was difficult / mistakes I made

- I was unsure whether the position and frequency indices should start at 0 or
  1. I worried that position 0 would make the first row “all zero.”
- I mixed up a zero **angle** with a zero **rotated vector**. This led me to
  expect the position-0 output to vanish or duplicate values.
- I first tried to build a full rotation matrix for every position, then got
  stuck on how to place values on its diagonals and how to avoid Python loops.
- I made the pairwise rotation tensor with an unwanted layout: concatenating
  values put them in an existing dimension when I actually needed a new
  dimension. This produced a plausible shape with wrong element order.
- When a test failed, I tried to infer the convention from one expected value
  before checking the position tensor and the pair ordering end-to-end.

### Durable model

RoPE rotates every adjacent pair of features in a query or key vector. For a
pair `(x_even, x_odd)` at token position `p`, its angle depends on both `p` and
the pair's frequency index. Position 0 gives angle 0, so its rotation is the
identity: the vector is unchanged, not zero.

### Formula / shape anchor

For pair index `i`, write the angle as

`φ(p, i) = p · θ^(-2i / d_k)`.

In multi-head attention here, `d_k` is the per-head key/query width: it is the
same quantity as `d_h = d_model / num_heads`, not the full `d_model` width.

With adjacent-pair ordering `(x[2i], x[2i+1])`, the rotated pair is

`(x[2i] cos φ - x[2i+1] sin φ, x[2i] sin φ + x[2i+1] cos φ)`.

The input and output have the same shape; only adjacent values in the final
feature dimension are mixed.

The frequency/pair index convention is zero-based. The relevant question is
not “does the first row contain zero?” but “does the p = 0 rotation leave every
pair unchanged?”

### Concept Q&A

**Q: Should the RoPE indices begin at 0 or 1?**  
**A:** Begin at 0. The first token's rotation is intentionally identity.

**Q: What does `d_k` mean in multi-head RoPE?**  
**A:** The width of one head's Q/K vector (`d_h`), after the packed projection
has been reshaped into heads.

**Q: Why can a zero angle still give nonzero output?**  
**A:** `cos(0) = 1` and `sin(0) = 0`, so the rotation returns the original
pair.

**Q: Must I materialize a `d_k × d_k` matrix per token?**  
**A:** No. Treat the last feature dimension as pairs and apply the pairwise
rotation with tensor operations. This is both clearer and avoids the loop/matrix
construction trap.

**Q: What should I inspect when RoPE numerics look almost right?**  
**A:** Check, in this order: token positions, frequency index convention,
adjacent-pair ordering, and the dimension along which values were stacked.

**Debug invariant:** applying RoPE at position 0 must return the input exactly
(up to floating-point precision).

## 2. `multihead_self_attention`

### What was difficult / mistakes I made

- I interpreted projection weights of shape `(d_model, d_model)` as though I
  needed to divide the projection matrices themselves into heads.
- I was uncertain whether to split the input, Q/K/V weights, or output
  projection weight. That confused the packed representation with the
  per-head representation.
- I tried stacking attention outputs in an order that did not match the output
  projection's expected feature layout.
- I initially reasoned that self-attention did not need a mask. For a causal
  language model, that would expose future tokens.
- I assumed a Python loop over heads was unavoidable, instead of first making
  the head axis explicit and using batched tensor operations.

### Durable model

The Q, K, and V projections are each full `d_model → d_model` maps. Their
output is a packed collection of all heads. Project first, then reshape the
last dimension into `num_heads × d_head`; do not divide the learned projection
matrices. Attention scores are computed independently along the new head axis.
After attention, restore the packed `d_model` feature axis before applying the
output projection.

For autoregressive LM attention, the causal mask is applied to scores before
softmax. A masked future position must contribute zero probability, not merely
have its value vector removed after softmax.

This assignment's language-model attention functions are causal because their
job is next-token prediction. “Self-attention” in a generic encoder is not
automatically causal; it can attend bidirectionally when the architecture's
contract permits it.

### Formula / shape anchor

For input `X: [B, T, D]`, projected Q/K/V initially have `[B, T, D]`; reshape
each to `[B, H, T, d_h]`, where `d_h = D / H`. For each head,

`scores = Q Kᵀ / sqrt(d_h) + causal_mask`,  
`weights = softmax(scores, key-axis)`,  
`head_output = weights V`.

The head output is `[B, H, T, d_h]`. Merge heads back to `[B, T, D]` before the
output projection.

### Concept Q&A

**Q: Do I divide Q/K/V or the input by the number of heads?**  
**A:** No. The head count changes the reshape: `d_head = d_model / num_heads`.
The learned projection remains a full `d_model × d_model` map.

**Q: Where does the output projection go?**  
**A:** After the per-head attention results have been put back into one
`d_model`-wide representation.

**Q: Is a causal mask needed for self-attention?**  
**A:** It depends on the model's contract. It is required for next-token,
autoregressive language modelling; encoder-style bidirectional self-attention
does not use one.

**Q: Where should the head dimension live conceptually?**  
**A:** It is an explicit axis after Q/K/V projection and before score/mask
operations, alongside batch and sequence axes.

**Debug invariant:** for causal attention, every probability at a future key
position is zero, and each unmasked score row's probabilities sum to one.

## 3. `multihead_self_attention_with_rope`

### What was difficult / mistakes I made

- I asked whether RoPE should be applied to attention output or final output.
- I considered rotating Q, K, and V, then splitting heads, mostly to reuse a
  prior helper without checking the mathematical contract.
- The 1-D RoPE test positions and 2-D batched-attention positions made it seem
  as though their semantics differed.
- I used `squeeze()` to force token positions into a convenient shape. That
  can silently remove the batch axis when batch size is 1.

### Durable model

RoPE changes the similarity calculation: rotate Q and K before their dot
product. Do not rotate V, attention probabilities, or the final output.
Position data may be shared across the batch (one-dimensional) or supplied per
example (batch by sequence); these are the same idea with different broadcast
shapes. Preserve axis meaning rather than removing singleton axes blindly.

### Concept Q&A

**Q: Which tensors receive RoPE?**  
**A:** Q and K only, before attention scores are computed.

**Q: Can every batch item have different positions or effective lengths?**  
**A:** Yes. A batch-shaped position tensor supports that; a single sequence of
positions can also be broadcast to every batch item.

**Q: Why is unrestricted `squeeze()` risky here?**  
**A:** A size-1 batch is still a batch. Removing it changes the meaning and
broadcasting behavior of the tensor.

**Debug invariant:** with all positions equal to zero, RoPE attention reduces
to the same Q/K geometry as ordinary attention.

## 4. `rmsnorm`

### What was difficult / mistakes I made

- I needed to re-establish where RMSNorm belongs before assembling a transformer
  block.
- The hard part was distinguishing feature-wise scaling from normalization
  across batch or sequence positions.

### Durable model

RMSNorm normalizes each token's `d_model` feature vector by its root mean
square, with epsilon for numerical stability, then applies a learned
feature-wise scale. Unlike LayerNorm, it does not subtract the mean. Batch and
sequence dimensions identify independent token vectors; normalization is across
the final feature dimension.

### Formula / shape anchor

For feature vector `x` and learned scale `g`,

`RMS(x) = sqrt(mean_i(x_i²) + ε)`,  
`RMSNorm(x)_i = g_i · x_i / RMS(x)`.

### Concept Q&A

**Q: What is RMSNorm relative to LayerNorm?**  
**A:** It uses RMS magnitude but no mean-centering step.

**Q: Which axis is normalized?**  
**A:** The model-feature axis for each individual token representation.

**Debug invariant:** ignoring epsilon, multiplying a nonzero token vector by a
positive scalar leaves its RMS-normalized value unchanged before learned
feature-wise scaling; the result has the same shape as the input.

## 5. `softmax` and `cross_entropy`

### What was difficult / mistakes I made

- I had the correct target-logit idea but the loss test failed because numerical
  stability and reduction axes still mattered.
- I wondered whether subtracting the maximum should affect both terms of
  `-x_target + log(sum(exp(x)))`.
- I asked why a stable softmax was needed if cross-entropy can be written in
  terms of logits directly.
- I had to confirm that the maximum is taken over the innermost class/vocabulary
  axis, independently for each example, rather than globally.

### Durable model

Cross-entropy needs the negative log probability of the target class. Compute
the log-normalizer stably with the log-sum-exp idea: subtract a per-example
maximum before exponentiating. The same shift must be reflected consistently
in the target-logit and normalizer expressions, so it cancels algebraically
while preventing overflow numerically.

It is often better to compute the stable log-softmax/cross-entropy expression
directly than to materialize probabilities. “Mathematically equivalent” does
not mean “equally safe in floating point.”

### Formula / shape anchor

For logits `x` and target class `y`, let `m = max_j x_j`, taken over classes.
Then

`loss = -(x_y - m - log(sum_j exp(x_j - m)))`.

Average this scalar over independent examples/tokens only after computing each
example's class-axis loss.

### Concept Q&A

**Q: Why subtract the maximum?**  
**A:** It preserves the exact softmax/cross-entropy value algebraically but
keeps exponentials in a safe numerical range.

**Q: Is the maximum global?**  
**A:** No. Take it over classes/vocabulary for each independent leading-index
location.

**Q: If I use the cross-entropy identity, do I still need softmax?**  
**A:** Not as a separately materialized tensor. You still need its stable
normalization logic through log-sum-exp.

**Debug invariant:** adding the same constant to every logit of one example
must not change its softmax probabilities or cross-entropy loss.

## 6. `transformer_block`

### What was difficult / mistakes I made

- I lost track of the block structure while several helpers were unfinished.
- Tests failed when the correct pieces were composed in the wrong order,
especially around normalization and residual connections.

### Durable model

The block is a sequence of two residual sublayers: attention and feed-forward.
For the assignment's pre-normalization architecture, normalize before each
sublayer, add that sublayer's output back to its input, then pass the resulting
representation onward. RoPE belongs inside the attention sublayer; the
feed-forward network transforms each position independently.

### Data-flow anchor

For block input `x`, the pre-norm residual flow is:

`x_attn = x + Attention(RMSNorm_attn(x))`  
`x_out = x_attn + FFN/SwiGLU(RMSNorm_ffn(x_attn))`.

The feed-forward/SwiGLU computation acts independently at every sequence
position. The LM's final RMSNorm is outside this repeated block, after the last
block and before the LM head.

### Concept Q&A

**Q: What is the basic block structure?**  
**A:** Attention sublayer with a residual connection, followed by feed-forward
sublayer with another residual connection; each uses its designated
normalization position.

**Q: Why does order matter if all pieces are present?**  
**A:** Residual paths preserve a particular signal, and pre-norm versus
post-norm changes the function and the values seen by every later operation.

**Debug invariant:** if an attention or FFN sublayer produced zero, its
residual stage would return its input unchanged.

## 7. `transformer_lm`

### What was difficult / mistakes I made

- I understood the transformer stack but was unsure why the final model needed
  `lm_head.weight`.
- I described the final map as fitting activations into “meaningful digits,”
which hid the probabilistic meaning of the output.

### Durable model

The language-model head is the final linear map from each `d_model` hidden
state to one logit per vocabulary item. Its output is not a token ID and not a
probability: it is an unnormalized score distribution from which the loss or a
decoder can choose the next token.

### Data-flow anchor

`token IDs → token embeddings → repeated Transformer blocks → final RMSNorm →
LM head → logits [B, T, vocab_size]`.

### Concept Q&A

**Q: Why is an LM head required after the transformer blocks?**  
**A:** Blocks output hidden features of width `d_model`; next-token prediction
requires a score for every vocabulary item.

**Q: Are the head's outputs meaningful token numbers?**  
**A:** No. They are logits. Softmax and target selection give them a
probabilistic interpretation.

**Debug invariant:** the last dimension of LM output is vocabulary size, not
`d_model`.

## 8. `gradient_clipping`

### What was difficult / mistakes I made

- I needed the conceptual distinction between clipping each gradient separately
  and clipping the model's combined gradient.

### Durable model

Global L2 gradient clipping measures the single L2 norm formed by all
parameters' gradients. If that combined norm exceeds the threshold, multiply
every participating gradient by the same factor. This preserves the overall
gradient direction while limiting its magnitude; it is not a per-tensor cap.

### Concept Q&A

**Q: Why clip gradients?**  
**A:** To prevent an unusually large update from destabilizing training.

**Q: Does each parameter get its own maximum norm?**  
**A:** Not for global-norm clipping. There is one norm and one shared scale
factor across all available gradients.

**Debug invariant:** when the global gradient norm is already at or below the
threshold, no gradient should change.

## 9. `AdamW`

### What was difficult / mistakes I made

- I initially thought the adapter could simply return the built-in PyTorch
  optimizer, then later confused returning an optimizer **class** with creating
  an optimizer **instance** before it has parameters.
- I expected `super().__init__()` alone to make the optimizer work, without a
  clear model of the base class's parameter groups and state handling.
- I did not initially understand why `param_groups` is a list or why a model
  might use more than one group.
- I questioned the need for a step counter, whether it belongs to a parameter
  or group, and whether a derived step size should be cached.
- I mixed decoupled weight decay with L2 regularization added to the gradient.
- I briefly mixed the two algebraically equivalent bias-correction conventions:
  using a bias-corrected learning-rate factor *and* bias-corrected moments.

### Durable model

An optimizer is constructed with parameters. Its parameter groups are lists
because different groups can have different hyperparameters (for example,
weight decay or learning rate). Moment state belongs to each parameter because
some parameters may not receive a gradient on every optimizer step.

AdamW keeps two running tensors per parameter: first and second gradient
moments, plus a step count. The count supports bias correction. Read the
learning rate from the parameter group on every step: a warmup/cosine scheduler
can change it. Derived quantities such as the bias corrections should be
recomputed, not stored as separate optimizer state.

Weight decay in AdamW is decoupled from the adaptive gradient calculation: it
shrinks the parameter separately, scaled by the current learning rate. For
bias correction, choose one internally consistent form: corrected moments with
the ordinary current learning rate, or uncorrected moments with the equivalent
corrected step-size form—never both.

### Formula / state anchor

For a parameter gradient `g_t`, the usual corrected-moment form is

`m_t = β₁m_(t-1) + (1-β₁)g_t`,  
`v_t = β₂v_(t-1) + (1-β₂)g_t²`,  
`m̂_t = m_t / (1-β₁ᵗ)`, `v̂_t = v_t / (1-β₂ᵗ)`,  
`parameter ← (1 - lr·weight_decay)·parameter - lr·m̂_t/(sqrt(v̂_t)+ε)`.

The persistent per-parameter state is `m`, `v`, and the step count; the
learning rate and bias-correction scalars are derived each step.

This formula is the ordinary AdamW minimization path. AMSGrad additionally
tracks a maximum second-moment state, and maximization reverses the update's
gradient direction; those options need their own corresponding handling.

### Concept Q&A

**Q: Why is `param_groups` a list?**  
**A:** A model may need distinct optimization settings for distinct parameter
sets, such as no weight decay on normalization scales.

**Q: Why track a count per parameter instead of one count per group?**  
**A:** A parameter with no gradient is skipped; its moment history should not
advance as though it received an update.

**Q: Should `alpha_t`/the learning rate be stored per parameter?**  
**A:** No. Read the group learning rate and recompute derived scalars each
step, since schedulers can change the learning rate.

**Q: Does AdamW add weight decay to the gradient?**  
**A:** No. Unlike coupled L2 regularization, AdamW performs a separate
parameter-shrinkage update.

**Q: Does the displayed AdamW formula cover AMSGrad and maximization?**  
**A:** No. It covers the ordinary minimization path; AMSGrad needs its running
maximum and maximization changes the gradient-update sign.

**Q: Why did the optimizer test fail before reaching `step()`?**  
**A:** The test expects an optimizer class/callable, then supplies model
parameters itself. Constructing an instance inside the adapter leaves it with
no parameters.

**Debug invariant:** after a step with zero gradient but nonzero weight decay,
the parameter shrinks; after a skipped (`grad is None`) parameter, that
parameter's moment count does not advance.

## 10. `get_lr_cosine_schedule`

### What was difficult / mistakes I made

- I initially treated learning rate as a fixed optimizer setting, which made
caching AdamW's derived step size seem reasonable.

### Durable model

The scheduler returns a learning rate for a given iteration: linear warmup
reaches the maximum rate, cosine annealing decreases it toward the minimum,
and iterations after the decay cycle stay at the minimum. The training loop
writes that current rate into the optimizer's relevant parameter group before
the next optimizer step.

### Formula / boundary anchor

With `T_w = warmup_iters`, `T_c = cosine_cycle_iters`, `α_max`, and `α_min`:

- before warmup ends: `α(t) = α_max · t / T_w`;
- from the warmup boundary through the cosine-cycle boundary:
  `α(t) = α_min + 0.5(α_max-α_min)[1 + cos(π(t-T_w)/(T_c-T_w))]`;
- after `T_c`: `α(t) = α_min`.

At `t = T_w`, both pieces give `α_max`; at `t = T_c`, cosine decay gives
`α_min`.

### Concept Q&A

**Q: Why must AdamW read `group["lr"]` every step?**  
**A:** The scheduler may have changed it since the preceding step.

**Q: What should happen after the cosine cycle?**  
**A:** The schedule should remain at its specified minimum learning rate.

**Debug invariant:** check the three boundaries explicitly: iteration 0 gives
zero during warmup, the warmup boundary gives the maximum, and the cycle end
and all later iterations give the minimum.

## 11. `save_checkpoint` and `load_checkpoint`

### What was difficult / mistakes I made

- I saved model and optimizer state but omitted the iteration counter.
- I restored both state dictionaries but left the loader with a bare `return`,
which returns `None`.
- The failed assertion `10 == None` exposed the API contract clearly: loading
must return the iteration that was saved.

### Durable model

A resumable training checkpoint contains the model state, optimizer state, and
the completed iteration count. Saving and loading are complementary: save all
three; load the first two into the objects supplied by the caller; return the
stored iteration so training can resume at the correct point.

### Concept Q&A

**Q: Why save the optimizer state as well as model weights?**  
**A:** AdamW's moments and per-parameter step counts affect the next update.
Restoring only weights does not faithfully resume training.

**Q: Why return the iteration instead of just restoring objects?**  
**A:** The outer training loop needs it for scheduling, logging, and choosing
the next iteration.

**Debug invariant:** a checkpoint round trip must preserve model state,
optimizer state, and the exact saved iteration.

## 12. UTF-8-aware byte-level tokenization and file chunking

### What was difficult / mistakes I made

- I first summarized UTF-8 as “one byte.”
- I understood that a string may need multiple units, but needed the precise
distinction: a Unicode code point may occupy multiple UTF-8 **bytes**.
- I needed to connect that fact to byte-based file boundaries and to recognize
that decoding with ignored errors can lose data silently.

### Durable model

UTF-8 is a variable-width encoding: ASCII code points use one byte, while
other Unicode code points use two to four bytes. Byte-level BPE uses UTF-8
because its byte vocabulary can represent arbitrary Unicode text losslessly.

An arbitrary file byte offset is not necessarily a Unicode-character boundary.
If chunking cuts through a multi-byte UTF-8 sequence, neither partial chunk can
faithfully decode the split code point without coordination. Each may still
decode its complete prefix/suffix. Suppressing decoding errors may make a
program continue, but it can discard parts of the original text. Chunk
boundaries must preserve complete UTF-8 sequences, or decoding must be handled
incrementally.

### Concept Q&A

**Q: Is UTF-8 one byte per character?**  
**A:** No. It is one to four bytes per Unicode code point; a user-visible
character can also consist of multiple code points.

**Q: Why is arbitrary byte chunking unsafe?**  
**A:** It can split one encoded code point between chunks.

**Q: Why is `errors="ignore"` not a real fix?**  
**A:** It hides the decoding failure by silently dropping data.

**Debug invariant:** for valid UTF-8 corpus input, encoding reconstructed
decoded text should reproduce the original bytes; if ignored errors made bytes
disappear, it will not.

## Cross-cutting testing and workflow lessons

- Read the adapter and the test name before assuming a test is absent. `pytest -k`
  matches test identifiers, not helper-function names used within test bodies.
- A collection failure is different from “no matching tests.” In this
  environment, a missing dependency (`jaxtyping`) prevented collection until
  the configured project environment was used.
- `git status` distinguishes untracked (`??`) source files from ignored (`!!`)
  generated files. A new module is not tracked until it is staged and committed.
- When a tensor test fails, start with semantic axes, shapes, and one tiny
  hand-checked example before changing formulas broadly.
