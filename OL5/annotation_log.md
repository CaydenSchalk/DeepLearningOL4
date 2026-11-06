**[W-1]** Learning differences between RNN and LSTM memory

**Why it matters**: Needed to understand the differences between a standard RNN and an LSTM

**Outcome/decision**: Learned how LSTMs use gates and a separate cell state to keep gradients from farther sequences flowing.

**Evidence**: "Assistant explains core difference: 'Vanilla RNN: Maintains a single hidden state, which is repeatedly overwritten as new words arrive. Earlier information can become difficult to preserve or learn from ... LSTM: Maintains a separate cell state (long-term memory) and uses learned gates to decide what to keep, forget, and expose."

> Excerpt:
>
> User: What exactly is the difference between an RNN and LSTM?
>
> Assistant: The fundamental difference between an RNN (Recurrent Neural Network) and an LSTM (Long Short-Term Memory) is how they store, update, and forget information over time.
> An LSTM is actually a special type of RNN, designed to solve one of the biggest problems with traditional RNNs: difficulty learning long-term dependencies.

---

**[FR-2]** Fixing shape mismatch on LSTM dims

**Why it matters**: Switching from an RNN to an LSTM in the translation model caused a shape mismatch error that wouldn't allow execution.

**Outcome/decision**: Ended up fixing the error and was able to continue execution.

**Evidence**: "Assistant clarifies dimension error: 'No a unidirectional LSTM does not require a larger hidden dimension than an RNN. If both have hidden_size = 64, both produce outputs with 64 features.'"

> Excerpt:
>
> User: Compared to the RNN layer, I'm getting a runtime error:
> RuntimeError: The size of tensor a (64) must match the size of tensor b (128) at non-singleton dimension 2
> On the eng->por problem, where should I look? Is it because the LSTM requires a larger dimension, or the opposite? The LSTM is not bi-directional.
>
> Assistant: No a unidirectional LSTM does not require a larger hidden dimension than an RNN. If both have hidden_size = 64, both produce outputs with 64 features.
> Your error:
> RuntimeError: The size of tensor a (64) must match the size of tensor b (128) at non-singleton dimension 2
> strongly suggests that two tensors with different feature dimensions are being combined somewhere in your encoder-decoder architecture.

---

**[D-OD-3]** Finding custom change ideas for parity problem

**Why it matters**: Needed ideas for the custom component of the assignment, specifically on the parity problem.

**Outcome/decision**: Identified the option I ended up going with, that being an auxiliary task for the encoder.

**Evidence**: "Assistant suggests: '1. Auxiliary prefix-parity prediction head: Makes the encoder learn parity at every time step instead of only receiving supervision at the end. 2. Learnable parity-gated memory cell: Builds the idea of retaining or flipping state into the architecture.'"

> Excerpt:
>
> User: I need to use an encoder/decoder architecture to work on the parity problem. I already have an architecture mapped out, but are there any custom improvements I should use that would help?
>
> Assistant: This suggests two especially promising custom components:
> 1. Auxiliary prefix-parity prediction head: Makes the encoder learn parity at every time step instead of only receiving supervision at the end.
> 2. Learnable parity-gated memory cell: Builds the idea of retaining or flipping state into the architecture.

---

**[D-DJ-4]** Decided to pick the prefix head over the memory cell

**Why it matters**: Needed to pick a good custom component without rewriting the entire architecture, in order to not change too much.

**Outcome/decision**: Ended up for sure selecting the auxiliary task for the encoder.

**Evidence**: "Assistant explains justification: 'It requires relatively few additional parameters, preserves your encoder–decoder structure, and directly targets the problem of retaining parity information across time.'"

> Excerpt:
>
> Assistant: Why I like this solution: It requires relatively few additional parameters, preserves your encoder-decoder structure, and directly targets the problem of retaining parity information across time.

---

**[D-EC-5]** Masking padding in aux loss calculation

**Why it matters**: Variable-length sequences have padding tokens that would give false targets, they also have specific token values such as <START> and <STOP>, so those need to be masked out.

**Outcome/decision**: Added a float length mask to the binary-cross entropy

**Evidence**: "Assistant provides masking code: 'First: If you're using padded variable-length sequences, mask the auxiliary loss so padding positions don't contribute'"

> Excerpt:
>
> Assistant: First: If you're using padded variable-length sequences, mask the auxiliary loss so padding positions don't contribute:
> 
> per_token_loss = F.binary_cross_entropy_with_logits(
>     prefix_logits,
>     prefix_targets,
>     reduction="none"
> )
> # lengths: [batch], number of real bits per sequence
> mask = (
>     torch.arange(x.size(1), device=x.device)[None, :]
>     < lengths[:, None]
> )
> aux_loss = (
>     (per_token_loss * mask).sum()
>     / mask.sum().clamp_min(1)
> )

---

**[W-6]** Explaining why LSTM performed worse on the parity problem

**Why it matters**: Needed to understand why the RNN performed better on the parity problem than the LSTM when both had about the same param count.

**Outdome/decision**: Learned that under equal params, simple RNNs have larger hidden sizes, and LSTM gating was built for selective retention.

**Evidence**: "Assistant explains: 'The most likely explanation is that parity requires learning an exact state-transition rule, whereas LSTMs are primarily designed to make selective memory retention easier'"

> Excerpt:
>
> User: On the parity problem, the LSTM did fairly worse than the simple RNN with about the same parameters. Why could that be?
>
> Assistant: That's a particularly interesting result, and it's entirely plausible for an LSTM to perform significantly worse than a simple RNN on the parity problem, even when both have approximately the same number of parameters.
> The most likely explanation is that parity requires learning an exact state-transition rule, whereas LSTMs are primarily designed to make selective memory retention easier.