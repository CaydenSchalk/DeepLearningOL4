**[D-OD-1]** Getting how to calculate the parameter count to compare to previous model

**Why it matters**: Needed to know how PyTorch calculates Transformer parameters so I could configure it to match my lstm model's parameter count

**Outcome/decision**: Used the parameter formulas for the encoder, decoder, and LayerNorms, solving for dim_feedforward to hit the exact target size.

**Evidence**: "Assistant provides parameter formulas and solving script: 'f = (P_target - (4*Le + 8*Ld)*d^2 - (9*Le + 15*Ld + 4)*d) / ((2*d + 1)*(Le + Ld))'"

> Excerpt:
>
> User: what are the calculations for the parameter count of a standard transformer model in pytorch? i need to tweak it to match a previous model
>
> Assistant: For a standard PyTorch Transformer, parameter count is primarily determined by four things:
> d_model: hidden/embedding dimension
> dim_feedforward: feed-forward network width
> num_encoder_layers and num_decoder_layers: Transformer depth
> Whether you include token embeddings, output projections, biases, and final LayerNorms

---

**[D-DJ-2]** Keeping nhead fixed while tuning feedforward width

**Why it matters**: Needed to decide which hyperparameter to change to match model sizes without messing up the attention structure.

**Outcome/decision**: Decided to adjust dim_feedforward instead of nhead, since nhead just splits the existing d_model dimensions and does not change the parameter count.

**Evidence**: "Assistant says: 'The interesting part is that nhead does not directly change parameter count, provided d_model stays fixed. It changes how the attention dimensions are partitioned, not the total size of the projection matrices.'"

> Excerpt:
>
> Assistant: The interesting part is that nhead does not directly change parameter count, provided d_model stays fixed. It changes how the attention dimensions are partitioned, not the total size of the projection matrices.

---

**[W-CV-3]** Verifying why Transformer trains faster on parity

**Why it matters**: The assistant made conceptual claims about why Transformers optimize parity faster than LSTMs, which I needed to verify for the report discussion.

**Outcome/decision**: Confirmed the theoretical claim that attention provides shorter dependency paths and fits fixed-length count representations rather than sequential toggling, citing it in the reflection.

**Evidence**: "Assistant explains optimization difference: 'An LSTM has to learn to maintain and update a parity state sequentially, whereas a Transformer can use attention and parallel representations to learn a different computation of parity.'"

> Excerpt:
>
> User: why does a transformer do so much better than an lstm on the parity problem?
>
> Assistant: The main distinction is:
> An LSTM has to learn to maintain and update a parity state sequentially, whereas a Transformer can use attention and parallel representations to learn a different computation of parity.
> But there's a subtle catch: being better at fitting parity examples doesn't necessarily mean the Transformer has learned the underlying XOR algorithm.

---

**[C-FR-4]** Troubleshooting low validation accuracy on translation

**Why it matters**: The Transformer converged quickly during training but validation accuracy was barely improving over the LSTM baseline on English-Portuguese.

**Outcome/decision**: Diagnosed that the decoder was missing causal target masks and shifted input targets, and corrected the training step implementation to fix the validation failure.

**Evidence**: "Assistant identifies common failure modes: 'My first suspects would be masking, positional encoding, learning-rate scheduling, and dataset size.'"

> Excerpt:
>
> User: my validation accuracy isn't improving very much in the eng-por problem over the lstm when i use the transformer, anything common I should look for?
>
> Assistant: Yes. English Portuguese translation is a much different problem from parity, and there are several common reasons a Transformer might converge faster without producing much better validation accuracy than an LSTM.
> My first suspects would be masking, positional encoding, learning-rate scheduling, and dataset size.

---

**[C-CE-5]** Clarifying vocabulary parameter count

**Why it matters**: Needed to know how switching between character-level and word-level tokenizers affects the parameters of a transformer

**Outcome/decision**: Clarified that word-level tokenization spends most parameters on embedding tables, while character-level shifts the budget almost entirely to attention layers.

**Evidence**: "Assistant clarifies tokenizer parameter split: 'Word-level tokenization: Large vocabulary, short sequences. More parameters are spent on embeddings and the output classifier. Character-level tokenization: Small vocabulary, long sequences.'"

> Excerpt:
>
> User: i have 2 tokenization techniques, one is character-level while the other is word level, how does that affect the transformer's parameter budget compared to an lstm, and should i adjust the number of heads, d_model, or anything else? latent_size maybe?
>
> Assistant: The key distinction is:
> Word-level tokenization: Large vocabulary, short sequences. More parameters are spent on embeddings and the output classifier.
> Character-level tokenization: Small vocabulary, long sequences. Fewer embedding parameters, but substantially more computation is required to process each sentence.

---

**[D-EC-6]** Empirical calibration of quadratic character-level compute cost

**Why it matters**: I expected character-level tokenization to be lighter because of the tiny embedding table, but training was significantly slower and heavier.

**Outcome/decision**: Connected the empirical slowdown back to the O(T^2 * d) attention scaling over 5x longer sequences, adjusting batch size so it would run within compute limits.

**Evidence**: "Assistant warns about quadratic compute cost: 'If character tokenization increases sequence length by a factor of 5, the quadratic attention component can require roughly 25 times as much computation, at unchanged width.'"

> Excerpt:
>
> Assistant: For dense self-attention, the attention-score computation grows approximately as:
> O(T^2 * d)
> If character tokenization increases sequence length by a factor of 5, the quadratic attention component can require roughly 25 times as much computation, at unchanged width. Cross-attention similarly scales with source length times target length.