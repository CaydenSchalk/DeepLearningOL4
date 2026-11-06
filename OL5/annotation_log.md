**[D-OD-1]** Getting how to calculate the parameter count to compare to previous model

**Why it matters**: Needed to know how PyTorch gets transformer parameters so I could change it to match my lstm model's parameter count.

**Outcome/decision**: Used the parameter formulas as a reference to see what I needed to tweak exactly.

**Evidence**: "Assistant gives param formulas: 'f = (P_target - (4*L_e + 8*L_d) ... / ((2*d + 1)*(L_e + L_d))'"

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

**[D-DJ-2]** Tuning model and noting not to change

**Why it matters**: Needed to decide which hyperparameter to change to match model sizes without messing up the model entirely.

**Outcome/decision**: Left n_head alone and d_model as well and dim_feedforward. I matched as close as possible to the LSTM's parameter count by changing layer counts and the vocab size.

**Evidence**: "Assistant says: 'The interesting part is that nhead does not directly change parameter count, provided d_model stays fixed. It changes how the attention dimensions are partitioned, not the total size of the projection matrices.'"

> Excerpt:
>
> Assistant: The interesting part is that nhead does not directly change parameter count, provided d_model stays fixed. It changes how the attention dimensions are partitioned, not the total size of the projection matrices.

---

**[W-CV-3]** Seeing why Transformer trains faster/better on the parity problem

**Why it matters**: I wanted to understand why exactly the transformer had an easier time on the parity problem.

**Outcome/decision**: I now understand why the transformer excels at the task, due to the transformer being able to use self-attention and not having to keep track of state like a RNN or LSTM.

**Evidence**: "Assistant explains difference: 'An LSTM has to learn to maintain and update a parity state sequentially, whereas a Transformer can use attention and parallel representations to learn a different computation of parity.'"

> Excerpt:
>
> User: why does a transformer do so much better than an lstm on the parity problem?
>
> Assistant: The main distinction is:
> An LSTM has to learn to maintain and update a parity state sequentially, whereas a Transformer can use attention and parallel representations to learn a different computation of parity.
> But there's a subtle catch: being better at fitting parity examples doesn't necessarily mean the Transformer has learned the underlying XOR algorithm.

---

**[C-FR-4]** Troubleshooting low validation accuracy on the translation problem

**Why it matters**: The transformer wasn't training very well at all, but training was really fast.

**Outcome/decision**: After noting the AI said dataset size, I passed n_sequences = None in order to use the entire dataset.

**Evidence**: "Assistant says: 'My first suspects would be masking, positional encoding, learning-rate scheduling, and dataset size.'"

> Excerpt:
>
> User: my validation accuracy isn't improving very much in the eng-por problem over the lstm when i use the transformer, anything common I should look for?
>
> Assistant: Yes. English Portuguese translation is a much different problem from parity, and there are several common reasons a Transformer might converge faster without producing much better validation accuracy than an LSTM.
> My first suspects would be masking, positional encoding, learning-rate scheduling, and dataset size.

---

**[C-CE-5]** Learning more about vocabulary parameter count

**Why it matters**: Wanted to know how switching between character-level and word-level tokenizers affects the parameters of a transformer

**Outcome/decision**: Understood that word-level tokenization spends more parameters on embeddings.

**Evidence**: "Assistant says: 'Word-level tokenization: Large vocabulary, short sequences. More parameters are spent on embeddings and the output classifier. Character-level tokenization: Small vocabulary, long sequences.'"

> Excerpt:
>
> User: i have 2 tokenization techniques, one is character-level while the other is word level, how does that affect the transformer's parameter budget compared to an lstm, and should i adjust the number of heads, d_model, or anything else? latent_size maybe?
>
> Assistant: The key distinction is:
> Word-level tokenization: Large vocabulary, short sequences. More parameters are spent on embeddings and the output classifier.
> Character-level tokenization: Small vocabulary, long sequences. Fewer embedding parameters, but substantially more computation is required to process each sentence.

---

**[D-DJ-6]** Choosing word-level tokenization instead of character-level tokenization

**Why it matters**: The sequence grows a lot longer with character-level tokenization over word-level

**Outcome/decision**: Kept the word-level tokenization obtained from the example code in the Transformers module lesson

**Evidence**: "Assistant talks about costs: 'If character tokenization increases sequence length by a factor of 5, the quadratic attention component can require roughly 25 times as much computation, at unchanged width.'"

> Excerpt:
>
> Assistant: For dense self-attention, the attention-score computation grows approximately as:
> O(T^2 * d)
> If character tokenization increases sequence length by a factor of 5, the quadratic attention component can require roughly 25 times as much computation, at unchanged width. Cross-attention similarly scales with source length times target length.