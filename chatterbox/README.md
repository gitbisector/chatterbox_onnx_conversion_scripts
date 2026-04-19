# Chatterbox Multilingual — ONNX Export (v2)

Four-graph ONNX export of [resemble-ai/chatterbox](https://github.com/resemble-ai/chatterbox)
aimed at high-throughput inference on ONNX Runtime with CUDA.  This is a
refactor of the original single-script export with three functional
additions: classifier-free guidance (CFG) baked into the language-model
graph, alignment-attention outputs used to force EOS, and a parameterised
CFM step count for the vocoder.

## What the export produces

| File | Purpose | Source class |
|------|---------|--------------|
| `speech_encoder.onnx` | Reference-audio → (cond_emb, prompt_tokens, x-vector, prompt_feat) | `PrepareConditionalsModel` |
| `embed_tokens.onnx` | (input_ids, position_ids, exaggeration) → inputs_embeds — batch=1 or batch=2 (CFG) | `InputsEmbeds` |
| `language_model.onnx` | T3 Llama backbone with CFG + alignment attention, fp16 | `LlamaForCFG` |
| `conditional_decoder.onnx` | (speech_tokens, speaker_embeddings, speaker_features) → waveform | `ConditionalDecoder` |

The `language_model.onnx` graph is new in v2.  The other three are
replacements for the upstream wrappers with behavior-compatible interfaces
plus the CFG / scatter-free improvements described below.

## What changed vs the upstream conversion script

### 1. Classifier-free guidance baked into the language model

The multilingual T3 is trained for CFG.  In v1 the LM graph was exported
batch=1 and callers had to run it twice (cond and uncond) and combine
logits outside the graph.  The v2 `language_model.onnx`:

- Accepts `inputs_embeds` of shape `(2, S, 1024)` — cond on row 0, uncond on row 1
- Takes a scalar `cfg_weight` (fp16)
- Returns `logits` of shape `(1, S, V)` computed as
  `cond + cfg_weight * (cond - uncond)` inside the graph

One LM call per step, one set of KV caches.

### 2. Alignment attention exposed as an output

Layers 9, 12, and 13 of the T3 Llama are the layers the PyTorch
`AlignmentStreamAnalyzer` consumes.  v2 emits their attention weights as
a new `attn_layers` output (shape `(3, num_heads, S, total_S)`), so an
ONNX-driven inference loop can reproduce Chatterbox's EOS-forcing
behavior.  Without this, short utterances hallucinate trailing speech
past the end of the input (cf.
[resemble-ai/chatterbox#97](https://github.com/resemble-ai/chatterbox/issues/97)).

`alignment_runtime.AlignmentStreamAnalyzer` is a pure-numpy port of the
upstream analyzer that runs on the attention output.

### 3. Scatter-free graphs

PyTorch's ONNX exporter emits `ScatterND` nodes for three patterns the
upstream export hits:

- `torch.zeros(...)` followed by conditional fills (in `InputsEmbeds`)
- `DynamicCache.update` (for KV writes)
- `scatter_` used for sparse residual merges (in the CFM stack)

ScatterND nodes prevent ORT CUDA graph capture, which is important for
throughput on long autoregressive decode loops.  Every v2 graph is
re-expressed using `Where`, `Mul`, `Add`, `Gather`, and `Concat`.

### 4. Parameterised CFM step count (vocoder)

The upstream decoder hardcodes `N=10` Euler steps.  v2 lets you export
`N=4`, `N=6`, or `N=10` by passing `n_cfm_timesteps=...` to
`ConditionalDecoder(...)`.  On an NVIDIA GB10 running the vocoder
accounts for the majority of wall-clock time; N=6 cuts it ~30 % with
negligible quality loss on a TTS→STT round-trip.

### 5. fp16 weights, fp32 numerically-sensitive islands

Language model weights + KV cache are fp16; softmax and attention scores
stay in fp32.  Embeddings are fp16.  Speech encoder and vocoder stay
fp32 (their bottlenecks aren't weight-bandwidth and they hit
ill-conditioned ops).

## Minimal usage

```python
# Export (once)
from chatterbox import chatterbox_to_onnx_conversion_script as conv

conv.export_model_to_onnx(
    multilingual=True,
    export_prepare_conditions=True,
    export_cond_decoder=True,
    audio_prompt_path="default_voice.wav",
    output_export_dir="converted",
)

# Inference
from chatterbox import chatterbox_multi_inference_script as run

run.run_inference(
    text="Bonjour, comment ça va?",
    language_id="fr",
    models_dir="converted",
    cfg_weight=0.5,
    temperature=0.8,
    output_file_name="output.wav",
)
```

The monolingual `chatterbox_inference_script.run_inference(...)` has the
same shape minus `language_id`.

## Benchmarks

Single-utterance latency on an NVIDIA DGX Spark (GB10, 128 GB unified
memory, sm_121, custom onnxruntime-gpu 1.24 aarch64 wheel) — 10-run mean
after warmup, CFM `N=6`, default voice, ~120 characters per sentence:

| Backend | Latency (mean) | Notes |
|---------|----------------|-------|
| PyTorch BF16 (upstream `ChatterboxMultilingualTTS`) | 3.00 s | BF16 mixed precision, CUDA graphs off |
| v2 ONNX (this repo) | 1.68 s | fp16 LM + fp32 vocoder, ORT CUDAExecutionProvider |

Quality gate: 4/4 pass on a TTS→STT similarity harness (Whisper
large-v3) across English / French / German / Dutch at ≥ 95 % word-level
similarity vs the source text.

The alignment-analyzer EOS fix has an additional effect not captured in
the benchmark above: the upstream ONNX (without alignment) frequently
runs to `max_new_tokens` on short inputs and tacks a few seconds of
hallucinated speech onto the end.  The v2 graph combined with the
analyzer stops cleanly at the EOS token, which is the user-visible
quality win.

## Caveats

- The LM is large (~1 GB of fp16 weights + external sidecar).  ORT
  `InferenceSession` creation takes 5–15 s cold on an NVMe disk.
- CUDA graph capture requires the process to stay on one KV-cache shape
  per decode-step size.  The inference scripts pre-warm the arena; in
  production you want to run a few warmup prompts so ORT has seen all
  the `past_len` extents it will encounter at steady state.
- `conditional_decoder_n{4,6,10}.onnx` are three separate files — pick
  one per deployment, they're not swappable at runtime.
- The alignment analyzer is deliberately conservative — it forces EOS
  once the attention head has firmly aligned past the end of the text
  span.  It will not save you from a model that hallucinates *before*
  the end of its input.

## License

MIT — same as the upstream repo.
