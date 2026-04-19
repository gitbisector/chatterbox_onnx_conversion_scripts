"""Export Chatterbox's input embedding layer (text/speech tokens + positions → embeddings).

The ``InputsEmbeds`` wrapper exports the text/speech token embedding path to a
single ONNX-traceable graph.  Given ``(input_ids, position_ids, exaggeration)``
it returns the per-token embeddings that feed into the T3 language model.

This version differs from the original monolithic export in three ways:

1. **Scatter-free**  — PyTorch's ONNX exporter emits ``ScatterND`` whenever it
   sees a ``torch.zeros(...)`` seed followed by conditional in-place fills
   (the pattern the original ``InputsEmbeds`` used).  ScatterND nodes
   prevent ORT CUDA-graph capture downstream.  This implementation computes
   every branch as a full tensor and combines them with elementwise masks,
   so the exporter emits only ``Where`` / ``Mul`` / ``Add`` / ``Gather``.
2. **Batch=2 CFG-aware**  — The multilingual T3 model is trained for
   classifier-free guidance (cond + uncond rows, with the uncond row's text
   embeddings zeroed).  The upstream export assumed batch=1; this one
   supports the two-row prefill directly in the graph, so an inference loop
   can run CFG without a second pass through the embedder.  Batch=1 callers
   are unaffected.
3. **fp16 weights**  — Halves memory footprint.  The exaggeration scalar
   input stays fp32 and is cast to fp16 immediately after the Linear.

Interface is identical to the upstream export (inputs
``input_ids, position_ids, exaggeration``; output ``inputs_embeds``), so the
driver in ``chatterbox_to_onnx_conversion_script.py`` needs no changes.
"""
import torch
import torch.nn as nn

from ._constants import START_SPEECH_TOKEN, EXAGGERATION_TOKEN


class InputsEmbeds(nn.Module):
    """Scatter-free, fp16, CFG-aware input embedder for Chatterbox.

    Replaces the original ``InputsEmbeds`` class.  Same external interface:
    constructed with the chatterbox model, called with
    ``(input_ids, position_ids, exaggeration)``.

    Key rewrite tricks:

    * Never seed a tensor with ``torch.zeros(...)`` and conditionally fill —
      the exporter traces that into ``ScatterND``.  Instead, compute every
      branch as a full tensor and combine with elementwise masks.
    * ``torch.where`` operates only on same-shape operands (no advanced
      indexing, which also becomes ``ScatterND``).
    * CFG row-1 zeroing uses ``torch.arange(B) == 1`` so the same graph works
      for both batch=1 and batch=2 callers.
    """

    def __init__(self, chatterbox):
        super().__init__()
        # Clone weights to fp16 in fresh modules so this wrapper is safe to drop
        # into any pipeline without side-effects on the shared chatterbox model.
        self.text_emb = self._clone_embedding_fp16(chatterbox.t3.text_emb)
        self.text_pos_emb = self._clone_embedding_fp16(chatterbox.t3.text_pos_emb.emb)
        self.speech_emb = self._clone_embedding_fp16(chatterbox.t3.speech_emb)
        self.speech_pos_emb = self._clone_embedding_fp16(chatterbox.t3.speech_pos_emb.emb)

        # emotion_adv_fc: (1, n_channels), no bias.  Keep weights in fp16 and
        # cast the fp32 scalar input to fp16 before the matmul.
        src = chatterbox.t3.cond_enc.emotion_adv_fc
        self.emotion_adv_fc = nn.Linear(
            src.in_features, src.out_features, bias=src.bias is not None
        ).to(torch.float16)
        with torch.no_grad():
            self.emotion_adv_fc.weight.copy_(src.weight.to(torch.float16))
            if src.bias is not None:
                self.emotion_adv_fc.bias.copy_(src.bias.to(torch.float16))

        self.start_speech_token = int(START_SPEECH_TOKEN)
        self.exaggeration_token = int(EXAGGERATION_TOKEN)

    @staticmethod
    def _clone_embedding_fp16(emb: nn.Embedding) -> nn.Embedding:
        out = nn.Embedding(emb.num_embeddings, emb.embedding_dim).to(torch.float16)
        with torch.no_grad():
            out.weight.copy_(emb.weight.to(torch.float16))
        return out

    def forward(
        self,
        input_ids: torch.Tensor,   # (B, S) int64
        position_ids: torch.Tensor,  # (B, S) int64
        exaggeration: torch.Tensor,  # (1,) float32
    ) -> torch.Tensor:
        """Returns (B, S, 1024) float16."""
        batch_size, seq_len = input_ids.shape

        # Build text / speech / exaggeration masks.  The first "0" in each row
        # marks the boundary between text prefix and speech suffix.  Rows
        # without a 0 are all-speech (matches autoregressive decode calls).
        idx = torch.arange(seq_len, device=input_ids.device, dtype=input_ids.dtype)
        idx = idx.unsqueeze(0).expand(batch_size, -1)  # (B, S)

        is_zero = input_ids == 0
        has_zero = is_zero.any(dim=1)  # (B,)
        # argmax on bool is unsupported by some ONNX opsets — cast to int64.
        first_zero = is_zero.to(torch.int64).argmax(dim=1)  # (B,), 0 if no zero
        minus_one = torch.full_like(first_zero, -1)
        zero_pos = torch.where(has_zero, first_zero, minus_one)  # (B,)

        exaggeration_mask = input_ids == self.exaggeration_token
        base_text_mask = (idx <= zero_pos.unsqueeze(1)) & has_zero.unsqueeze(1)
        text_mask = base_text_mask & ~exaggeration_mask
        speech_mask = ~base_text_mask & ~exaggeration_mask

        # Safe indices: mask ids to 0 on "off" positions so the lookup returns
        # a valid row that we then multiply by the mask (zeroing it) and sum
        # with the other branch.
        zero_idx = torch.zeros_like(input_ids)
        safe_text_ids = torch.where(text_mask, input_ids, zero_idx)
        safe_speech_ids = torch.where(speech_mask, input_ids, zero_idx)
        text_pos_ids = position_ids * text_mask.to(position_ids.dtype)
        speech_pos_ids = position_ids * speech_mask.to(position_ids.dtype)

        text_part = self.text_emb(safe_text_ids) + self.text_pos_emb(text_pos_ids)
        speech_part = self.speech_emb(safe_speech_ids) + self.speech_pos_emb(speech_pos_ids)

        text_mask_f = text_mask.to(torch.float16).unsqueeze(-1)
        speech_mask_f = speech_mask.to(torch.float16).unsqueeze(-1)
        exag_mask_f = exaggeration_mask.to(torch.float16).unsqueeze(-1)

        # CFG: zero the text portion of row 1 when B >= 2.  Rather than
        # branching on batch size (which the exporter would bake statically),
        # build a per-row scale: row 0 → 1.0, row 1 → 0.0, other → 1.0.
        # Works identically for B=1 (no scaling) and B=2 (CFG).
        row_ids = torch.arange(batch_size, device=input_ids.device)
        cfg_text_scale = torch.where(
            row_ids == 1,
            torch.zeros((), dtype=torch.float16, device=input_ids.device),
            torch.ones((), dtype=torch.float16, device=input_ids.device),
        ).view(batch_size, 1, 1)
        text_part = text_part * cfg_text_scale

        # Emotion-adv embedding for EXAGGERATION_TOKEN positions.
        exag_val = exaggeration.to(torch.float16).view(1, 1, 1)
        exag_embed = self.emotion_adv_fc(exag_val)  # (1, 1, D) → broadcasts

        # Combine: each position is exactly one of text/speech/exag/off, so
        # masked addition is correct and requires no scatter.
        out = (
            text_part * text_mask_f
            + speech_part * speech_mask_f
            + exag_embed * exag_mask_f
        )
        return out  # (B, S, D) fp16
